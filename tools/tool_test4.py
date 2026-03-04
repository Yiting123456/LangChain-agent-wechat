import csv
import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.tools import Tool
load_dotenv()

METRIS_URI = os.getenv('METRIS_URI') 
METRIS_USERNAME = os.getenv('METRIS_USERNAME')
METRIS_PASSWORD = os.getenv('METRIS_PASSWORD') 
#fig_path = os.getenv('PIC_OUTPUT') 



def get_metris_token():
    auth_data = {"username": METRIS_USERNAME, "password": METRIS_PASSWORD}
    auth_uri = f"{METRIS_URI}/api/account/authenticate"
    response = requests.post(auth_uri, json=auth_data, verify=False)  
    token_data = response.json()  
    token = token_data.get("id") 
    headers = {"Authorization": f"Bearer {token}"}  
    return {"base_url": METRIS_URI}, token, headers


def get_tags():
    metris_info, token, headers = get_metris_token()  
    METRIS_URI = metris_info["base_url"]  
    tags_uri = f'{METRIS_URI}/api/configuration/tags'
    response = requests.get(tags_uri, headers=headers, verify=False)
    return response.json()

#print(get_tags())

def get_tag_values(ids:int) -> dict:
    """ 查询指定标签的值，输入标签ID列表(整数)，输出标签值字典。\
    示例: 输入5，返回字典 {'tagID': 5, 'value': 94.85502624511719, 'timestamp': '2025-04-01T16:48:10.848Z', 'quality': 192}
    异常:
        ValueError: 输入参数无效时抛出
        ConnectionError: 网络请求失败时抛出
        RuntimeError: 接口返回非预期结果时抛出
        """
    try:
        metris_info, token, headers = get_metris_token()  
        METRIS_URI = metris_info["base_url"]  
        tag_values_uri = f'{METRIS_URI}/api/historian/v02/tagvalues'
        params = {'ids': [ids]}
        response = requests.get(tag_values_uri, headers=headers, params=params, verify=False)
        result = response.json()
        if not isinstance(result, list) or len(result) == 0:
            raise RuntimeError("接口返回空结果或非预期格式")
        return result[0]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"网络请求失败: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"处理请求时发生错误: {str(e)}") from e


def fix_trend_value(value: dict) -> dict:
    if 't' in value:
        value['x'] = value['t']
        del value['t']
    
    if 'v' in value:
        value['y'] = value['v']
        del value['v']
    elif 'st' in value:
        value['y'] = value['st']
        del value['st']
        
    return {
        'y': 0.0,
        **value
    }

def fix_trend_values(values: list) -> list:
    values = [fix_trend_value(v) for v in values]
    for d in values:
        d.update((k, datetime.fromtimestamp(v / 1000).isoformat()) for k, v in d.items() if k == "x")
    values = sorted(values, key=lambda v: v['x'])
    return values

def get_trend_values(ids):
    metris_info, token, headers = get_metris_token()  
    METRIS_URI = metris_info["base_url"]
    result = {}

    for tag_id in ids:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=15)

        trend_params = {
            'tagid': tag_id,
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'timeshift': 0,
            'interpolationmethod': 1,
            'interpolationresolution': 1080,
            'interpolationresolutiontype': 0,
            'aggregatefunction': 0,
            'trackingreferencestep': None
        }

        trend_uri = f'{METRIS_URI}/api/historian/v02/trendvalues'

        response = requests.get(trend_uri, headers=headers, params=trend_params, verify=False)
        
        if response.status_code == 200:
            try:
                data = fix_trend_values(response.json())
                print("返回数据：", data)
                result[tag_id] = data
            except Exception as e:
                print("解析 JSON 失败：", e)
                result[tag_id] = {'error': 'Invalid JSON'}
        else:
            result[tag_id] = {'error': f"Failed to retrieve data for tag_id {tag_id}"}

    return result


def get_tags_by_name(tag_names):
    all_tags = get_tags()  
    return [tag for tag in all_tags if tag['name'] in tag_names]  



csv_file = r"C:\Users\Administrator\yt\lc\RuiFeng_Datas.csv"

desc_dict = {}

with open(csv_file, mode='r', encoding='utf-16') as file:
    reader = csv.DictReader(file, delimiter=';')
    for row in reader:
        try:
            tag_id = int(row['ID'])
            tag = row['Tag'].strip()
            description = row['Description'].strip()
            desc_dict[tag_id] = f"{tag} {description}"
        except (ValueError, KeyError):
            continue


from sentence_transformers import SentenceTransformer, util
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

def match_tag_semantics(input_str: str, top_k: int = 3) -> dict:
    """
    根据输入字符串匹配语义最相近的 Tag 描述，返回 Top-K 个最相关的 ID 和描述。
    """
    if not input_str.strip():
        return {}

    input_embedding = model.encode(input_str, convert_to_tensor=True)
    descriptions = list(desc_dict.values())
    ids = list(desc_dict.keys())
    desc_embeddings = model.encode(descriptions, convert_to_tensor=True)

    cos_scores = util.cos_sim(input_embedding, desc_embeddings).flatten()
    top_results = torch.topk(cos_scores, k=top_k)

    result = {}
    for score, idx in zip(top_results.values, top_results.indices):
        result[ids[idx]] = descriptions[idx]

    return result

from langchain.tools import Tool

match_tag_tool = Tool(
    name="match_tag_semantics",
    func=match_tag_semantics,
    description="""
    用于根据用户输入的自然语言（中英文均可），匹配最相关的 Tag 描述，并返回对应的 ID。
    输入参数：
      - input_str: 用户的问题或描述（如“精筛出口流量设定值”或 “Outlet flow setpoint”）
    输出：
      - 一个字典，包含 Top 3 个最相关的 Tag ID 及其描述
    """
)

get_tag_values_tool = Tool(
    name="get_tag_values",
    func=get_tag_values,
    description="""用于查询单个标签ID对应的标签值。
    必须满足以下要求：
    1. 输入参数必须是**单个整数**（例如 5、10 等），不可传入列表、字符串或其他格式
    2. 函数内部会自动将该整数转换为列表格式供接口使用，无需手动处理
    示例：查询ID为5的标签时，直接输入 5 即可，返回结果为包含该标签详细信息的字典{'tagID': 5, 'value': 94.855...}
    """
)
