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



from sentence_transformers import SentenceTransformer, util
import torch
from langchain.tools import Tool  

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(r'./paraphrase-multilingual-MiniLM-L12-v2', device=device)

#tags = get_tags()
#result = {tag['id']: tag.get('description', 'No Description Available') for tag in tags}
#print(result)
#desc_dict = {key: value for key, value in result.items() if value and value != "No Description Available"}
#print(desc_dict)

#desc_dict = {92: "引风机转速",94: "蒸汽阀门",95: "连续吹扫",96: "吹灰器蒸汽",97: "主蒸汽流量",98: "连续吹扫",99: "给水流量",100: "主蒸汽压力",101: "汽包水位",102: "饱和蒸汽电导率",103: "主蒸汽电导率",105: "一次风量",106: "二次风量",107: "三次风量",108: "炉膛压力",109: "烟囱氧残留量",110: "DNCG流量",111: "黑液流量",112: "黑液干固物",113: "左侧省煤器出口温度",114: "主蒸汽流量",115: "给水箱电导率",116: "左侧省煤器出口温度",117: "吹灰器蒸汽",118: "给水流量",119: "主蒸汽压力",120: "汽包水位",121: "一次风量",122: "二次风量",124: "炉膛压力",125: "引风机转速",126: "烟囱氧残留量",127: "DNCG流量",128: "黑液流量",129: "黑液干固物",130: "主蒸汽电导率",131: "蒸汽阀门",132: "蒸汽阀门",133: "主蒸汽流量",134: "连续吹扫",248: "饱和蒸汽电导率"}

desc_dict = {
    # 核心磨浆机参数
    30: '废料磨浆机 A 流量',
    31: '废料磨浆机 B 出口流量',
    46: '废料磨浆机 A 出口压力',
    47: '废料磨浆机 B 出口压力',
    57: '废料磨浆机 A 出口温度',
    58: '废料磨浆机 B 出口温度',
    66: '废料磨浆机功率监控',
    
    # 关键槽体与输送
    1490: '未精制废料槽液位',
    1492: '精磨后进料浓度',
    1493: '废料磨浆机比能量',
    
    
    # 主要化学品参数
    375: '1# 氢氧化钠储槽液位',
    392: '1# 过氧化氢储槽液位',
    1257: '浸渍 1 氢氧化钠浓度',
    1271: '浸渍 2 过氧化氢浓度',
    1288: '磨浆机 1 过氧化氢浓度',
    
    # 总流量与产量数据
    1302: '新鲜水总流量',
    1350: '过氧化氢总添加流量',
    1351: '氢氧化钠总添加流量',
    11654: ' 实际产量 ',
    11655: ' 修正产量 ',
    
    # 质量检测数据
    11971: ' 化验室白度数据 ',
    11972: ' 化验室游离度数据 ',
    11973: ' 化验室抗张指数数据 ',
    
    # 设备运行状态
    11610: ' 磨浆机 1 运行状态 '
}


def match_chinese_semantics(input_str: str) -> dict:
    """
    中文语义相似度匹配函数：根据输入字符串匹配字典中语义相近的描述
    
    参数：
        input_str: 字符串类型，待匹配的中文字符串，如 "蒸汽相关的参数"
    
    返回：
        字典类型，格式为 {id: 中文描述}
    """
    if not input_str.strip():
        return {}
    input_embedding = model.encode(input_str, convert_to_tensor=True)
    descriptions = list(desc_dict.values())
    ids = list(desc_dict.keys())
    desc_embeddings = model.encode(descriptions, convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, desc_embeddings).flatten()
    result = {}
    threshold=0.5
    for i in range(len(cos_scores)):
        score = cos_scores[i].item()
        if score > threshold:
            result[ids[i]] = descriptions[i]
    return result


get_sim_ID_tool = Tool(
    name="match_chinese_semantics",
    func=match_chinese_semantics,
    description="""用于根据输入的中文字符串，从指定字典中匹配语义相似的条目。
    必须满足以下要求：
    1. 输入参数必须包含：
       - input_str：字符串，待匹配的中文内容（如"蒸汽相关的参数"）
    2. 函数内部会自动计算语义相似度，返回所有相似度超过阈值0.7结果。
    示例：
        输入： input_str="蒸汽相关参数"
        输出：{94: "蒸汽阀门", 97: "主蒸汽流量", 100: "主蒸汽压力"}
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

