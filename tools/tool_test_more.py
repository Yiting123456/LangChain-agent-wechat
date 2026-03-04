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



from sentence_transformers import SentenceTransformer, util
import torch
from langchain.tools import Tool  

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(r'./paraphrase-multilingual-MiniLM-L12-v2', device=device)

desc_dict = {
    92: "引风机转速",
    94: "蒸汽阀门",
    95: "连续吹扫",
    96: "吹灰器蒸汽",
    97: "主蒸汽流量",
    98: "连续吹扫",
    99: "给水流量",
    100: "主蒸汽压力",
    101: "汽包水位",
    102: "饱和蒸汽电导率",
    103: "主蒸汽电导率",
    105: "一次风量",
    106: "二次风量",
    107: "三次风量",
    108: "炉膛压力",
    109: "烟囱氧残留量",
    110: "DNCG流量",
    111: "黑液流量",
    112: "黑液干固物",
    113: "左侧省煤器出口温度",
    114: "主蒸汽流量",
    115: "给水箱电导率",
    116: "左侧省煤器出口温度",
    117: "吹灰器蒸汽",
    118: "给水流量",
    119: "主蒸汽压力",
    120: "汽包水位",
    121: "一次风量",
    122: "二次风量",
    124: "炉膛压力",
    125: "引风机转速",
    126: "烟囱氧残留量",
    127: "DNCG流量",
    128: "黑液流量",
    129: "黑液干固物",
    130: "主蒸汽电导率",
    131: "蒸汽阀门",
    132: "蒸汽阀门",
    133: "主蒸汽流量",
    134: "连续吹扫",
    248: "饱和蒸汽电导率"
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

#tags = get_tags()
#result = {tag['id']: tag.get('description', 'No Description Available') for tag in tags}
#desc_dict_English = {key: value for key, value in tags.items() if value and value != "No Description Available"}
desc_dict_English = {
    92: 'ID_fan_speed',
    94: 'Steam valve',
    95: 'Continous_purge',
    96: 'Sootblower_steam',
    97: 'Main_steam_flow',
    99: 'Feedwater_flow',
    100: 'Main_steam_pressure',
    101: 'Drum_level',
    103: 'Main_steam_conductivity',
    105: 'Primary_air_flow',
    106: 'Secondary_air_flow',
    107: 'Tertiary_air_flow',
    108: 'Furnace_pressure',
    109: 'O2_residual_stack',
    110: 'DNCG_flow',
    111: 'Black_liquor_flow',
    112: 'Black_liquor_dry_solids',
    113: 'Eco2_temperature_left',
    114: 'Main_steam_flow',
    115: 'Feedwater_tank_conductivity.',
    117: 'Sootblower_steam',
    118: 'Feedwater_flow',
    119: 'Main_steam_pressure',
    120: 'Drum_level',
    121: 'Primary_air_flow',
    122: 'Secondary_air_flow',
    124: 'Furnace_pressure',
    125: 'ID_fan_speed',
    126: 'O2_residual_stack',
    127: 'DNCG_flow',
    128: 'Black_liquor_flow',
    129: 'Black_liquor_dry_solids',
    130: 'Main_steam_conductivity',
    131: 'Steam valve',
    132: 'Steam valve',
    133: 'Main_steam_flow',
    134: 'Continous_purge',
    142: 'Test Batch File Tag',
    248: 'Saturated_steam_conductivity'
}

#print(desc_dict_English)

def match_English_semantics(input_str: str) -> dict:
    """
    英文语义相似度匹配函数：根据输入字符串匹配字典中语义相近的描述
    
    参数：
        input_str: 字符串类型，待匹配的英文字符串，如 "Related steam parameters"
    
    返回：
        字典类型，格式为 {id:英文描述}
    """
    if not input_str.strip():
        return {}
    input_embedding = model.encode(input_str, convert_to_tensor=True)
    descriptions = list(desc_dict_English.values())
    ids = list(desc_dict_English.keys())
    desc_embeddings = model.encode(descriptions, convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, desc_embeddings).flatten()
    result = {}
    threshold=0.5
    for i in range(len(cos_scores)):
        score = cos_scores[i].item()
        if score > threshold:
            result[ids[i]] = descriptions[i]
    return result

get_sim_ID_tool_Chinse = Tool(
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


get_sim_ID_tool_English = Tool(
    name="match_English_semantics",
    func=match_English_semantics,
    description="""用于根据输入的英文字符串，从指定字典中匹配语义相似的条目。
    必须满足以下要求：
    1. 输入参数必须包含：
       - input_str：字符串，待匹配的英文内容（如"Related steam parameters"）
    2. 函数内部会自动计算语义相似度，返回所有相似度超过阈值0.7结果。
    示例：
        输入： input_str="Related steam parameters"
        输出：{ 97: 'Main_steam_flow', 100: 'Main_steam_pressure',}
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


