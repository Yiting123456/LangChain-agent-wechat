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

print(get_tags())


def get_tag_values(ids:int) -> dict:
    """ 查询指定标签的值，输入标签ID列表(整数)，输出标签值字典。\
    示例: 输入5，返回[{'tagID': 5, 'value': 94.85502624511719, 'timestamp': '2025-04-01T16:48:10.848Z', 'quality': 192}]
        """
    metris_info, token, headers = get_metris_token()  
    METRIS_URI = metris_info["base_url"]  
    tag_values_uri = f'{METRIS_URI}/api/historian/v02/tagvalues'
    params = {'ids': [ids]}
    response = requests.get(tag_values_uri, headers=headers, params=params, verify=False)
    return response.json() 


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



get_tag_values_tool = Tool(
    name="get_tag_values",
    func=get_tag_values,
    description="""用于查询单个标签ID对应的标签值。
    必须满足以下要求：
    1. 输入参数必须是**单个整数**（例如 5、10 等），不可传入列表、字符串或其他格式
    2. 函数内部会自动将该整数转换为列表格式供接口使用，无需手动处理
    示例：查询ID为5的标签时，直接传入 5 即可，返回结果为包含该标签详细信息的字典/列表（如 {'tagID': 5, 'value': 94.855...}）
    """
)
