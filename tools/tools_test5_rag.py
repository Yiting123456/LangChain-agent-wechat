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
import numpy as np
import re


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
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

_IDS = list(desc_dict.keys())
_TEXTS = [desc_dict[i] for i in _IDS]
_EMB = model.encode(_TEXTS, convert_to_numpy=True)

_EMB = _EMB / np.linalg.norm(_EMB, axis=1, keepdims=True)

if _FAISS_AVAILABLE:
    _DIM = _EMB.shape[1]
    _INDEX = faiss.IndexFlatIP(_DIM)  
    _INDEX.add(_EMB.astype('float32'))
else:
    _INDEX = None  

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")

def _tokenize(text: str) -> set:
    if not text:
        return set()
    return set(_TOKEN_PATTERN.findall(text.lower()))

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _hybrid_retrieve(query: str, top_k: int = 5, alpha: float = 0.75):
    """
    混合检索：语义分数*(alpha) + 关键词Jaccard*(1-alpha)
    返回 [{'id':..., 'text':..., 'score':...}]
    """
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    cand_k = min(top_k * 2, len(_IDS))  
    if _INDEX is not None:
        scores, idxs = _INDEX.search(q_emb.astype('float32'), cand_k)
        sem_scores = scores[0]
        sem_idxs = idxs[0]
    else:
        sem_scores = (_EMB @ q_emb.T).ravel() 
        sem_idxs = np.argsort(-sem_scores)[:cand_k]

    q_tokens = _tokenize(query)
    fused = []
    for j in sem_idxs:
        kw_score = _jaccard(q_tokens, _tokenize(_TEXTS[j]))
        score = float(alpha * sem_scores[j] + (1.0 - alpha) * kw_score)
        fused.append({"id": _IDS[j], "text": _TEXTS[j], "score": score})

    fused = sorted(fused, key=lambda x: x["score"], reverse=True)[:top_k]
    return fused

def match_tag_semantics(input_str: str, top_k: int = 3) -> dict:
    q = (input_str or "").strip()
    if not q:
        return {}

    results = _hybrid_retrieve(q, top_k=max(1, top_k), alpha=0.75)
    lines = [f"[TagID={r['id']}] {r['text']}" for r in results]
    context = "\n".join(lines)

    return {
        "query": q,
        "results": [
            {"id": r["id"], "description": r["text"], "score": round(float(r["score"]), 6)}
            for r in results
        ],
        "context": context,
        "notes": "分数为语义(α=0.75)与关键词Jaccard(1-α)的线性融合；向量余弦已归一化处理。"
    }


from langchain.tools import Tool

match_tag_tool = Tool(
    name="match_tag_semantics",
    func=match_tag_semantics,
    description="""
    用于根据用户输入的自然语言（中英文均可），匹配最相关的 Tag 描述，并返回对应的 ID。
    输入参数：
      - input_str: 用户的问题或描述（如“精筛出口流量设定值”或 “Outlet flow setpoint”）
    输出：
      - 一个字典，包含用户输入的问题，描述对应的 Top 3 个最相关的 Tag ID 及其描述结果，以及检索上下文信息，和备注说明
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
