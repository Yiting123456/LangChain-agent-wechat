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

def get_tag_values(ids:int) -> dict:
    """ 查询指定标签的值，输入标签ID列表(整数)，输出标签值字典。\
    示例: 输入5，返回字典 {'tagID': 5, 'value': 94.85502624511719, 'timestamp': '2025-04-01T16:48:10.848Z', 'quality': 192}
    异常:
        ValueError: 输入参数无效时抛出
        ConnectionError: 网络请求失败时抛出
        RuntimeError: 接口返回非预期结果时抛出
        """
    try:
        auth_data = {"username": "Yiting", "password": "Metris123*"}
        auth_uri = "https://172.22.89.119:9000/api/account/authenticate"
        response = requests.post(auth_uri, json=auth_data, verify=False)  
        token_data = response.json()  
        token = token_data.get("id") 
        headers = {"Authorization": f"Bearer {token}"}   
        tag_values_uri = 'https://172.22.89.119:9000/api/historian/v02/tagvalues'
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


# print(get_tag_values(5))



# -*- coding: utf-8 -*-
import os
import json
from typing import List, Union, Optional, Dict, Any, Tuple

import requests
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError
import warnings
from urllib3.exceptions import InsecureRequestWarning

API_BASE       = os.getenv("API_BASE", "https://4dd6651edf6c.ngrok-free.app")
AUTH_URI       = os.getenv("AUTH_URI",       f"{API_BASE}/api/account/authenticate")
TAG_VALUES_URI = os.getenv("TAG_VALUES_URI", f"{API_BASE}/api/historian/v02/tagvalues")

AUTH_TIMEOUT   = float(os.getenv("AUTH_TIMEOUT", "10"))
DATA_TIMEOUT   = float(os.getenv("DATA_TIMEOUT", "15"))

AUTH_USERNAME  = os.getenv("AUTH_USERNAME", "Yiting")
AUTH_PASSWORD  = os.getenv("AUTH_PASSWORD", "Metris123*")

USE_REPEATED_IDS_PARAM = os.getenv("USE_REPEATED_IDS_PARAM", "true").lower() == "true"

_verify_env = os.getenv("VERIFY_TLS", "false")
if _verify_env.lower() in ("true", "false"):
    VERIFY_OPT: Union[bool, str] = (_verify_env.lower() == "true")
else:
    VERIFY_OPT = _verify_env  

DEBUG_LOG      = os.getenv("DEBUG_LOG", "true").lower() == "true"

if VERIFY_OPT is False:
    warnings.simplefilter("ignore", InsecureRequestWarning)


def dlog(*args: Any) -> None:
    """调试打印，仅在 DEBUG_LOG=True 时输出"""
    if DEBUG_LOG:
        print("[CODE-DEBUG]", *args)


def ensure_object(maybe_json: Any) -> Dict[str, Any]:
    """
    将输入转换为 dict：
    - 如果是 dict，直接返回
    - 如果是 JSON 字符串（可能带 ``` 包裹），解析后返回
    """
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        s = maybe_json.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json\n"):
                s = s[5:]
        return json.loads(s)
    raise ValueError("输入既非对象也非字符串，无法解析成 JSON")


def normalize_ids(ids: Optional[Union[int, str, List[Union[int, str]]]]) -> List[int]:
    """
    将各种形式的 ids 统一成整数列表；空值返回 [1]
    支持：单值 int/str、逗号分隔字符串、列表（元素为数字或字符串数字）
    """
    if ids is None or ids == "null":
        return [1]
    if isinstance(ids, int):
        return [ids]
    if isinstance(ids, str):
        s = ids.strip()
        if not s:
            return [1]
        out: List[int] = []
        for p in s.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                out.append(int(p))
            except Exception:
                continue
        return out or [1]
    if isinstance(ids, list):
        out: List[int] = []
        for p in ids:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out or [1]
    return [1]


def build_params(tag_ids: List[int]) -> Union[Dict[str, str], List[Tuple[str, int]]]:
    """构建 GET 查询参数：允许重复则 ids=1&ids=2&ids=3，否则 ids=1,2,3"""
    if USE_REPEATED_IDS_PARAM:
        return [("ids", i) for i in tag_ids]
    else:
        return {"ids": ",".join(str(i) for i in tag_ids)}


def extract_ids_from_input(input_data: Any) -> List[int]:
    """
    兼容多种输入结构：
    - {"payload":{"ids":[1,2,3]}}
    - {"ids":[1,2]}
    - "1,2,3"
    - 1
    - null
    """
    try:
        obj = ensure_object(input_data)
    except Exception:
        return normalize_ids(input_data)

    if isinstance(obj, dict):
        if "payload" in obj and isinstance(obj["payload"], dict):
            if "ids" in obj["payload"]:
                return normalize_ids(obj["payload"]["ids"])
        if "ids" in obj:
            return normalize_ids(obj["ids"])

    return [1]


def get_token(session: requests.Session) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """调用认证接口，返回 (token, 错误对象)；兼容 id/token/access_token/accessToken 字段"""
    try:
        dlog("Auth →", AUTH_URI)
        resp = session.post(
            AUTH_URI,
            json={"username": AUTH_USERNAME, "password": AUTH_PASSWORD},
            verify=VERIFY_OPT,
            timeout=AUTH_TIMEOUT,
        )
        status = resp.status_code
        dlog("Auth status:", status)
        resp.raise_for_status()

        obj = resp.json()
        token = obj.get("id") or obj.get("token") or obj.get("access_token") or obj.get("accessToken")
        if not token:
            return None, {"stage": "auth", "status": status, "message": "认证接口未返回 token/id", "resp": obj}
        return token, None

    except (Timeout, ConnectionError) as e:
        return None, {"stage": "auth", "message": f"网络/超时异常：{str(e)}"}
    except HTTPError as e:
        return None, {"stage": "auth", "message": f"HTTP错误：{str(e)}"}
    except RequestException as e:
        return None, {"stage": "auth", "message": f"请求异常：{str(e)}"}
    except Exception as e:
        return None, {"stage": "auth", "message": f"解析异常：{str(e)}"}


def fetch_tag_values(session: requests.Session, token: str, tag_ids: List[int]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """查询标签数据（GET + params），如 API 要求 POST JSON 可改为 session.post(json={"ids": tag_ids})"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        params  = build_params(tag_ids)
        dlog("Data →", TAG_VALUES_URI, "params:", params)

        resp = session.get(
            TAG_VALUES_URI,
            headers=headers,
            params=params,
            verify=VERIFY_OPT,
            timeout=DATA_TIMEOUT,
        )
        status = resp.status_code
        dlog("Data status:", status)
        resp.raise_for_status()
        return resp.json(), None

    except (Timeout, ConnectionError) as e:
        return None, {"stage": "data", "message": f"网络/超时异常：{str(e)}"}
    except HTTPError as e:
        return None, {"stage": "data", "message": f"HTTP错误：{str(e)}"}
    except RequestException as e:
        return None, {"stage": "data", "message": f"请求异常：{str(e)}"}
    except Exception as e:
        return None, {"stage": "data", "message": f"解析异常：{str(e)}"}


def main(payload: Any = None, input: Any = None, event: Any = None, **kwargs) -> Dict[str, Any]:
    """
    Studio/Dify 可能用 main(payload=...) 调用，也可能传 input/event/其他名。
    这里统一接住，并始终返回 **dict**，其中 key 为 "result"。
    """
    try:
        incoming = None
        for cand in (payload, input, event, kwargs.get("data"), kwargs.get("body"), kwargs.get("args")):
            if cand is not None:
                incoming = cand
                break
        if incoming is None:
            incoming = globals().get("payload", None) or globals().get("input", None) or globals().get("event", None)

        tag_ids = extract_ids_from_input(incoming)

        session = requests.Session()
        session.trust_env = True

        token, auth_err = get_token(session)
        if auth_err:
            return {"result": {"ok": False, "ids_used": tag_ids, "error": auth_err}}

        data, data_err = fetch_tag_values(session, token, tag_ids)
        if data_err:
            return {"result": {"ok": False, "ids_used": tag_ids, "error": data_err}}

        return {"result": {"ok": True, "ids_used": tag_ids, "data": data}}

    except Exception as e:
        return {
            {"result": {
                "ok": False,
                "ids_used": extract_ids_from_input(payload),
                "error": {"message": f"Unexpected error: {str(e)}"}
            }}}


#return string

# -*- coding: utf-8 -*-

import os
import json
from typing import List, Union, Optional, Dict, Any, Tuple

import requests
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError
import warnings
from urllib3.exceptions import InsecureRequestWarning

API_BASE       = os.getenv("API_BASE", "https://4dd6651edf6c.ngrok-free.app")
AUTH_URI       = os.getenv("AUTH_URI",       f"{API_BASE}/api/account/authenticate")
TAG_VALUES_URI = os.getenv("TAG_VALUES_URI", f"{API_BASE}/api/historian/v02/tagvalues")

AUTH_TIMEOUT   = float(os.getenv("AUTH_TIMEOUT", "10"))
DATA_TIMEOUT   = float(os.getenv("DATA_TIMEOUT", "15"))

AUTH_USERNAME  = os.getenv("AUTH_USERNAME", "Yiting")
AUTH_PASSWORD  = os.getenv("AUTH_PASSWORD", "Metris123*")

USE_REPEATED_IDS_PARAM = os.getenv("USE_REPEATED_IDS_PARAM", "true").lower() == "true"

_verify_env = os.getenv("VERIFY_TLS", "false")
if _verify_env.lower() in ("true", "false"):
    VERIFY_OPT: Union[bool, str] = (_verify_env.lower() == "true")
else:
    VERIFY_OPT = _verify_env 

DEBUG_LOG      = os.getenv("DEBUG_LOG", "true").lower() == "true"

try:
    VALUE_DECIMALS = int(os.getenv("VALUE_DECIMALS", "").strip()) if os.getenv("VALUE_DECIMALS") else None
except Exception:
    VALUE_DECIMALS = None

if VERIFY_OPT is False:
    warnings.simplefilter("ignore", InsecureRequestWarning)


def dlog(*args: Any) -> None:
    if DEBUG_LOG:
        print("[CODE-DEBUG]", *args)

def safe_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def ensure_object(maybe_json: Any) -> Dict[str, Any]:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        s = maybe_json.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json\n"):
                s = s[5:]
        return json.loads(s)
    raise ValueError("输入既非对象也非字符串，无法解析为 JSON 对象")

def normalize_ids(ids: Optional[Union[int, str, List[Union[int, str]]]]) -> List[int]:
    if ids is None or ids == "null":
        return [1]
    if isinstance(ids, int):
        return [ids]
    if isinstance(ids, str):
        s = ids.strip()
        if not s:
            return [1]
        out: List[int] = []
        for p in s.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                out.append(int(p))
            except Exception:
                continue
        return out or [1]
    if isinstance(ids, list):
        out: List[int] = []
        for p in ids:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out or [1]
    return [1]

def build_params(tag_ids: List[int]) -> Union[Dict[str, str], List[Tuple[str, int]]]:
    if USE_REPEATED_IDS_PARAM:
        return [("ids", i) for i in tag_ids]
    else:
        return {"ids": ",".join(str(i) for i in tag_ids)}

def extract_ids_from_input(input_data: Any) -> List[int]:
    try:
        obj = ensure_object(input_data)
    except Exception:
        return normalize_ids(input_data)

    if isinstance(obj, dict):
        if "payload" in obj and isinstance(obj["payload"], dict):
            if "ids" in obj["payload"]:
                return normalize_ids(obj["payload"]["ids"])
        if "ids" in obj:
            return normalize_ids(obj["ids"])
    return [1]

def get_token(session: requests.Session) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    try:
        dlog("Auth →", AUTH_URI)
        resp = session.post(
            AUTH_URI,
            json={"username": AUTH_USERNAME, "password": AUTH_PASSWORD},
            verify=VERIFY_OPT,
            timeout=AUTH_TIMEOUT,
        )
        status = resp.status_code
        dlog("Auth status:", status)
        resp.raise_for_status()

        obj = resp.json()
        token = obj.get("id") or obj.get("token") or obj.get("access_token") or obj.get("accessToken")
        if not token:
            return None, {"stage": "auth", "status": status, "message": "认证接口未返回 token/id", "resp": obj}
        return token, None

    except (Timeout, ConnectionError) as e:
        return None, {"stage": "auth", "message": f"网络/超时异常：{str(e)}"}
    except HTTPError as e:
        return None, {"stage": "auth", "message": f"HTTP错误：{str(e)}"}
    except RequestException as e:
        return None, {"stage": "auth", "message": f"请求异常：{str(e)}"}
    except Exception as e:
        return None, {"stage": "auth", "message": f"解析异常：{str(e)}"}

def fetch_tag_values(session: requests.Session, token: str, tag_ids: List[int]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    try:
        headers = {"Authorization": f"Bearer {token}"}
        params  = build_params(tag_ids)
        dlog("Data →", TAG_VALUES_URI, "params:", params)

        resp = session.get(
            TAG_VALUES_URI,
            headers=headers,
            params=params,
            verify=VERIFY_OPT,
            timeout=DATA_TIMEOUT,
        )
        status = resp.status_code
        dlog("Data status:", status)
        resp.raise_for_status()
        return resp.json(), None

    except (Timeout, ConnectionError) as e:
        return None, {"stage": "data", "message": f"网络/超时异常：{str(e)}"}
    except HTTPError as e:
        return None, {"stage": "data", "message": f"HTTP错误：{str(e)}"}
    except RequestException as e:
        return None, {"stage": "data", "message": f"请求异常：{str(e)}"}
    except Exception as e:
        return None, {"stage": "data", "message": f"解析异常：{str(e)}"}

def shape_rows(data: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            tag_id = item.get("tagID")
            ts     = item.get("timestamp")
            val    = item.get("value")
            qual   = item.get("quality")
            vstr   = item.get("valueString", "")

            if VALUE_DECIMALS is not None and isinstance(val, (int, float)):
                try:
                    val = round(float(val), VALUE_DECIMALS)
                except Exception:
                    pass

            rows.append({
                "tagID": int(tag_id) if tag_id is not None else None,
                "timestamp": ts,
                "value": val,
                "quality": qual,
                "valueString": vstr
            })
    return rows

def to_result_json_string_ok(ids_used: List[int], data: Any) -> str:
    rows = shape_rows(data)
    payload = {
        "ok": True,
        "ids_used": list(ids_used or []),
        "rows": rows,
        "meta": {
            "count": len(rows),
            "ids_unique": sorted(set(ids_used or []))
        }
    }
    return safe_dumps(payload)

def to_result_json_string_error(ids_used: List[int], error: Optional[Dict[str, Any]]) -> str:
    payload = {
        "ok": False,
        "ids_used": list(ids_used or []),
        "error": {
            "stage": (error or {}).get("stage"),
            "status": (error or {}).get("status"),
            "message": (error or {}).get("message"),
        }
    }
    return safe_dumps(payload)


def main(payload: Any = None, input: Any = None, event: Any = None, **kwargs) -> Dict[str, Any]:
    try:
        incoming = None
        for cand in (payload, input, event, kwargs.get("data"), kwargs.get("body"), kwargs.get("args")):
            if cand is not None:
                incoming = cand
                break
        if incoming is None:
            incoming = globals().get("payload", None) or globals().get("input", None) or globals().get("event", None)

        tag_ids = extract_ids_from_input(incoming)
        dlog("IDs used:", tag_ids)

        session = requests.Session()
        session.trust_env = True

        token, auth_err = get_token(session)
        if auth_err:
            return {"result": to_result_json_string_error(tag_ids, auth_err)}

        data, data_err = fetch_tag_values(session, token, tag_ids)
        if data_err:
            return {"result": to_result_json_string_error(tag_ids, data_err)}

        return {"result": to_result_json_string_ok(tag_ids, data)}

    except Exception as e:
        ids_fallback: List[int] = []
        try:
            ids_fallback = extract_ids_from_input(payload)
        except Exception:
            pass
        return {"result": to_result_json_string_error(ids_fallback, {"stage": "code", "message": f"Unexpected error: {str(e)}"})}


print(main({"payload": {"ids": [1]}}))