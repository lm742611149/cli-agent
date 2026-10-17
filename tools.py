import ast
import operator as op
import os
from pathlib import Path
from typing import Annotated, Literal

import httpx

from agent_loop import ToolRegistry

reg = ToolRegistry()
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "workspace")).resolve()
WMO = {
    0: "晴",
    1: "大致晴",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    95: "雷暴",
}


@reg.tool
def weather_now(
    city: Annotated[
        str,
        "城市名,中文或英文。重名城市可以加省份或国家,例如 '苏州 江苏'、'Paris France'",
    ],
    unit: Annotated[Literal["celsius", "fahrenheit"], "温度单位,默认摄氏"] = "celsius",
) -> dict:
    """查询一个城市当前的天气:气温、天气状况、风速。
    用户问天气、冷不冷、要不要带伞、适合穿什么时使用。只能查当前,不能查历史或未来几天。
    一次只查一个城市;要比较多个城市就并行调用多次。"""
    name = city.split()[0]
    geo = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1, "language": "zh"},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return {
            "error": f"找不到城市 '{city}'。可以试试英文名(如 'Hangzhou')或者更常见的写法。"
        }
    p = geo["results"][0]
    cur = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        timeout=10,
        params={
            "latitude": p["latitude"],
            "longitude": p["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": unit,
        },
    ).json()["current"]
    return {
        "city": f"{p['name']}({p.get('admin1', '')})",
        "temperature": cur["temperature_2m"],
        "unit": unit,
        "weather": WMO.get(cur["weather_code"], f"代码{cur['weather_code']}"),
        "wind_kmh": cur["wind_speed_10m"],
    }


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}


@reg.tool
def calculate(
    expression: Annotated[
        str, "算式,只能包含数字、+ - * / // % ** 和括号,例如 '(21.5 + 18.2) / 2'"
    ],
) -> dict:
    """精确计算一个数学表达式。任何涉及数字运算的地方都用它,不要心算,哪怕看起来很简单。
    不支持变量、函数(如 sqrt)和单位;需要开方可以写成 '** 0.5'。"""

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("指数太大")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("只支持数字和 + - * / // % ** 括号")

    try:
        v = ev(ast.parse(expression, mode="eval").body)
        return {
            "expression": expression,
            "result": round(v, 10) if isinstance(v, float) else v,
        }
    except ZeroDivisionError:
        return {"error": "除数为 0"}
    except Exception as e:
        return {"error": f"无法计算 '{expression}':{e}"}


def _safe(path: str) -> Path | None:
    target = (WORKSPACE / path).resolve()
    return target if target.is_relative_to(WORKSPACE) else None


@reg.tool
def files_list(
    subdir: Annotated[str, "workspace 下的子目录,默认根目录 '.'"] = ".",
) -> dict:
    """列出 workspace 目录(或其子目录)里的文件和文件夹名。不知道文件叫什么时先用它,再用 files_read 读。"""
    d = _safe(subdir)
    if d is None or not d.is_dir():
        return {"error": f"'{subdir}' 不是 workspace 里的目录"}
    items = sorted(d.iterdir())[:100]
    return {
        "dir": subdir,
        "items": [p.name + ("/" if p.is_dir() else "") for p in items],
    }


@reg.tool
def files_read(
    path: Annotated[str, "workspace 里的相对路径,例如 'notes/todo.md'"],
    start_line: Annotated[int, "从第几行开始,默认 1"] = 1,
    max_lines: Annotated[int, "最多读几行,默认 80,最大 300"] = 80,
) -> dict:
    """读取 workspace 里一个文本文件的指定行范围,每行带行号。只能读 workspace 内的文件。
    文件很长时会分段返回,用 next_start_line 继续读。"""
    f = _safe(path)
    if f is None:
        return {"error": "只能读取 workspace 目录内的文件"}
    if not f.is_file():
        return {"error": f"文件 '{path}' 不存在。先用 files_list 看看有哪些文件。"}
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return {"error": f"'{path}' 不是 UTF-8 文本文件,无法读取"}
    max_lines = max(1, min(max_lines, 300))
    chunk = lines[start_line - 1 : start_line - 1 + max_lines]
    out = {
        "path": path,
        "total_lines": len(lines),
        "lines": [f"{i}: {s}" for i, s in enumerate(chunk, start_line)],
    }
    if start_line - 1 + len(chunk) < len(lines):
        out["next_start_line"] = start_line + len(chunk)
    return out


@reg.tool
def web_search(
    query: Annotated[str, "搜索词,像在搜索引擎里输入的那样,简短具体"],
    max_results: Annotated[int, "返回几条,默认 5,最多 10"] = 5,
) -> dict:
    """在互联网上搜索,返回标题、链接和摘要。用于查最新信息、你不确定的事实、或者用户明确要求搜索时。
    结果只有摘要,不是全文;摘要不够时换更具体的搜索词再搜一次,不要重复搜同一个词。"""
    n = max(1, min(max_results, 10))
    try:
        if os.environ.get("TAVILY_API_KEY"):
            r = httpx.post(
                "https://api.tavily.com/search",
                timeout=20,
                headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
                json={"query": query, "max_results": n},
            ).json()
            hits = [
                {"title": x["title"], "url": x["url"], "snippet": x["content"][:300]}
                for x in r.get("results", [])
            ]

        else:  # 默认用 DuckDuckGo,不用 key,但国内直连通常访问不了
            from ddgs import DDGS

            hits = [
                {"title": x["title"], "url": x["href"], "snippet": x["body"][:300]}
                for x in DDGS().text(query, max_results=n)
            ]
    except Exception as e:
        return {
            "error": f"搜索服务暂时不可用({type(e).__name__})。可以稍后重试,或告诉用户无法联网搜索。"
        }
    return (
        {"query": query, "results": hits}
        if hits
        else {"query": query, "results": [], "hint": "没有结果,换个说法试试"}
    )
