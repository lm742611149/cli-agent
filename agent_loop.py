import inspect
import json
import os
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from openai import OpenAI

PY2JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _param_schema(tp: Any) -> dict:
    desc = None
    if get_origin(tp) is Annotated:
        tp, desc = get_args(tp)[0], get_args(tp)[1]
    if get_origin(tp) is Literal:
        schema = {"type": "string", "enum": list(get_args(tp))}
    else:
        schema = {"type": PY2JSON.get(tp, "string")}
    if desc:
        schema["description"] = desc
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self.funcs: dict[str, Callable] = {}
        self.schemas: list[Any] = []

    def tool(self, fn: Callable) -> Callable:
        hints = get_type_hints(fn, include_extras=True)
        sig = inspect.signature(fn)
        props, required = {}, []
        for name, p in sig.parameters.items():
            props[name] = _param_schema(hints.get(name, str))
            if p.default is inspect.Parameter.empty:
                required.append(name)
        self.funcs[fn.__name__] = fn
        self.schemas.append(
            {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": inspect.getdoc(fn) or "",
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }
        )

        return fn

    def call(self, name: str, arguments: str) -> str:
        if name not in self.funcs:
            return json.dumps(
                {"error": f"没有叫 {name} 的工具,可用的有 {list(self.funcs)}"},
                ensure_ascii=False,
            )
        try:
            args = json.loads(arguments or "{}")
            result = self.funcs[name](**args)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result, ensure_ascii=False, default=str)


class Agent:
    def __init__(
        self, registry: ToolRegistry, system: str, max_steps: int = 10
    ) -> None:
        self.client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"]
        )
        self.model = os.environ["LLM_MODEL"]
        self.registry = registry
        self.max_steps = max_steps
        self.messages: list[Any] = [{"role": "system", "content": system}]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        for step in range(1, self.max_steps + 1):
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.registry.schemas
            )
            msg = resp.choices[0].message
            if resp.usage:
                self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                self.usage["completion_tokens"] += resp.usage.completion_tokens
            self.messages.append(self._assistant_dict(msg))

            if not msg.tool_calls:
                print(f"[step {step}] 完成 ({time.time() - t0:.1f}s)")
                return msg.content or ""

            for tc in msg.tool_calls:
                if tc.type != "function":
                    continue
                print(f"[step {step}] 调用 {tc.function.name}({tc.function.arguments})")
                out = self.registry.call(tc.function.name, tc.function.arguments)
                print(f"           结果 {out[:120]}")
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": out}
                )

        return f"已达到最大步数 {self.max_steps},任务没有完成。最后的进展见上面的日志。"

    @staticmethod
    def _assistant_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        reasoning = getattr(
            msg, "reasoning_content", None
        )  # DeepSeek 思考模式,带 tools 时要回传
        if reasoning:
            d["reasoning_content"] = reasoning
        return d
