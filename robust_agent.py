import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ToolTimeout
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agent_loop import ToolRegistry

RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass
class Limits:
    max_steps: int = 12  # 最多调几次模型
    max_tokens: int = 60_000  # 输入+输出 token 总预算
    max_seconds: float = 120  # 整个任务的墙钟时间
    tool_timeout: float = 15  # 单个工具最长执行时间
    max_tool_chars: int = 4_000  # 单个工具结果最多放进上下文多少字符
    max_repeats: int = 2  # 完全相同的调用最多允许几次


@dataclass
class AgentResult:
    answer: str
    status: str  # ok / max_steps / budget / timeout / loop / llm_error
    steps: int
    usage: dict = field(default_factory=dict)
    needs_human: bool = False


class RobustAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        system: str,
        limits: Limits | None = None,
        client: Any = None,
    ) -> None:
        self.client = client or OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            timeout=60,
            max_retries=0,
        )
        self.model = os.environ.get("LLM_MODEL", "fake")
        self.registry = registry
        self.limits = limits or Limits()
        self.messages: list[Any] = [{"role": "system", "content": system}]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.seen: dict[str, int] = {}
        self.pool = ThreadPoolExecutor(max_workers=4)

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _llm(self, **kwargs: Any) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model, messages=self.messages, **kwargs
        )

    def _run_tool(self, name: str, arguments: str) -> str:
        future = self.pool.submit(self.registry.call, name, arguments)
        try:
            out = future.result(timeout=self.limits.tool_timeout)
        except ToolTimeout:
            return json.dumps(
                {
                    "error": f"工具 {name} 查过 {self.limits.tool_timeout} 秒没有返回。可以换个参数重试,或者告诉用户暂时查不到。"
                },
                ensure_ascii=False,
            )
        if len(out) > self.limits.max_tool_chars:
            cut = self.limits.max_tool_chars
            out = (
                out[:cut]
                + f"\n…[结果过长,已截断:共 {len(out)} 字符,只显示前 {cut}。需要的话请用更精确的参数重新查询]"
            )
        return out

    def _is_loop(self, name: str, arguments: str) -> bool:
        key = hashlib.md5(f"{name}:{arguments}".encode()).hexdigest()
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key] > self.limits.max_repeats

    def _wrap_up(self, status: str, step: int, reason: str) -> AgentResult:
        """撞到上限时,不让工作白做:禁用工具,让模型基于已有信息给出部分答案。"""
        self.messages.append(
            {
                "role": "user",
                "content": f"(系统提示:{reason}。不要再调用工具,根据目前已经拿到的信息给出你能给的最好回答,并明确说明哪些部分没有完成。)",
            }
        )
        try:
            msg = (
                self._llm(tools=self.registry.schemas, tool_choice="none")
                .choices[0]
                .message
            )
            answer = msg.content or ""
        except Exception as e:
            answer = (
                f"抱歉,这个问题暂时没能处理完({type(e).__name__}),已转给人工同事跟进。"
            )
        self.messages.append({"role": "assistant", "content": answer})  # 收尾的回答也进历史,日志和下一轮对话才完整
        return AgentResult(answer, status, step, dict(self.usage), needs_human=True)

    @staticmethod
    def _assistant_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            d["reasoning_content"] = reasoning
        return d

    def run(self, user_input: str) -> AgentResult:
        self.messages.append({"role": "user", "content": user_input})
        start = time.monotonic()
        for step in range(1, self.limits.max_steps + 1):
            if time.monotonic() - start > self.limits.max_seconds:
                return self._wrap_up("timeout", step, "任务耗时超过上限")
            if sum(self.usage.values()) > self.limits.max_tokens:
                return self._wrap_up("budget", step, "token 预算已用完")
            try:
                resp = self._llm(tools=self.registry.schemas)
            except RETRYABLE as e:
                return AgentResult(
                    f"模型服务暂时不可用({type(e).__name__}),请稍后再试。",
                    "llm_error",
                    step,
                    dict(self.usage),
                    True,
                )
            msg = resp.choices[0].message
            if resp.usage:
                self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                self.usage["completion_tokens"] += resp.usage.completion_tokens
            self.messages.append(self._assistant_dict(msg))
            if not msg.tool_calls:
                return AgentResult(msg.content or "", "ok", step, dict(self.usage))

            looping = False
            for tc in msg.tool_calls:
                if tc.type != "function":
                    continue
                if self._is_loop(tc.function.name, tc.function.arguments):
                    looping = True
                    out = json.dumps(
                        {
                            "error": "这个调用和之前完全一样,结果不会变。请换一种做法,或者直接根据已有信息回答。"
                        },
                        ensure_ascii=False,
                    )
                else:
                    out = self._run_tool(tc.function.name, tc.function.arguments)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": out}
                )
            if looping and max(self.seen.values()) > self.limits.max_repeats + 1:
                return self._wrap_up("loop", step, "检测到重复调用,任务可能陷入循环")
        return self._wrap_up("max_steps", self.limits.max_steps, "已达到最大步数")
