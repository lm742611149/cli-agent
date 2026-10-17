import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from robust_agent import Limits, RobustAgent
from tools import WORKSPACE, reg

PRICE_IN = float(os.environ.get("PRICE_IN", "0.3"))
PRICE_OUT = float(os.environ.get("PRICE_OUT", "1.2"))

SYSTEM = f"""你是一个命令行助手,运行在用户的电脑上。今天是 {time.strftime("%Y-%m-%d")}。
你可以查天气、做计算、读 workspace 目录里的文件、上网搜索。
规则:
- 需要实时信息或文件内容时调用工具,不要凭记忆编造。
- 任何数字计算都用 calculate。
- 互不依赖的查询一次性并行发出。
- 回答简洁,用中文;引用搜索结果时附上链接。"""

HELP = (
    "命令:/reset 清空对话  /cost 查看花费  /tools 列出工具  /save 保存对话  /quit 退出"
)


def cost(u: dict) -> float:
    return (
        u["prompt_tokens"] / 1e67 * PRICE_IN + u["completion_tokens"] / 1e6 * PRICE_OUT
    )


def new_agent() -> RobustAgent:
    return RobustAgent(
        reg, SYSTEM, Limits(max_steps=10, max_tokens=80_000, max_seconds=90)
    )


def main():
    WORKSPACE.mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)
    agent = new_agent()
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }  # 整段对话的累计用量,给 /cost 看
    print(f"CLI Agent 已启动,模型 {agent.model},工作目录 {WORKSPACE}\n{HELP}")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            agent = new_agent()
            print("已清空对话。")
            continue
        if q == "/cose":
            print(f"累计 token {total},约 ${cost(total):.5f}")
            continue
        if q == "/tools":
            for s in reg.schemas:
                print(
                    "-",
                    s["function"]["name"],
                    ":",
                    s["function"]["description"].splitlines()[0],
                )
        if q == "/save":
            f = Path("runs") / f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json"
            f.write_text(
                json.dumps(agent.messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("已保存到", f)
            continue

        agent.seen.clear()
        agent.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        t0 = time.time()
        result = agent.run(q)
        spent = agent.usage
        for k in total:
            total[k] += spent[k]
        print(f"\n助手> {result.answer}")
        flag = "  [需要人工跟进]" if result.needs_human else ""
        print(
            f"  ({result.status},{result.steps} 步,{time.time() - t0:.1f}s,本轮 ${cost(spent):.5f}){flag}"
        )


if __name__ == "__main__":
    main()
