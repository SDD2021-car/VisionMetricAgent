import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # 你已安装 langgraph-checkpoint

from tools_eval import add_pair, list_pairs, eval_pairs, save_last_report

# 我想计算gt=“/NAS_data/yjy/GF3_new/testA”,gen="/NAS_data/yjy/GF3_new/testB"的psnr

def build_agent():
    llm = ChatOpenAI(
        model="qwen2.5-32b-instruct",
        temperature=0.2,
        api_key="sk-7a52945e5f4f47b1aec5ff885a25383f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    tools = [add_pair, list_pairs, eval_pairs, save_last_report]

    # system = (
    #     "你是图像生成实验评估助手，必须通过工具计算指标，禁止编造数值。\n"
    #     "当用户要求计算指标但 pairs 为空时，必须先问是否添加(gt_dir, gen_dir)。"
    # )

    checkpointer = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=checkpointer)

    return agent


def main():
    agent = build_agent()
    print("Metric-Agent CLI（输入 exit 退出）")

    # 用 thread_id 让同一次会话的记忆持续
    config = {"configurable": {"thread_id": "cli-session"}}
    system = (
        "你是图像生成实验评估助手，必须通过工具计算指标，禁止编造数值。\n"
        "当用户要求计算指标但 pairs 为空时，必须先问是否添加(gt_dir, gen_dir)。"
    )
    while True:
        user_in = input("You> ").strip()
        if user_in.lower() in ("exit", "quit"):
            break
        if not user_in:
            continue

        result = agent.invoke(
            {"messages": [("system", system), ("user", user_in)]},
            config=config,
        )

        # result["messages"] 是完整对话，最后一条通常是 assistant
        print("Agent>", result["messages"][-1].content)


if __name__ == "__main__":
    main()
