"""
4、
【模块名称】极简 Agent 智能体与工具调用机制 (Function Calling Loop)
【解决痛点】大模型本身存在知识滞后与计算幻觉；通过工具调用机制，让 LLM 学会输出结构化 JSON 调用外部 Python 工具获取真实世界数据。
【核心流程】
  1. 定义外部真实工具 (如真正的数学计算器、天气查询函数)
  2. 构造 Prompt 告诉 LLM 有哪些工具可用，并要求其输出 JSON 格式
  3. 模拟 LLM 决策并输出函数调用指令 (Function Call)
  4. 外部 Python 解释器执行函数获取真实结果
  5. 将执行结果作为 Context 塞回 LLM，由 LLM 生成最终人类自然语言回复
【数据流向】用户 Query -> LLM 工具决策 JSON -> 本地 Python 函数执行 -> 观测结果回传 -> LLM 终验回复
"""

import json


# 1. 定义我们提供给智能体的“外部工具库”（真实的 Python 函数）
def calculate_complex_math(expression: str):
    """用于精确计算复杂数学表达式的计算器"""
    try:
        # 实际生产中可使用更安全的数学解析库，此处用于演示
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {str(e)}"


def get_current_city_weather(city: str):
    """模拟查询指定城市实时天气的接口"""
    weather_data = {
        "杭州": "晴朗，气温 21°C，微风",
        "北京": "晴间多云，气温 15°C，北风3级"
    }
    return weather_data.get(city, "未查询到该城市的天气信息")


# 工具名字与真实函数的映射字典
tool_registry = {
    "calculate_complex_math": calculate_complex_math,
    "get_current_city_weather": get_current_city_weather
}


# 2. 模拟 LLM 的决策大脑 (这里模拟了一个被 SFT 训练好的大模型输出)
def mock_llm_brain(prompt: str):
    """
    模拟大模型处理 Prompt。
    当检测到需要工具时，它不会硬猜，而是主动输出一段 JSON 字符串！
    """
    if "天气" in prompt and "杭州" in prompt:
        # LLM 决定调用天气函数
        return json.dumps({
            "action": "call_tool",
            "tool_name": "get_current_city_weather",
            "tool_args": {"city": "杭州"}
        })
    elif "乘以" in prompt:
        # LLM 决定调用计算器
        return json.dumps({
            "action": "call_tool",
            "tool_name": "calculate_complex_math",
            "tool_args": {"expression": "98765 * 43210"}
        })
    else:
        # 不需要工具，直接回答
        return json.dumps({
            "action": "final_answer",
            "response": "你好！我是你的学术 AI 助手，有什么可以帮你的？"
        })


# 3. 核心：Agent 执行主循环 (Agent Runtime Loop)
def run_agent(user_query: str):
    print(f"\n【用户提问】: {user_query}")

    # 步骤 1: 将用户指令喂给 LLM 大脑思考
    llm_thought = mock_llm_brain(user_query)
    decision = json.loads(llm_thought)

    # 步骤 2: 判断大模型是要调用工具，还是要直接说话
    if decision["action"] == "call_tool":
        tool_name = decision["tool_name"]
        tool_args = decision["tool_args"]
        print(f"【LLM 思考】: 我自己算不准/查不到，我决定调用工具 -> {tool_name}，参数: {tool_args}")

        # 步骤 3: 触发真实 Python 代码执行工具
        tool_func = tool_registry[tool_name]
        tool_result = tool_func(**tool_args)
        print(f"【工具执行真实结果】: {tool_result}")

        # 步骤 4: 把工具结果塞回给 LLM 生成最终人话
        final_response = f"根据实时工具查询返回，结果是：{tool_result}。"
        print(f"【Agent 最终给用户的回复】: {final_response}")
    else:
        print(f"【Agent 直接回复】: {decision['response']}")


if __name__ == "__main__":
    # 测试用例 1: 复杂算术计算 (调用计算器)
    run_agent("请帮我算一下 98765 乘以 43210 等于几？")

    # 测试用例 2: 外部实时状态 (调用天气接口)
    run_agent("请问杭州现在的天气怎么样？")

    # 测试用例 3: 普通问答 (无需工具)
    run_agent("打个招呼吧！")