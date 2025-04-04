import asyncio
from src.llm_proxy import DeepSeekLLM, BaseTool
from src.agent import Agent
from pydantic import BaseModel
from typing import List
import os
# 定义计算器工具
class CalculatorSchema(BaseModel):
    x: float
    y: float
    operation: str

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "一个简单的计算器，支持加减乘除运算，以及特殊运算符sp"
    argSchema = CalculatorSchema

    def _run(self, x: float, y: float, operation: str) -> str:
        if operation == "+":
            return f"{x} + {y} = {x + y}"
        elif operation == "-":
            return f"{x} - {y} = {x - y}"
        elif operation == "*":
            return f"{x} * {y} = {x * y}"
        elif operation == "/":
            if y == 0:
                return "错误：除数不能为0"
            return f"{x} / {y} = {x / y}"
        elif operation == "sp":
            return f"{x} sp {y} = {x + y + 5}"
        else:
            return f"不支持的运算：{operation}"

# 定义天气查询工具
class WeatherSchema(BaseModel):
    city: str
    date: str

class WeatherTool(BaseTool):
    name = "weather"
    description = "查询指定城市的天气情况"
    argSchema = WeatherSchema

    def _run(self, city: str, date: str) -> str:
        # 模拟天气查询
        return f"{city}在{date}的天气是：晴朗，温度25°C"

def chat_io(agent: Agent, question: str, stream: bool = True):
    """测试单个对话"""
    print(f"用户: {question}")
    print("助手: ", end='', flush=True)
    
    response = agent.chat(
        question,
        model="deepseek-chat",
        temperature=0.7,
        stream=stream
    )
    
    if stream:
        for chunk in response:
            print(chunk, end='', flush=True)
    else:
        print(response)
    print("\n")

def test_agent_features():
    """测试 Agent 的各种功能"""
    # 初始化 LLM
    api_key = os.environ.get("MODEL_API_KEY")  # 替换为实际的 API key
    llm = DeepSeekLLM(api_key)
    
    # 1. 创建基础助手
    print("\n=== 1. 创建基础助手 ===")
    base_agent = Agent(
        name="安娜",
        backstory="你是一个漂亮的大姐姐，喜欢帮助别人，喜欢开玩笑",
        goal="提供准确和有用的帮助，并总结问题",
        llm=llm,
        tools=[CalculatorTool, WeatherTool]
    )

    base_agent.llm.dump_session()

    #帮我计算5 sp 7
    chat_io(base_agent, "现在有一个问题："
                       "美丽的公主一口气吃了5 sp 2个苹果，但是其中2 sp 0个是有毒的"
                       "请问公主吃了多少个没毒的苹果？"
                       )
    chat_io(base_agent, "答案是什么？")

async def main():
    try:
        test_agent_features()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())