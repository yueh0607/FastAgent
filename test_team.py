import asyncio
import os
from src.llm_proxy import DeepSeekLLM, BaseTool
from src.agent import Agent
from src.agent.team import Team
from pydantic import BaseModel

# 定义一个简单的工具用于测试
class GreetingSchema(BaseModel):
    name: str

class GreetingTool(BaseTool):
    name = "greeting"
    description = "一个简单的问候工具"
    argSchema = GreetingSchema

    def _run(self, name: str) -> str:
        return f"你好，{name}！"

def create_test_agent(name: str, backstory: str, goal: str, llm) -> Agent:
    """创建测试用的Agent"""
    return Agent(
        name=name,
        backstory=backstory,
        goal=goal,
        llm=llm,
        default_model="deepseek-chat",
        tools=[GreetingTool()],
        allow_ask_other=True
    )

def chat_io(agent: Agent, question: str):
    """测试单个对话"""
    print(f"\n用户对{agent.name}说: {question}")
    print(f"{agent.name}: ", end='', flush=True)
    
    response = agent.chat(
        question,
        model="deepseek-chat",
        temperature=0.7
    )
    
    for chunk in response:
        print(chunk, end='', flush=True)
    print("\n")

def test_team_features():
    """测试 Team 的各种功能"""
    # 初始化 LLM
    api_key = os.environ.get("MODEL_API_KEY")

    # 创建多个Agent
    leader = create_test_agent(
        "队长",
        "你是团队的领导者，负责协调团队成员工作",
        "确保团队高效运作",
        DeepSeekLLM(api_key)
    )
    
    tech_expert = create_test_agent(
        "技术专家",
        "你是团队的技术专家，擅长解决技术问题",
        "提供技术支持和解决方案",
        DeepSeekLLM(api_key)
    )
    
    creative = create_test_agent(
        "创意师",
        "你是团队的创意师，擅长提出创新想法",
        "提供创新的解决方案",
        DeepSeekLLM(api_key)
    )

    # 创建团队
    team = Team(
        name="超级团队",
        goal="高效解决各种问题",
        backstory="我们是一个充满活力的团队，每个成员都有自己的专长",
        agents=[leader,tech_expert,creative]
    )


    # 测试团队成员间的交互
    print("\n=== 测试团队成员交互 ===")
    chat_io(leader, "请技术专家帮我看看这个问题如何解决")
    
    # 测试移除成员
    print("\n=== 测试移除团队成员 ===")
    team.remove_agent(creative)
    print(f"移除创意师后的团队成员数量: {len(team.agents)}")

    # 测试剩余成员的功能
    print("\n=== 测试剩余成员功能 ===")
    chat_io(tech_expert, "我们现在还有哪些团队成员？")

    
def main():
    
    test_team_features()


if __name__ == "__main__":
    main()