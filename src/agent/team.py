from .agent import Agent
from src.llm_proxy import BaseTool,BaseModel,LLMMessage
from typing import Any, Generator
from pydantic import Field


class AskTeamMemberInput(BaseModel):
    name: str = Field(description="The name of the team member to ask for help")
    question: str = Field(description="The question to ask the team member")

class AskTeamMemberTool(BaseTool):
    name: str = "ask_team_member"
    description: str = "Ask other members in your team for help"
    argSchema: BaseModel = AskTeamMemberInput
    team:Any

    def __init__(self, team:Any):
        self.team = team

    def _run(self, name: str, question: str) -> Generator[str, None, None]:
        for agent in self.team.agents:
            if agent.name == name:
                return agent.chat_default(question)
        return "Agent not found"

class Team:
    name: str
    goal: str
    backstory: str
    agents: list[Agent]
    team_tools:list[BaseTool]

    def __init__(self,name:str,goal:str,backstory:str,agents:list[Agent]=[],team_tools:list[BaseTool]=[]):
        self.name = name
        self.goal = goal
        self.backstory = backstory
        self.agents = agents
        temp_agents = self.agents
        self.agents = []
        for agent in temp_agents:
            self.add_agent(agent)
            if agent.allow_ask_other:
                agent.add_tool(AskTeamMemberTool(self))

    def add_agent(self, agent: Agent):
        #找到agent的第一条system消息，插入到后面
        for i, msg in enumerate(agent.llm.session):
            if msg.role == "system":
                agent.llm.session.insert(i+1, LLMMessage(role="system", content=self._get_team_prompt()))
                break
        else:
            agent.llm.session.append(LLMMessage(role="system", content=self._get_team_prompt()))

        for tool in self.team_tools:
            agent.add_tool(tool)
        self.agents.append(agent)

    def remove_agent(self, agent: Agent):
        #找到agent的第1条system消息之后的system消息
        for i, msg in enumerate(agent.llm.session):
            if msg.role == "system":
                agent.llm.session.pop(i+1)
                break
        if agent.allow_ask_other:
            agent.remove_tool(AskTeamMemberTool.name)
        for tool in self.team_tools:
            agent.remove_tool(tool.name)
        self.agents.remove(agent)

    def _get_team_prompt(self):
        return f"""
        Your team name: {self.name}\n
        Your team goal: {self.goal}\n
        Your team backstory: {self.backstory}\n
        You can ask other members in your team for help by use ask_team_member function.
        Other members in your team:
        {'\n'.join([self._get_agent_prompt(agent) for agent in self.agents])}
        """
    def _get_agent_prompt(self, agent: Agent):
        return f"""
        Member name: {agent.name}\n
        Member backstory: {agent.backstory}\n
        Member tools: {'\n'.join([f"{tool.name}:{tool.description}" for tool in agent.function_call.tools.values()])}\n
        """
        
        

    