from typing import List, Dict, Any, Generator, Union
from src.llm_proxy.llm_base import LLMBase, LLMMessage
from src.llm_proxy.function_call import FunctionCall
from src.llm_proxy.tool import BaseTool
from typing import Type

class Agent:
    """A conversational agent that can use tools and maintain context."""
    
    def __init__(
        self,
        name: str,
        backstory: str,
        goal: str,
        llm: LLMBase,
        tools: List[Type[BaseTool]] = None
    ):
        """
        Initialize an agent with its identity and capabilities.
        
        Args:
            name: The name of the agent
            backstory: The agent's background story and personality
            goal: The agent's primary objective
            llm: An instance of LLMBase for language model interactions
            tools: List of tool classes the agent can use
        """
        self.name = name
        self.backstory = backstory
        self.goal = goal
        self.llm = llm
        self.function_call = FunctionCall()
        
        # Initialize tools if provided
        if tools:
            for tool in tools:
                self.function_call.add_tool(tool)
        
        # Initialize the system message
        self._update_system_message()
    
    def _update_system_message(self):
        """Update the system message with current agent identity and tools."""
        # Base system message with agent identity
        base_content = f"""You are {self.name}\n,your backstory: {self.backstory}\n
Your goal is: {self.goal}\n"""

        # Add tools information if any tools are available
        tools_prompt = self.function_call.get_system_prompt()
        if tools_prompt:
            base_content += f"\n{tools_prompt}"

        # Create new system message
        self.system_message = LLMMessage(
            role="system",
            content=base_content
        )
        
        # Update session while preserving other messages
        if self.llm.session:
            # Find and replace the system message
            for i, msg in enumerate(self.llm.session):
                if msg.role == "system":
                    self.llm.session[i] = self.system_message
                    break
            else:
                # If no system message found, add it at the beginning
                self.llm.session.insert(0, self.system_message)
        else:
            # If session is empty, just add the system message
            self.llm.session = [self.system_message]
    
    def chat(self, message: str, model:str=None,temperature:float=0.7,stream:bool=True) -> Union[Generator[str, None, None], str]:
        """
        Have a conversation with the agent.
        
        Args:
            message: The user's message
            model: The model to use
            temperature: The temperature to use
            stream: Whether to stream the response
            
        Returns:
            Either a generator yielding response chunks or the complete response string
        """
        # Add user message to session
        user_message = LLMMessage(role="user", content=message)
        self.llm.session.append(user_message)
        
        # Get response from LLM
        response = self.llm.chat_with_context(
            message,
            model=model,
            temperature=temperature,
            stream=stream
        )
        
        if stream:
            buffer = ""
            # Handle streaming response with function calls
            tool_response = self.function_call.handle_stream(response)
            for chunk in tool_response:
                buffer += chunk
                yield chunk
            self.llm.chat_with_context(
                LLMMessage(role="assistant", content=buffer),
                model=model,
                temperature=temperature,
                stream=stream
            )
        else:
            # Handle non-streaming response with function calls
            buffer = self.function_call.handle_completion(response)
            self.llm.chat_with_context(
                LLMMessage(role="assistant", content=buffer),
                model=model,
                temperature=temperature,
                stream=stream
            )
            return buffer
    
    def clear_context(self):
        """Clear the conversation history while maintaining the system message."""
        self.llm.session = [self.system_message]
    
    def add_tool(self, tool: Type[BaseTool]):
        """Add a new tool to the agent's capabilities."""
        self.function_call.add_tool(tool)
        # Update system message with new tool information
        self._update_system_message()
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent's capabilities."""
        if tool_name in self.function_call.tools:
            del self.function_call.tools[tool_name]
            self._update_system_message()
    
    def get_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.function_call.tools.keys())
    
    def update_identity(self, name: str = None, backstory: str = None, goal: str = None):
        """Update agent's identity information."""
        if name is not None:
            self.name = name
        if backstory is not None:
            self.backstory = backstory
        if goal is not None:
            self.goal = goal
        self._update_system_message()
