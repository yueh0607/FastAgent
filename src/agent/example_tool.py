from agent import Agent, Tool

def search_web(query: str) -> str:
    """Example function to simulate web search"""
    return f"Search results for: {query}"

def main():
    # Create an agent
    agent = Agent()
    
    # Create a search tool
    search_tool = Tool(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        },
        function=search_web
    )
    
    # Add the tool to the agent
    agent.add_tool(search_tool)
    
    # Example usage
    user_input = "Search for information about Python"
    response = agent.process_input(user_input)
    print(response)
    
    # Execute the tool
    result = agent.execute_tool("search_web", {"query": "Python programming"})
    print(result)

if __name__ == "__main__":
    main() 