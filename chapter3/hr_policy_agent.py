from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
# from langchain_openai import AzureChatOpenAI
from langchain_cohere import ChatCohere

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

#-----------------------------------------------------------------------
# Setup the LLM for the HR Policy Agent
# This uses the Azure OpenAI service with a specific deployment
# Please replace the environment variables with your own values
#-----------------------------------------------------------------------

# endpoint = os.getenv("ENDPOINT_URL")
# deployment = os.getenv("DEPLOYMENT_NAME")
# subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version=os.getenv("API_VERSION")

# model=AzureChatOpenAI(
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
#     api_version=api_version,
#     deployment_name=deployment,
# )

# Initialize Cohere model
CO_API_KEY = os.getenv("CO_API_KEY")
model = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=CO_API_KEY)

#-----------------------------------------------------------------------
# Define the HR policy agent that will use the MCP server
# to answer queries about HR policies.
#-----------------------------------------------------------------------
async def run_hr_policy_agent(prompt: str) -> str:

    # Make sure the right path to the server file is passed.
    hr_mcp_server_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), 
                                            "hr_policy_server.py"))
    print("HR MCP server path: ", hr_mcp_server_path)

    # Create the server parameters for the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=[hr_mcp_server_path],
    )

    # Create a client session to connect to the MCP server
    async with stdio_client(server_params) as (read,write):
        async with ClientSession(read,write) as session:
            print("initializing session")
            await session.initialize()

            print("\nloading tools & prompt")
            hr_policy_tools = await load_mcp_tools(session)
            hr_policy_prompt = await load_mcp_prompt(session, 
                                "get_llm_prompt", 
                                arguments={"query": prompt})

            print("\nTools loaded :", hr_policy_tools[0].name)
            print("\nPrompt loaded :", hr_policy_prompt)

            print("\nCreating agent")
            agent=create_react_agent(model,hr_policy_tools)

            print("\nAnswering prompt : ", prompt)
            agent_response = await agent.ainvoke(
                {"messages": hr_policy_prompt})

            return agent_response["messages"][-1].content

    return "Error"

if __name__ == "__main__":
    # Run the HR policy agent with a sample query
    print("\nRunning HR Policy Agent...")
    response = asyncio.run(
        run_hr_policy_agent("What is the policy on remote work?"))

    print("\nResponse: ", response)