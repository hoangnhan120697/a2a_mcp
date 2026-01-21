# a2a_client_router_agent.py

from a2a.client import A2AClient
from typing import Any
import httpx
from uuid import uuid4
from a2a.types import (
    SendMessageRequest,
    MessageSendParams,
    SendStreamingMessageRequest,
)

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, AnyMessage
import operator
import asyncio
import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
import uuid
import json

load_dotenv()

# FIX: Changed from AzureChatOpenAI to ChatCohere to avoid Azure connection issues
# Original used Azure OpenAI which required ENDPOINT_URL, DEPLOYMENT_NAME, etc.
CO_API_KEY = os.getenv("CO_API_KEY")
model = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=CO_API_KEY)

# ---------------------------------------------------------------
# Generic method to invoke a remote agent with A2A
# ---------------------------------------------------------------


async def execute_a2a_agent(agent_card_url: str,
                            user: str,
                            prompt: str) -> str:

    print("Retrieving agent card at ", agent_card_url)
    
    # FIX: Extended timeout to 90 seconds (from 30s) to handle slower LLM responses
    # Added pool timeout to prevent connection pool issues
    timeout = httpx.Timeout(90.0, connect=10.0, pool=10.0)
    
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:

        client = await A2AClient.get_client_from_agent_card_url(
            httpx_client, agent_card_url
        )
        print("Agent URL received :", client.url)

        input_dict = {"user": user, "prompt": prompt}

        send_message_payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "text", "text": json.dumps(input_dict)},
                ],
                "messageId": uuid4().hex,
            },
        }

        print("prompting agent ", client.url)
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )
        response = await client.send_message(request)

        # Extract text from the response object
        response_json = response.model_dump(mode='json', exclude_none=True)
        text = response_json.get("result").get("parts")[0].get("text")
        print("Response from agent = ", text)
        return text


# ---------------------------------------------------------------
# LangGraph based router (main ) agent that can route a prompt
# to the appropriate agent
# ---------------------------------------------------------------
class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class RouterHRAgent:

    def __init__(self, model, system_prompt, user, debug=False):

        self.system_prompt = system_prompt
        self.model = model
        self.debug = debug
        self.user = user

        router_graph = StateGraph(RouterAgentState)
        router_graph.add_node("Router", self.call_llm)
        router_graph.add_node("Policy_Agent", self.policy_agent_node)
        router_graph.add_node("Timeoff_Agent", self.timeoff_agent_node)
        router_graph.add_node("Unsupported_functions", self.unsupported_node)

        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {"POLICY": "Policy_Agent",
             "TIMEOFF": "Timeoff_Agent",
             "UNSUPPORTED": "Unsupported_functions"}
        )

        # One way routing, not coming back to router
        router_graph.add_edge("Policy_Agent", END)
        router_graph.add_edge("Timeoff_Agent", END)
        router_graph.add_edge("Unsupported_functions", END)

        # Set where there graph starts
        router_graph.set_entry_point("Router")
        self.router_graph = router_graph.compile()

    def call_llm(self, state: RouterAgentState):
        messages = state["messages"]

        if self.debug:
            print(f"Call LLM received {messages}")

        # If system prompt exists, add to messages in the front
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        # invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Call LLM result {result}")
        return {"messages": [result]}

    # FIX: Changed from synchronous 'def' to 'async def' to avoid event loop errors
    # Original used asyncio.run() which creates nested event loops (not allowed)
    # Now uses 'await' to work within the same event loop
    async def policy_agent_node(self, state: RouterAgentState):
        messages = state["messages"]
        # Call the policy agent
        prompt = messages[0].content
        print(f"Policy agent node received {prompt}")

        # FIX: Changed from asyncio.run() to await
        response = await execute_a2a_agent("http://localhost:9001",
                                           self.user, prompt)

        if self.debug:
            print(f"Policy agent node response : {response}")

        return {"messages": [AIMessage(content=response)]}

    # FIX: Changed from synchronous 'def' to 'async def' (same fix as policy_agent_node)
    async def timeoff_agent_node(self, state: RouterAgentState):
        messages = state["messages"]

        # Call the timeoff agent
        prompt = messages[0].content
        print(f"Timeoff agent node received {prompt}")

        # FIX: Changed from asyncio.run() to await
        response = await execute_a2a_agent("http://localhost:9002",
                                           self.user, prompt)
        if self.debug:
            print(f"Timeoff agent node response : {response}")

        return {"messages": [AIMessage(content=response)]}

    def unsupported_node(self, state: RouterAgentState):
        messages = state["messages"]

        print("Unsupported node invoked")

        response = """Sorry, I cannot help you with this request.
        I only support HR policy queries and timeoff requests.
        Please contact your HR representative for assistance."""

        if self.debug:
            print(f"Unsupported node response : {response}")

        return {"messages": [AIMessage(content=response)]}

    def find_route(self, state: RouterAgentState):
        last_message = state["messages"][-1]
        if self.debug:
            print("Router: Last result from LLM : ", last_message)

        # Set the last message as the destination
        destination = last_message.content

        print(f"Destination chosen : {destination}")
        return destination


if __name__ == "__main__":

    # FIX: Wrapped everything in async main() function and use single asyncio.run()
    # Original had multiple blocking calls which caused event loop conflicts
    # This ensures all async operations run in the same event loop
    async def main():
        try:
            # Create the chatbot
            # Select user
            user = "Alice"
            # Setup the system prompt
            system_prompt = """ 
            You are a Router, that analyzes the input query and chooses 3 options:
            POLICY: If the query is about HR policies, like leave, remote work, etc.
            TIMEOFF: If the query is about time off requests, both creating requests and checking balances
            UNSUPPORTED: Any other query that is not related to HR policies or time off requests.
        
            The output should only be just one word out of the possible 3 : POLICY, TIMEOFF, UNSUPPORTED.
            """

            router_hr_agent = RouterHRAgent(model,
                                            system_prompt,
                                            user,
                                            debug=False)

            # To print the graph
            # graph_image=router_hr_agent.router_graph.get_graph().draw_mermaid_png()
            # with open("chapter6/router_agent.png", "wb") as f:
            #     f.write(graph_image)

            # Send a sequence of messages to chatbot and get its response
            # This simulates the conversation between the user and the Agentic chatbot
            user_inputs = [
                "Tell me about payroll processing",
                "What is the policy for remote work?",
                "What is my vacation balance?",
                "File a time off request for 5 days starting from 2025-05-05",
                "What is vacation balance now?",
            ]

            # Create a new thread
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            for input in user_inputs:
                print(f"----------------------------------------\nUSER : {input}")
                # Format the user message
                user_message = {"messages": [HumanMessage(input)]}
                # FIX: Changed from invoke() to ainvoke() because we have async nodes
                # LangGraph requires ainvoke() when any node in the graph is async
                ai_response = await router_hr_agent.router_graph.ainvoke(
                    user_message, config=config)
                # Print the response
                print(f"\nAGENT : {ai_response['messages'][-1].content}")

        except Exception as e:
            import traceback
            print(f"An error occurred: {e}")
            print(f"\nError type: {type(e).__name__}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("\n" + "="*80)
            print("Make sure all servers are running:")
            print("  - HR Policy Agent on port 9001")
            print("  - Timeoff Agent on port 9002")
            print("="*80)
    
    # FIX: Single asyncio.run() call for all operations (instead of multiple calls)
    asyncio.run(main())
