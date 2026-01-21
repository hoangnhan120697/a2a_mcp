from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import sys
import os
import json

#Import the HR timeoff Agent implementation in this wrapper
sys.path.append(os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../chapter4')))
import timeoff_agent

class TimeoffAgentExecutor(AgentExecutor):
    "Executes functions of the Timeoff agent."

    def __init__(self):
        print("TimeoffAgentExecutor initialized")
        
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,) -> None:

        user_input = json.loads(context.get_user_input())
        print("prompt received: ", user_input.get("prompt"))
        
        # Call the HR timeoff agent function
        result = await timeoff_agent.run_timeoff_agent(
                        user=user_input.get("user"), prompt=user_input.get("prompt"))
        
        print("Result received: ", result)
        await event_queue.enqueue_event(new_agent_text_message(result))

    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,) -> None:
        
        raise Exception("Not implemented")

if __name__ == "__main__":

    timeoff_skill = AgentSkill(
        id="TimeoffSkill",
        name="Timeoff Agent Skills",
        description="Performs timeoff operations.",
        tags=["HR", "timeoff"],
        examples=[
            "What is my timeoff balance?",
            "Create a timeoff request for 5 days from 30-June-2025",
        ],
    )

    timeoff_agent_card = AgentCard(
        name="HR timeoff Agent",
        description="Performs timeoff operations.",
        url="http://localhost:9002/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[timeoff_skill],
    )
    
    timeoff_request_handler = DefaultRequestHandler(
        agent_executor=TimeoffAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    timeoff_server = A2AStarletteApplication(
        agent_card=timeoff_agent_card,
        http_handler=timeoff_request_handler,
    )

    # Start the Server
    import uvicorn
    uvicorn.run(timeoff_server.build(), 
                host="0.0.0.0", 
                port=9002, 
                log_level="info")
