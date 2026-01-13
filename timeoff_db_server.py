import os
from dotenv import load_dotenv
from fastmcp import FastMCP

from timeoff_datastore import TimeOffDatastore

#-----------------------------------------------------------------------
#Setup the MCP Server
#-----------------------------------------------------------------------
load_dotenv()
timeoff_mcp = FastMCP("Timeoff-MCP-Server")

#-----------------------------------------------------------------------
#Initialize the Timeoff Datastore
#-----------------------------------------------------------------------
timeoff_db = TimeOffDatastore()

#Tool to get time off balance for an employee
@timeoff_mcp.tool()
def get_timeoff_balance(employee_name: str) -> str:
    """Get the timeoff balance for the employee, given their name"""

    print("Getting timeoff balance for employee: ", employee_name)
    return timeoff_db.get_timeoff_balance(employee_name)

#Tool to add a time off request for an employee
@timeoff_mcp.tool() 
def request_timeoff(employee_name: str, start_day:str, days: int) -> str:
    """File a  timeoff request for the employee, 
        given their name, start day and number of days"""

    print("Requesting timeoff for employee: ", employee_name)
    return timeoff_db.add_timeoff_request(
                employee_name, start_day, days)  

#Get prompt for the LLM to use to answer the query
@timeoff_mcp.prompt()
def get_llm_prompt(user: str, prompt: str) -> str:
    """Generates a a prompt for the LLM to use to answer the query
    give a user and a query"""
    print("Generating prompt for user: ", user)
    return f"""
    You are a helpful timeoff assistant. 
    Execute the action requested in the query using the tools provided to you.
    Action: {prompt}
    The tasks need to be executed in terms of the user {user}
    
    """

#-----------------------------------------------------------------------
#Run the Timeoff Server
#-----------------------------------------------------------------------

# Test code
#print("Time off balance for Alice: ", get_timeoff_balance("Alice"))
#print("Add time off request for Alice: ", request_timeoff("Alice", "2025-05-05",5))
#print("New Time off balance for Alice: ", get_timeoff_balance("Alice"))

if __name__ == "__main__":
    timeoff_mcp.run(transport="streamable-http",
                    host="localhost",
                    port=8000,
                    path="/",
                    log_level="debug")