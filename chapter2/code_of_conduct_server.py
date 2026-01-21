#-----------------------------------------------------------------------
#MCP Server that serves the code_of_conduct.pdf file
# as a MCP resource
#-----------------------------------------------------------------------

import os
from dotenv import load_dotenv
# https://gofastmcp.com/
from fastmcp import FastMCP
import PyPDF2

#-----------------------------------------------------------------------
#Setup the MCP Server
#-----------------------------------------------------------------------
load_dotenv()
hr_coc_mcp = FastMCP("HR-CoC-MCP-Server")

#-----------------------------------------------------------------------
#Setup Resources
#-----------------------------------------------------------------------
pdf_filename = "code_of_conduct.pdf"
pdf_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), pdf_filename))
pdf_uri = f"file:///{pdf_full_path.replace(os.sep, '/')}"

#Decorator to register the resource with the MCP server
@hr_coc_mcp.resource(
    uri=pdf_uri,
    name="Code of Conduct",
    description="Provides code of conduct policies for the company",
    mime_type="text/plain",
)
#Function to handle requests for the code of conduct
def get_code_of_conduct() -> str:
    """Returns the text content of the code of conduct PDF file."""

    #Open the file and read its contents
    with open(pdf_full_path, "rb") as code_of_conduct:
        reader = PyPDF2.PdfReader(code_of_conduct)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

#Code to test the server standalone
#print(get_code_of_conduct())
#-----------------------------------------------------------------------
#Run the Server
#-----------------------------------------------------------------------
if __name__ == "__main__":
    hr_coc_mcp.run(transport="stdio")
