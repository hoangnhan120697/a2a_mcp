import os
from dotenv import load_dotenv
from fastmcp import FastMCP

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------------------------------------------------
# Setup the MCP Server
# -----------------------------------------------------------------------
load_dotenv()
hr_policies_mcp = FastMCP("HR-Policies-MCP-Server")

# -----------------------------------------------------------------------
# Setup the Vector Store for use in retrieving policies
# This will use the hr_policy_document.pdf file as its source
# -----------------------------------------------------------------------

pdf_filename = "hr_policy_document.pdf"
pdf_full_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), pdf_filename))

# Load and split the PDF document
loader = PyPDFLoader(pdf_full_path)
policy_documents = loader.load_and_split()

# Create embeddings
policy_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create In memory vector store
policy_vector_store = InMemoryVectorStore.from_documents(
    policy_documents, policy_embeddings)

# -----------------------------------------------------------------------
# Setup the MCP tool to query for policies, given a user query string
# -----------------------------------------------------------------------


@hr_policies_mcp.tool()
def query_policies(query: str):
    """Query the HR policies document for information about
    leave, timeoff, benefits, work hours, remote work and 
    workplace conduct policies"""

    # Perform a similarity search in the vector store
    results = policy_vector_store.similarity_search(query, k=3)
    return results

# -----------------------------------------------------------------------
# Setup the MCP prompt to dynamically generate the prompt for the LLM
# using the input query.
# -----------------------------------------------------------------------


@hr_policies_mcp.prompt()
def get_llm_prompt(query: str) -> str:
    """Generates a a prompt for the LLM to use to answer the query"""

    return f"""
    You are a helpful HR assistant. Answer the following query about HR policies
    by only using the tools provided to you. Do not make up any information.

    Query: {query}
    """

# -----------------------------------------------------------------------
# Run the policy Server
# -----------------------------------------------------------------------

# test
# print(query_policies("What is the policy on remote work?"))


if __name__ == "__main__":
    hr_policies_mcp.run(transport="stdio")
