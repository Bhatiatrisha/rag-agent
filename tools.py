# tools.py
from langchain_core.tools import tool
from retriever import retrieve_docs

@tool
def search_documents(query: str) -> str:
    """Search the internal document database for relevant information.
    Use this when the user asks about telecom, LTE, 5G, SDN, NFV, or RAG."""
    docs = retrieve_docs(query, top_k=3)
    return "\n\n".join(docs)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. 
    Use this for any numerical calculations.
    Example input: '300 * 1024' or '20 / 4'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool  
def get_current_info(topic: str) -> str:
    """Get current status or general info when no documents are available.
    Use this as a fallback when search_documents returns no relevant results."""
    return f"No specific internal documents found for '{topic}'. Answering from general knowledge."

# Export all tools as a list
tools = [search_documents, calculate, get_current_info]