"""
medical_agent2.py

Agentic RAG pipeline for extracting patient medical conditions.
Create an agent that create vector query and fetch the relevant information from the vector database.
using:
- Ollama LLM
- LangChain Agent + Tools
- PGVector Vector Store
"""

# ==========================
# Imports
# ==========================
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

import os
import re
from vector_helper import VectorHelper

from dotenv import load_dotenv
load_dotenv()

# ==========================
# Configuration
# ==========================
TOP_K = 10

# ==========================
# Pydantic Models
# ==========================
class PatientSearchInput(BaseModel):
    query: str = Field(description="Medical question or search query")
    file_path: str = Field(description="Exact file path of the patient document to search")

# ==========================
# LLM Setup (Ollama)
# ==========================
llm = ChatOpenAI(
    model="llama3.2",
    base_url="http://localhost:11434/v1",  # Ollama endpoint
    api_key="ollama",
    temperature=0
)

# ==========================
# Vector Store (PGVector)
# ==========================
vectorstore = VectorHelper()

# ==========================
# RAG Tool (Retrieval)
# ==========================
@tool("patient_document_search", description="Retrieve patient information regarding the all medical history",args_schema=PatientSearchInput)
def rag_patient_retrieval(query: str, file_path: str) -> str:
    """
    Retrieve patient information from PGVector
    """
    docs = vectorstore.search_knowledge_base(query, k=TOP_K, filter={"source": file_path})
    print(f"file_path: {file_path}")
    print(f"Retrieval Query: {query}")
    print(f"Found {len(docs)} documents for query.")
    
    if not docs:
        return ""

    return "\n\n".join(doc.page_content for doc in docs)

tools = [rag_patient_retrieval]

# ==========================
# Agent Prompt (Decision Only)
# ==========================
agent_system_prompt = SystemMessage(
    content="""
You are a retrieval agent.

Your work is to retrieve the patient information based on the user question from the vector database using the RAG tool ** patient_document_search **

Identify the query and create the described RAG search query to retrieve the relevant information from the document. 

The tool requires:
- query → the medical question
- file_path → the patient document path provided in the conversation

Always extract the file_path from the conversation and pass it to the tool.
You must ALWAYS call the patient_document_search tool.
Return ONLY the tool output. Do not summarize. Do not answer.
"""
)

# ==========================
# Create Agent that create a vector query and fetch the relevant information from the vector database using the RAG tool
# ==========================
rag_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=agent_system_prompt,
)

# ==========================
# Public Functions
# ==========================
def run_rag_agent(user_query: str, file_number: int):
    """
    Agent that create a vector query and fetch the relevant information from the vector database using the RAG tool
    """

    # Step 1: Agent decides & retrieves context
    agent_result = rag_agent.invoke(
        {
            "messages": [
                HumanMessage(content=user_query)
            ]
        },
        config=RunnableConfig(
            tags=[f"doc:{file_number}"]
        )
    )

    # Step 2: Extract content from the last message
    last_message = agent_result["messages"][-1].content

    # Step 3: Parse the content into the Pydantic model
    try:
        # Using JsonOutputParser to handle possible markdown wrapper or extra text
        parsed_json = parser.parse(last_message)
        return DiagnosisExtraction(**parsed_json)
    except Exception as e:
        print(f"Error parsing agent output to Pydantic: {e}")
        # Attempt to find JSON in the string if direct parsing fails
        json_match = re.search(r'\{.*\}', last_message, re.DOTALL)
        if json_match:
            try:
                parsed_json = parser.parse(json_match.group())
                return DiagnosisExtraction(**parsed_json)
            except:
                pass
        return last_message