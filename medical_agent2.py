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
from langchain_core.output_parsers import PydanticOutputParser

import os
import re
from vector_helper import VectorHelper
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_result


from dotenv import load_dotenv
load_dotenv()

# ==========================
# Configuration
# ==========================
TOP_K = 1000

# ==========================
# Pydantic Models
# ==========================
class PatientSearchInput(BaseModel):
    query: str = Field(description="Medical question or search query")
    file_path: str = Field(description="Exact file path of the patient document to search")

class MajorConditions(BaseModel):
    key: str = Field(description="The name of the condition (e.g., IBS, Depression, Diabetes)")
    value: str = Field(description="A brief reason, context, or supporting detail (e.g., date diagnosed, symptoms, related medication, or history)")
    start_date: str = Field(description="The start date of the condition in MM-DD-YYYY format mention in the patient information")
    end_date: str = Field(description="The end date of the condition in MM-DD-YYYY format mention in the patient information")
    status: str = Field(description="The status of the condition (e.g., ongoing, cleaned up)")

class DiagnosisExtraction(BaseModel):
    major_conditions: List[MajorConditions] = Field(default_factory=list)


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
def rag_patient_retrieval(query: list, file_path: str) -> str:
    """
    Retrieve patient information from PGVector
    """
    similarity_threshold = 0.8
    docs = vectorstore.search_with_cosine_similarity(query[0], k=TOP_K, filter={"source": file_path})
    print(f"file_path: {file_path}")
    print(f"Retrieval Query: {query}")
    print(f"Found {len(docs)} documents for query.")
    
    if not docs:
        return ""

    filtered_docs = []
    for doc, score in docs:
        # logger.error(f"Chunk has score {score:.4f}")
        if score is not None and score >= similarity_threshold:
            print(f"Chunk has score {score:.2f}")
            filtered_docs.append(doc)

    return "\n\n".join(doc.page_content for doc in filtered_docs)

tools = [rag_patient_retrieval]
parser = PydanticOutputParser(pydantic_object=DiagnosisExtraction)

# ==========================
# Agent Prompt (Decision Only)
# ==========================
agent_system_prompt = SystemMessage(
    content="""
You are a medical retrieval and reasoning agent.

Your job is to answer the user’s medical question by retrieving information from a patient document using the tool patient_document_search.

You MUST follow this retrieval workflow EXACTLY:

RETRIEVAL WORKFLOW

Step 1 — Query Generation
• Read the user request carefully
• Generate a focused semantic multiple search query to retrieve relevant medical information
• Call the tool patient_document_search with:

query = your generated semantic search list of queries as an array like : ["query 1","query 2","query 3"]. Need to create maximum 3 queries only and all the queries are unique and relevent to the user request.
file_path = the patient document path provided in the conversation context

Step 2 — Multi-Attempt Retrieval
You MUST perform retrieval THREE TIMES total.

For each attempt:
• Reformulate the query differently to capture missing medical context
• Each query should target different aspects, such as:

diagnoses

chronic conditions

medical history timelines

resolved vs ongoing conditions

You MUST call the tool in all 3 attempts before producing a final answer.

Step 3 — Context Review
After the 3 tool calls:
• Collect all returned text chunks
• Remove duplicate or near-duplicate chunks
• Merge the remaining unique chunks into ONE combined context block

Do NOT answer before all 3 retrieval attempts are completed.

REASONING TASK

You are given patient medical context. From this, extract ALL major or chronic medical conditions.

For each condition identify:
• key — Condition name
• value — Short supporting detail from context
• start_date — When condition was first noted (MM-DD-YYYY)
• end_date — When resolved OR "ongoing"
• status — "ongoing" or "cleaned up"

If a date is unclear, infer cautiously from the context. Do NOT invent facts.

OUTPUT FORMAT (STRICT)

Return ONLY valid JSON in this exact format:

{
"major_conditions": [
{
"key": "Condition Name",
"value": "Brief reason or supporting detail",
"start_date": "MM-DD-YYYY",
"end_date": "MM-DD-YYYY or ongoing",
"status": "ongoing or cleaned up"
}
]
}

If no conditions are found:

{
"major_conditions": []
}

Do NOT include explanations, markdown, notes, or extra text. ONLY output JSON.
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
        return parsed_json
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