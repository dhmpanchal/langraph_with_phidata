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
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator
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
    query: list[str] = Field(description="Medical question or search queries")
    file_path: str = Field(description="Exact file path of the patient document to search")

    @field_validator("query", mode="before")
    @classmethod
    def parse_query_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                import json
                # Handle cases like '["q1", "q2"]'
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                # If it's a single string, wrap it in a list
                return [v]
        return v

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
    temperature=0.3,
)

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.3,
#     max_tokens=1000
# )

# ==========================
# Vector Store (PGVector)
# ==========================
vectorstore = VectorHelper()

# ==========================
# RAG Tool (Retrieval)
# ==========================
def deduplicate_chunks(chunks: list[str]) -> str:
    seen = set()
    unique_chunks = []

    if not chunks:
        return ""

    for chunk in chunks:
        clean_chunk = chunk.strip()
        if clean_chunk and clean_chunk not in seen:
            seen.add(clean_chunk)
            unique_chunks.append(clean_chunk)

    return "\n\n".join(unique_chunks)

@tool("patient_document_search", description="Retrieve patient information regarding the all medical history",args_schema=PatientSearchInput)
def rag_patient_retrieval(query: list, file_path: str) -> str:
    """
    Retrieve patient information from PGVector
    """
    similarity_threshold = 0.8
    all_chunks = []

    print(f"Retrieval Query: {query}")
    print(f"file_path: {file_path}")

    for q in query:
        docs = vectorstore.search_with_cosine_similarity(q, k=TOP_K, filter={"source": file_path})
        print(f"Found {len(docs)} documents for query: {q}")
        
        for doc, score in docs:
            if score is not None and score >= similarity_threshold:
                all_chunks.append(doc.page_content)

    final_context = deduplicate_chunks(all_chunks)
    return final_context

tools = [rag_patient_retrieval]
parser = PydanticOutputParser(pydantic_object=DiagnosisExtraction)

# ==========================
# Agent Prompt (Decision Only)
# ==========================
agent_system_prompt = SystemMessage(
    content=f"""
You are an expert medical reasoning assistant. Your goal is to extract active diagnoses and chronic medical conditions from the patient's medical record.

WORKFLOW:
1. Generate up to 3 focused semantic search queries to find the patient's medical history and active conditions.
2. Call the 'patient_document_search' tool with these query and the provided 'file_path'.
    - query = your generated semantic search list of queries as an array like : ["query 1","query 2","query 3"]. Need to create maximum 3 queries only and all the queries are unique and relevent to the user request.
    - file_path = the patient document path provided in the conversation context
3. Based ONLY on the retrieved context, extract all major or chronic medical conditions.

For each condition, provide:
- key: name of the condition
- value: specific supporting detail or symptom from context
- start_date: original diagnosis date (MM-DD-YYYY) or "Unknown"
- end_date: resolution date or "ongoing"
- status: "ongoing" or "cleaned up"

{parser.get_format_instructions()}

IMPORTANT:
- If no conditions are found, return an empty list for 'major_conditions'.
- Maintain strict JSON format. No conversational filler or markdown outside the JSON block.
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
        # Attempt direct parsing
        return parser.parse(last_message)
    except Exception as e:
        print(f"Primary parsing failed: {e}. Attempting regex recovery...")
        # Attempt to find JSON in the string
        json_match = re.search(r'\{.*\}', last_message, re.DOTALL)
        if json_match:
            try:
                # Use standard json parse if pydantic parser is too strict on the fragment
                import json
                data = json.loads(json_match.group())
                return DiagnosisExtraction(**data)
            except Exception as e2:
                print(f"Regex recovery failed: {e2}")
        
        return last_message