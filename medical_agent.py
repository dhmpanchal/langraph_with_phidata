"""
medical_agent.py

Agentic RAG pipeline for extracting patient medical conditions
using:
- Groq LLM
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
from vector_helper import VectorHelper

from dotenv import load_dotenv
load_dotenv()


# ==========================
# Configuration
# ==========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOP_K = 50

# ==========================
# Pydantic Models
# ==========================

class MajorConditions(BaseModel):
    key: str = Field(description="The name of the condition (e.g., IBS, Depression, Diabetes)")
    value: str = Field(description="A brief reason, context, or supporting detail (e.g., date diagnosed, symptoms, related medication, or history)")
    start_date: str = Field(description="The start date of the condition in MM-DD-YYYY format mention in the patient information")
    end_date: str = Field(description="The end date of the condition in MM-DD-YYYY format mention in the patient information")
    status: str = Field(description="The status of the condition (e.g., ongoing, cleaned up)")

class DiagnosisExtraction(BaseModel):
    major_conditions: List[MajorConditions] = Field(default_factory=list)


# ==========================
# LLM Setup (Groq)
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

@tool("patient_document_search", description="Retrieve patient information regarding the all medical history")
def rag_patient_retrieval(query: str) -> str:
    """
    Retrieve patient information from PGVector
    """
    docs = vectorstore.search_knowledge_base(query, k=TOP_K, filter={"source": "data/text_files/froster.txt"})
    if not docs:
        return ""

    return "\n\n".join(doc.page_content for doc in docs)


tools = [rag_patient_retrieval]


# ==========================
# Agent Prompt (Decision Only)
# ==========================

agent_system_prompt = SystemMessage(
    content="""
You are a medical reasoning assistant.

Your work is to retrieve the patient information base on the user question from the vector database using the RAG tool.
Identify the query and create the described RAG search query to retrive the relivent information from the document. 
After getting chunks of the document from the RAG tool, fulfil the user question and answer it.
"""
)


# ==========================
# Agent Creation
# ==========================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=agent_system_prompt,
)


# ==========================
# TASK PROMPT (YOUR ORIGINAL PROMPT)
# ==========================

TASK_PROMPT = """
    You are clinical experts who review the given patient infromation give below as a context. Your task is to extract a list of all major or chronic medical conditions mentioned in the given patient infromations and need to find dates of the medical conditions from when it detected and calculate the proper dates in formats and from when it was cleaned up or still it is on going.

    For each condition, provide:
    - key: The name of the condition (e.g., IBS, Depression, Diabetes)
    - value: A brief reason, context, or supporting detail
    - start_date: The start date of the condition mention in the patient information
    - end_date: The end date of the condition mention in the patient information
    - status: ongoing or cleaned up

    Return STRICT JSON:

    {
    "major_conditions": [
        {
        "key": "Condition Name",
        "value": "Supporting detail or reason",
        "start_date": "same as in the patient information",
        "end_date": "same as in the patient information",
        "status": "ongoing/cleaned up"
        }
    ]
    }

    If no major conditions are found:
    {
    "major_conditions": []
    }

    context:
    {patient_info}

    Ensure that the output is strictly in JSON format without any additional text.
"""

prompt=PromptTemplate(
    input_variables=["patient_info"],
    template=TASK_PROMPT
)

parser = JsonOutputParser()
task_chain = prompt | llm | parser


# ==========================
# Public Function
# ==========================

def run_medical_agent(user_query: str, file_number: int) -> str:
    """
    Main entry point for Agentic RAG medical extraction
    """

    # Step 1: Agent decides & retrieves context
    agent_result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=user_query)
            ]
        },
        config=RunnableConfig(
            tags=[f"doc:{file_number}"]
        )
    )

    print(f"agent_result: {agent_result}")

    return agent_result


# ==========================
# Example Usage
# ==========================

# if __name__ == "__main__":
#     query = """
#     Retrieve the patient's comprehensive history of medical diagnoses,
#     including the diagnosis name, the specific date it was established,
#     and the current status of each condition.
#     """

#     result = run_medical_agent(query)
#     print(result)
