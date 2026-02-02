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

TOP_K = 1000
vectorstore = VectorHelper()

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


rag_llm = ChatOpenAI(
    model="llama3.2",
    base_url="http://localhost:11434/v1",  # Ollama endpoint
    api_key="ollama",
    temperature=0.3,
)

task_llm = ChatOpenAI(
    model="llama3.2",
    base_url="http://localhost:11434/v1",  # Ollama endpoint
    api_key="ollama",
    temperature=0.3,
    max_tokens=1000,
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)

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

@tool("patient_document_search", description="Retrieve patient information regarding the all medical history",args_schema=PatientSearchInput, return_direct=True)
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

agent_system_prompt = SystemMessage(
    content="""
You are a retrieval-only agent.

Your task:
- Generate up to 3 semantic search queries
- Call `patient_document_search`
- DO NOT analyze or summarize results

The tool output will be returned directly.
"""
)

rag_agent = create_agent(
    model=rag_llm,          # normal llama3.2
    tools=tools,
    system_prompt=agent_system_prompt,
)

def run_retrieval_agent(user_query: str, file_path: str, file_number: int = 4002) -> str:
    result = rag_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"""
                    User question: {user_query}
                    file_path: {file_path}
                    """
                )
            ]
        },
        config=RunnableConfig(
            tags=[f"doc:{file_number}"]
        )
    )

    # Last message MUST be tool output (string)
    return result["messages"][-1].content


def run_task_chain(patient_info: str, file_number: int = 4002):
    TASK_PROMPT = """
        You are clinical experts who review the given patient infromation give below as a context. Your task is to extract a list of all major or chronic medical conditions mentioned in the given patient infromations and need to find dates of the medical conditions from when it detected and calculate the proper dates in formats and from when it was cleaned up or still it is on going.

        For each condition, provide:
        - key: The name of the condition (e.g., IBS, Depression, Diabetes)
        - value: A brief reason, context, or supporting detail
        - start_date: The start date of the condition mention in the patient information
        - end_date: The end date of the condition mention in the patient information
        - status: ongoing or cleaned up

        Return the output strictly following these format instructions:  
        {format_instructions}

        If no major conditions are found:
        {{
        "major_conditions": []
        }}

        context:
        {patient_info}

        Ensure that the output is strictly in JSON format without any additional text.
    """
    parser = PydanticOutputParser(pydantic_object=DiagnosisExtraction)
    format_instructions = parser.get_format_instructions()
    task_llm.bind(format_instructions=format_instructions)

    prompt = ChatPromptTemplate.from_template(TASK_PROMPT)
    chain = prompt | task_llm

    response_text = chain.invoke(
        {"patient_info": patient_info, "format_instructions": format_instructions},
        config=RunnableConfig(tags=[f"doc:{file_number}"]),
    )
    return response_text
