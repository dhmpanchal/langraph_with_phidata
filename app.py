
from medica_agent3 import run_task_chain
from medica_agent3 import run_retrieval_agent
from medical_agent2 import run_rag_agent
from medical_agent import run_medical_agent
from data_ingestion import DataIngestion

import re

def write_to_file(file_path: str, content: str):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Content written to {file_path}")

# read context from file 
def read_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def normalize_text(text: str) -> str:
    """Clean and normalize extracted text while preserving line breaks for context separation."""
    text = text.replace("\xa0", " ")  # remove non-breaking spaces
    # Collapse spaces/tabs, but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)  # normalize multiple newlines into single newline
    return text.strip()

def main():
    # file_path = "data/pdf/85jkr_harrypotter_1.pdf"
    # file_path = "data/text_files/froster.txt"
    file_path = "data/pdf/Document 12 Leonard Asthma (1).pdf"
    # file_path = "data/docs/froster.docx"
    context_file_path = "data/retrieved_contexts/froster_txt_context_single_query.txt"

    # 1. Data ingestion
    data_ingestion = DataIngestion(file_path)
    data_ingestion.run_data_ingestion_pipeline()

    # 2. Retrieval Pipeline using Agent tool calls
    user_query = """
    Extract a list of all major or chronic medical conditions mentioned in the given patient infromations and need to find dates of the medical conditions from when it detected and calculate the proper dates in formats and from when it was cleaned up or still it is on going.
    """

    final_query = f"""
    USER QUESTION:
    {user_query}

    PATIENT FILE PATH:
    {file_path}
    """

    # Agent that create vector query and fetch the relevant information from the vector database.
    # and feed that information to the LLM to extract the relevant information.
    # result = run_medical_agent(final_query, file_number=4001)
    # print(result)

    # Agent that create vector query and fetch the relevant information from the vector database.
    # final_output = run_rag_agent(final_query, file_number=4001)
    # print("======= Final Output =======")
    # print(f"Raw agent output: {final_output}")
    # print("===================================")

    # patient_info = run_retrieval_agent(final_query, file_path)
    # patient_info = normalize_text(patient_info)
    # patient_info = read_from_file(context_file_path)
    # print("======= Patient Information =======")
    # print(patient_info.strip())
    # print("===================================")
    # write_to_file(context_file_path, patient_info.strip())

    # result = run_task_chain(patient_info.strip())
    # print("======= Task Chain Result =======")
    # print(result)
    # print("===================================") 

if __name__ == "__main__":
    main()
