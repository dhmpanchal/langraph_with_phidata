
from medical_agent2 import run_rag_agent
from medical_agent import run_medical_agent
from data_ingestion import DataIngestion

def main():
    # file_path = "data/pdf/85jkr_harrypotter_1.pdf"
    # file_path = "data/text_files/froster.txt"
    file_path = "data/pdf/froster.pdf"

    # 1. Data ingestion
    # data_ingestion = DataIngestion(file_path)
    # data_ingestion.run_data_ingestion_pipeline()

    # 2. Retrieval Pipeline using Agent tool calls
    user_query = """
    Retrieve the patient's comprehensive history of medical diagnoses,
    including the diagnosis name, the specific date it was established,
    and the current status of each condition.
    """

    final_query = f"""
    USER QUESTION:
    {user_query}

    PATIENT FILE PATH:
    {file_path}
    """

    # Agent that create vector query and fetch the relevant information from the vector database.
    # and feed that information to the LLM to extract the relevant information.
    result = run_medical_agent(final_query, file_number=4001)
    # print(result)

    # Agent that create vector query and fetch the relevant information from the vector database.
    # patient_info = run_rag_agent(final_query, file_number=4001)
    print("======= Patient Information =======")
    print(f"Raw agent output: {result}")
    print("===================================")

if __name__ == "__main__":
    main()
