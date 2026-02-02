
from medica_agent3 import run_task_chain
from medica_agent3 import run_retrieval_agent
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

    patient_info = run_retrieval_agent(final_query, file_path)
    print("======= Patient Information =======")
    print(patient_info.strip())
    print("===================================")

    result = run_task_chain(patient_info.strip())
    print("======= Task Chain Result =======")
    print(result)
    print("===================================") 

if __name__ == "__main__":
    main()
