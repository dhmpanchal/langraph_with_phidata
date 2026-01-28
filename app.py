
from medical_agent import run_medical_agent
from data_ingestion import DataIngestion

def main():
    # file_path = "data/pdf/85jkr_harrypotter_1.pdf"
    file_path = "data/text_files/froster.txt"

    # 1. Data ingestion
    # data_ingestion = DataIngestion(file_path)
    # data_ingestion.run_data_ingestion_pipeline()

    # 2. Retrieval Pipeline using Agent tool calls
    query = """
    You are clinical experts who review the given patient infromation give below as a context. Your task is to extract a list of all major or chronic medical conditions mentioned in the given patient infromations and need to find dates of the medical conditions from when it detected and calculate the proper dates in formats and from when it was cleaned up or still it is on going.

    For each condition, provide:
    - key: The name of the condition (e.g., IBS, Depression, Diabetes)
    - value: A brief reason, context, or supporting detail
    - start_date: The start date of the condition mention in the patient information
    - end_date: The end date of the condition mention in the patient information
    - status: ongoing or cleaned up
    """

    result = run_medical_agent(query, file_number=4001)
    print(result)

if __name__ == "__main__":
    main()
