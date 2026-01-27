
from data_ingestion import DataIngestion

def main():
    file_path = "data/pdf/85jkr_harrypotter_1.pdf"

    # 1. Data ingestion
    data_ingestion = DataIngestion(file_path)
    data_ingestion.run_data_ingestion_pipeline()


if __name__ == "__main__":
    main()
