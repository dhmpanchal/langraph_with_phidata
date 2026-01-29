from vector_helper import VectorHelper
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader
from pathlib import Path

class DataIngestion:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vector_helper = VectorHelper()

    def load_document_pdf(self):
        document = PyMuPDFLoader(file_path=str(Path(self.file_path)))
        docs = document.load() 
        return docs
    
    def load_document_text(self):
        document = TextLoader(file_path=str(Path(self.file_path)))
        docs = document.load() 
        return docs

    def run_data_ingestion_pipeline(self):
        # load document
        docs = self.load_document_pdf()
        # docs = self.load_document_text()

        # create vectorization
        success, message = self.vector_helper.create_vectorization_from_documents(docs)
        if success:
            print(message)
        else:
            print(f"Failed to create vectorization: {message}")
        

