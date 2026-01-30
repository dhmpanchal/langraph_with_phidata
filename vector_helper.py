from datetime import datetime
import os
from langchain_community.vectorstores.pgvector import PGVector

from langchain_text_splitters import RecursiveCharacterTextSplitter
import db_helper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from medical_category_agent import MedicalCategoryAgent
import dotenv
dotenv.load_dotenv()

class VectorHelper:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.connection_string = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
        
        self.vectorstore = PGVector(
            embedding_function=self.embeddings,
            connection_string=self.connection_string,
            collection_name="embeddings"
        )
        self.category_agent = MedicalCategoryAgent()
    
    def create_vectorization(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        file_name: str = ""
    ):
        """
        Chunk the provided text and store embeddings into PostgreSQL using PGVector.

        Returns (success: bool, message: str)
        """
        print(f"start creating chunks...")
        db_helper.ensure_pgvector_extension()

        # Verify AWS credentials are available (avoid empty embeddings)
        # session_creds = boto3.Session(region_name=aws_region).get_credentials()
        # if session_creds is None:
        #     raise RuntimeError(
        #         "AWS credentials not found. Configure AWS_ACCESS_KEY_ID/SECRET, or an IAM role/profile."
        #     )

        # Health check: ensure embedder returns a non-empty vector
        try:
            probe = self.embeddings.embed_query("healthcheck") or []
        except Exception as exc:
            raise RuntimeError(f"Bedrock embeddings failed: {exc}") from exc
        if not probe or len(probe) == 0:
            raise RuntimeError(
                "Bedrock embeddings returned an empty vector. Verify model access and credentials."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ". "],
        )

        metadatas = [{
            'filename': file_name,
            'uploaded_at': datetime.datetime.now().isoformat(),
        }]
        documents = text_splitter.create_documents([text], metadatas=metadatas)

        self.vectorstore.add_documents(documents)

        return True, f"Stored {len(documents)} documents under '{file_name}'."
    
    def create_vectorization_from_documents(
        self,
        documents: list,
        chunk_size: int = 300,
        chunk_overlap: int = 100
    ):
        """
        Chunk the provided text and store embeddings into PostgreSQL using PGVector.

        Returns (success: bool, message: str)
        """
        print(f"start creating chunks...")
        db_helper.ensure_pgvector_extension()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        documents = text_splitter.split_documents(documents)

        # for doc in documents:
        #     category = self.category_agent.classify_chunk(doc.page_content)
        #     doc.metadata["category"] = category

        # print("Categories assigned:", [doc.metadata["category"] for doc in documents])

        self.vectorstore.add_documents(documents)

        return True, f"Stored {len(documents)} documents."
    
    def search_knowledge_base(self, query: str, k: int = 5, filter: dict = None):
        """
        Search the knowledge base for the given query.

        Returns a list of documents most similar to the query.
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def search_with_cosine_similarity(self, query: str, k: int = 5, filter: dict = None):
        """
        Search the knowledge base for the given query.

        Returns a list of documents most similar to the query.
        """
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)