import os
from typing import List, Optional, Dict, Tuple
from phi.embedder.base import Embedder
from langsmith_wrapper import LangSmithTracker
import dotenv
dotenv.load_dotenv()

# Import for Hugging Face embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

# Keep boto3 imports for backward compatibility (if anyone still needs AWS)
try:
    import boto3
    import json
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class HuggingFaceEmbedder(Embedder):
    """Custom embedder for Hugging Face models with LangSmith tracking"""
    
    model_id: str = "abhinand/MedEmbed-base-v0.1"
    dimensions: int = 768  # Standard BERT-base dimension
    tracker: Optional[LangSmithTracker] = None
    model_instance: Optional[object] = None  # The actual SentenceTransformer model
    
    class Config:
        arbitrary_types_allowed = True  # Allow SentenceTransformer type
    
    def __init__(
        self,
        model_id: str = "abhinand/MedEmbed-base-v0.1",
        dimensions: Optional[int] = None,
        tracker: Optional[LangSmithTracker] = None,
        device: Optional[str] = None,  # 'cuda', 'cpu', or None for auto
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceEmbedder. "
                "Install it with: pip install sentence-transformers"
            )
        
        # Initialize the model first (before calling super().__init__)
        print(f"📥 Loading Hugging Face model: {model_id}...")
        model_instance = SentenceTransformer(model_id, device=device)
        
        # Get actual dimensions from the model
        actual_dimensions = model_instance.get_sentence_embedding_dimension()
        
        # Use provided dimensions or auto-detect
        if dimensions is None:
            dimensions = actual_dimensions
        elif dimensions != actual_dimensions:
            print(f"⚠️  Warning: Specified dimensions ({dimensions}) differ from model dimensions ({actual_dimensions})")
            print(f"   Using model dimensions: {actual_dimensions}")
            dimensions = actual_dimensions
        
        # Initialize parent Pydantic model with all fields
        super().__init__(
            dimensions=dimensions,
            model_id=model_id,
            tracker=tracker,
            model_instance=model_instance
        )
        
        print(f"✅ Model loaded successfully! Embedding dimensions: {self.dimensions}")
    
    @property
    def model(self) -> str:
        """Return model_id as model for compatibility with chunking strategies"""
        return self.model_id
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            # Generate embedding
            embedding = self.model_instance.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list of floats
            embedding_list = embedding.tolist()
            
            # Track with LangSmith
            if self.tracker and self.tracker.enabled:
                self.tracker.track_embedding(
                    text=text[:200],  # Only log first 200 chars
                    model_id=self.model_id,
                    embedding_dim=len(embedding_list)
                )
            
            return embedding_list
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise
    
    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        """Get embedding and usage information (required by phidata)"""
        try:
            embedding = self.get_embedding(text)
            
            # Estimate token usage (approximate)
            usage = {
                "prompt_tokens": len(text.split()),
                "total_tokens": len(text.split())
            }
            
            return embedding, usage
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (batched for efficiency)"""
        try:
            # Batch encode for efficiency
            embeddings = self.model_instance.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10  # Show progress for large batches
            )
            
            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            return embeddings_list
            
        except Exception as e:
            print(f"❌ Error getting embeddings: {e}")
            raise


# Keep BedrockTitanEmbedder for backward compatibility
class BedrockTitanEmbedder(Embedder):
    """Custom embedder for AWS Bedrock Titan with LangSmith tracking"""
    
    model_id: str = "amazon.titan-embed-text-v2:0"
    region_name: str = "eu-west-2"
    dimensions: int = 1024
    tracker: Optional[LangSmithTracker] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    client: Optional[object] = None
    
    @property
    def model(self) -> str:
        """Return model_id as model for compatibility with chunking strategies"""
        return self.model_id
    
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: str = "eu-west-2",
        dimensions: int = 1024,
        tracker: Optional[LangSmithTracker] = None,
        access_key_id = os.environ.get("AWS_ACCESS_KEY"),
        secret_access_key = os.environ.get("AWS_SECRET_KEY"),
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for BedrockTitanEmbedder. "
                "Install it with: pip install boto3"
            )
        
        # Get credentials from env if not provided
        if access_key_id is None:
            access_key_id = os.environ.get("AWS_ACCESS_KEY")
        if secret_access_key is None:
            secret_access_key = os.environ.get("AWS_SECRET_KEY")
            
        # Initialize parent Pydantic model (only pass dimensions to parent)
        super().__init__(dimensions=dimensions)
        
        # Set our custom attributes
        self.model_id = model_id
        self.region_name = region_name
        self.tracker = tracker
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        
        # Initialize Bedrock client with optional credentials
        client_kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self.region_name
        }
        
        # Only add credentials if explicitly provided
        if self.access_key_id and self.secret_access_key:
            client_kwargs['aws_access_key_id'] = self.access_key_id
            client_kwargs['aws_secret_access_key'] = self.secret_access_key
        
        self.client = boto3.client(**client_kwargs)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                raise ValueError("No embedding returned from Bedrock")
            
            # Track with LangSmith
            if self.tracker and self.tracker.enabled:
                self.tracker.track_embedding(
                    text=text[:200],  # Only log first 200 chars
                    model_id=self.model_id,
                    embedding_dim=len(embedding)
                )
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise
    
    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        """Get embedding and usage information (required by phidata)"""
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding:
                raise ValueError("No embedding returned from Bedrock")
            
            # Track with LangSmith
            if self.tracker and self.tracker.enabled:
                self.tracker.track_embedding(
                    text=text[:200],
                    model_id=self.model_id,
                    embedding_dim=len(embedding)
                )
            
            # Return embedding and usage info
            # Titan doesn't provide detailed token usage, so we estimate
            usage = {
                "prompt_tokens": len(text.split()),
                "total_tokens": len(text.split())
            }
            
            return embedding, usage
            
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

