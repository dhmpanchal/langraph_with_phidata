import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from langsmith import Client
from langsmith.run_helpers import traceable
import json

class LangSmithTracker:
    """Wrapper for LangSmith tracking"""
    
    def __init__(self, project_name: Optional[str] = None):
        self.enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "default")
        
        if self.enabled:
            try:
                self.client = Client()
                print(f"✅ LangSmith tracking enabled for project: {self.project_name}")
            except Exception as e:
                print(f"⚠️  LangSmith initialization failed: {e}")
                self.enabled = False
        else:
            self.client = None
            print("ℹ️  LangSmith tracking disabled")
    
    @traceable(run_type="retriever", name="Knowledge Base Search")
    def track_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        num_results: int
    ) -> Dict[str, Any]:
        """Track knowledge base search operations"""
        return {
            "query": query,
            "num_results_requested": num_results,
            "num_results_found": len(results),
            "results": results
        }
    
    @traceable(run_type="embedding", name="Generate Embeddings")
    def track_embedding(
        self,
        text: str,
        model_id: str,
        embedding_dim: int
    ) -> Dict[str, Any]:
        """Track embedding generation"""
        return {
            "text_length": len(text),
            "model_id": model_id,
            "embedding_dimension": embedding_dim,
            "timestamp": datetime.now().isoformat()
        }
    
    @traceable(run_type="llm", name="Bedrock Mistral Call")
    def track_llm_call(
        self,
        prompt: str,
        response: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track LLM calls"""
        return {
            "model_id": model_id,
            "prompt": prompt,
            "response": response,
            "context_provided": context is not None,
            "context_length": len(context) if context else 0,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": len(prompt.split()),
            "response_tokens": len(response.split()),
            "timestamp": datetime.now().isoformat()
        }
    
    @traceable(run_type="chain", name="RAG Agent Run")
    def track_agent_run(
        self,
        query: str,
        response: str,
        search_results: List[Dict[str, Any]],
        execution_time: float
    ) -> Dict[str, Any]:
        """Track complete agent execution"""
        return {
            "query": query,
            "response": response,
            "num_search_results": len(search_results),
            "execution_time_seconds": execution_time,
            "timestamp": datetime.now().isoformat()
        }

