import os
import time
import uuid
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from fastapi import Request
from langfuse import Langfuse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp



# Load environment variables
load_dotenv()

class LLMMonitor:
    """Class for monitoring LLM calls with Langfuse."""
    
    def __init__(self):
        """Initialize Langfuse client."""
        langfuse = Langfuse(
            secret_key="sk-lf-11f8a463-bc58-4c9a-a5a4-530776ae3d04",
            public_key="pk-lf-2cc1030e-f3d0-4381-b3cb-9805c0848436",
            host="https://cloud.langfuse.com"
        )
        
        self.enabled = all([
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY")
        ])
    
    def create_trace(
        self, 
        user_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new trace to track a complete interaction."""
        if not self.enabled:
            return str(uuid.uuid4())
        
        trace_id = str(uuid.uuid4())
        self.langfuse.trace(
            id=trace_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        return trace_id
    
    def log_llm_call(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        completion: str,
        latency: float,
        metadata: Optional[Dict[str, Any]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Log an LLM call to Langfuse."""
        if not self.enabled:
            return str(uuid.uuid4())
        
        generation_id = str(uuid.uuid4())
        self.langfuse.generation(
            id=generation_id,
            trace_id=trace_id,
            name="llm_call",
            model=model,
            prompt=prompt,
            completion=completion,
            start_time=time.time() - latency,
            end_time=time.time(),
            metadata=metadata or {},
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            temperature=temperature
        )
        return generation_id

    def log_retrieval(
        self,
        trace_id: str,
        query: str,
        documents: List[Dict[str, Any]],
        latency: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a document retrieval operation."""
        if not self.enabled:
            return str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        self.langfuse.span(
            id=span_id,
            trace_id=trace_id,
            name="document_retrieval",
            start_time=time.time() - latency,
            end_time=time.time(),
            metadata={
                "query": query,
                "num_documents": len(documents),
                "documents": documents,
                **(metadata or {})
            }
        )
        return span_id
    
    def log_user_feedback(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None
    ) -> None:
        """Log user feedback on an interaction."""
        if not self.enabled:
            return
        
        self.langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment
        )
    
    def flush(self) -> None:
        """Send all pending data to Langfuse."""
        if self.enabled:
            self.langfuse.flush()


class LangfuseMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatically tracing requests."""
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware."""
        super().__init__(app)
        self.monitor = LLMMonitor()
        self.monitored_endpoints = {"/answer", "/get_sources", "/feedback"}
    
    async def dispatch(self, request: Request, call_next):
        """Process the request and add monitoring."""
        if request.url.path in self.monitored_endpoints:
            trace_id = self.monitor.create_trace(
                metadata={"endpoint": request.url.path}
            )
            request.state.trace_id = trace_id
            
            # Measure response time
            start_time = time.time()
            response = await call_next(request)
            latency = time.time() - start_time
            
            # Log basic metrics for the request
            self.monitor.langfuse.span(
                name="api_call",
                trace_id=trace_id,
                start_time=start_time,
                end_time=start_time + latency,
                metadata={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "latency": latency
                }
            )
            self.monitor.flush()
            return response
        
        return await call_next(request)
