from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    query: str
    knowledge_base_id: Optional[str] = None
    conversation_id: Optional[str] = None
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    use_rag: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None

class KnowledgeBaseCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class KnowledgeBaseInfo(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int
    size: int

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    status: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_id: str
    knowledge_base_id: str
    created_at: datetime
    chunk_count: int

class ProviderInfo(BaseModel):
    id: str
    name: str
    api_key_url: str
    doc_url: str
    models_url: str
    models: List[str]

class LLMConfig(BaseModel):
    provider: str
    api_key: str
    api_base: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
