from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.models.schemas import ChatMessage, LLMConfig

class BaseLLM(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
    
    @abstractmethod
    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def achat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Any:
        pass
    
    @abstractmethod
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        pass
    
    def format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]
