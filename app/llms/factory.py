"""
LLM 工厂模块 - 参考 Bisheng 的 _llm_node_type 设计
根据 provider_type 自动创建对应的 LLM 客户端
"""

from typing import Dict, Any, Optional, Type
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatZhipuAI

from app.llms.config import (
    LLMConfigManager, LLMServer, LLMModel, 
    ProviderType, ModelType, config_manager
)
from app.models.schemas import LLMConfig as LLMConfigSchema
from loguru import logger


class LLMFactory:
    """
    LLM 工厂类 - 根据提供商类型创建对应的客户端
    参考 Bisheng 的 _llm_node_type 设计
    """
    
    @classmethod
    def create_llm_client(cls, provider_id: str, model_id: str, **kwargs) -> Any:
        """
        创建 LLM 客户端
        
        Args:
            provider_id: 提供商ID
            model_id: 模型ID
            **kwargs: 额外参数（temperature, max_tokens等）
        
        Returns:
            对应的 LLM 客户端实例
        """
        server = config_manager.get_server(provider_id)
        if not server:
            raise ValueError(f"Provider {provider_id} not found")
        
        model = server.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found in provider {provider_id}")
        
        # 根据提供商类型创建对应的客户端
        creator = cls._get_creator(server.provider_type)
        return creator(server, model, **kwargs)
    
    @classmethod
    def _get_creator(cls, provider_type: ProviderType) -> callable:
        """获取对应提供商类型的创建函数"""
        creators = {
            # OpenAI 兼容格式
            ProviderType.OPENAI: cls._create_openai_compatible,
            ProviderType.QWEN: cls._create_openai_compatible,
            ProviderType.VOLCENGINE: cls._create_openai_compatible,
            ProviderType.DEEPSEEK: cls._create_openai_compatible,
            ProviderType.MOONSHOT: cls._create_openai_compatible,
            ProviderType.SILICON: cls._create_openai_compatible,
            ProviderType.MINIMAX: cls._create_openai_compatible,
            
            # 特殊格式
            ProviderType.AZURE_OPENAI: cls._create_azure_openai,
            ProviderType.ZHIPU: cls._create_zhipu,
            ProviderType.TENCENT: cls._create_tencent,
            ProviderType.QIANFAN: cls._create_qianfan,
            ProviderType.SPARK: cls._create_spark,
        }
        
        creator = creators.get(provider_type)
        if not creator:
            # 默认使用 OpenAI 兼容格式
            logger.warning(f"Unknown provider type {provider_type}, using openai compatible")
            return cls._create_openai_compatible
        
        return creator
    
    @classmethod
    def _create_openai_compatible(cls, server: LLMServer, model: LLMModel, **kwargs) -> ChatOpenAI:
        """创建 OpenAI 兼容格式的客户端"""
        params = {
            "model": model.model_id,
            "api_key": server.api_key or "empty",
            "base_url": server.api_base,
            "temperature": kwargs.get("temperature", model.temperature or 0.7),
        }
        
        if model.max_tokens:
            params["max_tokens"] = model.max_tokens
        
        if model.top_p:
            params["top_p"] = model.top_p
        
        # 合并用户自定义参数
        params.update(model.config)
        params.update(kwargs)
        
        return ChatOpenAI(**params)
    
    @classmethod
    def _create_azure_openai(cls, server: LLMServer, model: LLMModel, **kwargs) -> AzureChatOpenAI:
        """创建 Azure OpenAI 客户端"""
        params = {
            "azure_deployment": model.model_id,
            "api_key": server.api_key,
            "azure_endpoint": server.api_base,
            "api_version": server.api_version or "2024-02-01",
            "temperature": kwargs.get("temperature", model.temperature or 0.7),
        }
        
        if model.max_tokens:
            params["max_tokens"] = model.max_tokens
        
        params.update(model.config)
        params.update(kwargs)
        
        return AzureChatOpenAI(**params)
    
    @classmethod
    def _create_zhipu(cls, server: LLMServer, model: LLMModel, **kwargs) -> ChatOpenAI:
        """创建智谱AI客户端 - 使用OpenAI兼容模式"""
        params = {
            "model": model.model_id,
            "api_key": server.api_key or "empty",
            "base_url": server.api_base or "https://open.bigmodel.cn/api/paas/v4",
            "temperature": kwargs.get("temperature", model.temperature or 0.7),
        }
        
        if model.max_tokens:
            params["max_tokens"] = model.max_tokens
        
        params.update(model.config)
        params.update(kwargs)
        
        return ChatOpenAI(**params)
    
    @classmethod
    def _create_tencent(cls, server: LLMServer, model: LLMModel, **kwargs) -> ChatOpenAI:
        """创建腾讯混元客户端 - 使用OpenAI兼容模式"""
        params = {
            "model": model.model_id,
            "api_key": server.api_key or "empty",
            "base_url": server.api_base or "https://hunyuan.tencentcloudapi.com",
            "temperature": kwargs.get("temperature", model.temperature or 0.7),
        }
        
        if model.max_tokens:
            params["max_tokens"] = model.max_tokens
        
        params.update(model.config)
        params.update(kwargs)
        
        return ChatOpenAI(**params)
    
    @classmethod
    def _create_qianfan(cls, server: LLMServer, model: LLMModel, **kwargs) -> ChatOpenAI:
        """创建百度千帆客户端 - 使用OpenAI兼容模式"""
        params = {
            "model": model.model_id,
            "api_key": server.api_key or "empty",
            "base_url": server.api_base or "https://qianfan.baidubce.com/v2",
            "temperature": kwargs.get("temperature", model.temperature or 0.7),
        }
        
        if model.max_tokens:
            params["max_tokens"] = model.max_tokens
        
        params.update(model.config)
        params.update(kwargs)
        
        return ChatOpenAI(**params)
    
    @classmethod
    def _create_spark(cls, server: LLMServer, model: LLMModel, **kwargs):
        """创建讯飞星火客户端"""
        try:
            from langchain_community.chat_models import ChatSparkLLM
            
            # 解析 API Key 和 Secret
            api_key = server.api_key or ""
            api_secret = ""
            if ":" in api_key:
                api_key, api_secret = api_key.split(":", 1)
            
            params = {
                "spark_app_id": api_key,
                "spark_api_key": api_secret,
                "spark_api_secret": api_secret,
                "model": model.model_id,
                "temperature": kwargs.get("temperature", model.temperature or 0.7),
            }
            
            if model.max_tokens:
                params["max_tokens"] = model.max_tokens
            
            params.update(model.config)
            params.update(kwargs)
            
            return ChatSparkLLM(**params)
        except ImportError:
            logger.error("ChatSparkLLM not installed, please install langchain-community")
            # 降级为 OpenAI 兼容模式
            return cls._create_openai_compatible(server, model, **kwargs)


class EmbeddingFactory:
    """Embedding 模型工厂"""
    
    @classmethod
    def create_embedding_client(cls, provider_id: str, model_id: str):
        """创建 Embedding 客户端"""
        server = config_manager.get_server(provider_id)
        if not server:
            raise ValueError(f"Provider {provider_id} not found")
        
        model = server.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found in provider {provider_id}")
        
        # 默认使用 OpenAI 兼容格式
        from langchain_openai import OpenAIEmbeddings
        
        params = {
            "model": model.model_id,
            "api_key": server.api_key or "empty",
            "base_url": server.api_base,
        }
        
        params.update(model.config)
        
        return OpenAIEmbeddings(**params)


def get_llm_config_schema(provider_id: str, model_id: str) -> Optional[LLMConfigSchema]:
    """
    获取 LLM 配置 Schema（用于兼容旧代码）
    """
    server = config_manager.get_server(provider_id)
    if not server:
        return None
    
    model = server.get_model(model_id)
    if not model:
        return None
    
    return LLMConfigSchema(
        provider=provider_id,
        model=model_id,
        api_key=server.api_key or "",
        api_base=server.api_base,
        temperature=model.temperature or 0.7,
        max_tokens=model.max_tokens or 4096,
        top_p=model.top_p or 0.9,
    )


def get_available_llms() -> list[dict]:
    """获取可用的 LLM 列表（用于前端展示）"""
    result = []
    for provider_id, server in config_manager.servers.items():
        if not server.is_enabled or not server.api_key:
            continue
        for model in server.models:
            if model.is_enabled and model.model_type == ModelType.LLM:
                result.append({
                    "provider_id": provider_id,
                    "provider_name": server.provider_name,
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "full_name": f"{server.provider_name} - {model.model_name}",
                })
    return result


def get_available_embeddings() -> list[dict]:
    """获取可用的 Embedding 列表（用于前端展示）"""
    result = []
    for provider_id, server in config_manager.servers.items():
        if not server.is_enabled or not server.api_key:
            continue
        for model in server.models:
            if model.is_enabled and model.model_type == ModelType.EMBEDDING:
                result.append({
                    "provider_id": provider_id,
                    "provider_name": server.provider_name,
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "full_name": f"{server.provider_name} - {model.model_name}",
                })
    return result


def get_available_speech_models() -> list[dict]:
    """获取可用的语音模型列表（用于前端展示）"""
    result = []
    for provider_id, server in config_manager.servers.items():
        if not server.is_enabled or not server.api_key:
            continue
        for model in server.models:
            if model.is_enabled and model.model_type in (ModelType.SPEECH_TTS, ModelType.SPEECH_ASR):
                result.append({
                    "provider_id": provider_id,
                    "provider_name": server.provider_name,
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "model_type": model.model_type.value,
                    "full_name": f"{server.provider_name} - {model.model_name}",
                })
    return result
