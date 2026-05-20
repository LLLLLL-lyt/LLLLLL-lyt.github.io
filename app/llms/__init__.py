"""
LLM 模块 - 大语言模型管理
参考 Bisheng 平台的模型配置设计
"""

# 基础类
from app.llms.base import BaseLLM

# 配置管理
from app.llms.config import (
    config_manager,
    LLMConfigManager,
    LLMServer,
    LLMModel,
    ModelType,
    ProviderType,
    ModelStatus,
    DEFAULT_SERVER_TEMPLATES,
)

# 工厂方法
from app.llms.factory import (
    LLMFactory,
    EmbeddingFactory,
    get_llm_config_schema,
    get_available_llms,
    get_available_embeddings,
    get_available_speech_models,
)

# LLM 提供商
from app.llms.providers import (
    # LLM 类
    OpenAILLM,
    QwenLLM,
    VolcengineLLM,
    SiliconLLM,
    DeepSeekLLM,
    ZhipuLLM,
    MoonshotLLM,
    MinimaxLLM,
    TencentLLM,
    QianfanLLM,
    SparkLLM,
    # 函数
    get_llm_provider,
    get_llm_from_config,
    LLM_REGISTRY,
)

__all__ = [
    # 基础
    "BaseLLM",
    
    # 配置管理
    "config_manager",
    "LLMConfigManager",
    "LLMServer",
    "LLMModel",
    "ModelType",
    "ProviderType",
    "ModelStatus",
    "DEFAULT_SERVER_TEMPLATES",
    
    # 工厂
    "LLMFactory",
    "EmbeddingFactory",
    "get_llm_config_schema",
    "get_available_llms",
    "get_available_embeddings",
    "get_available_speech_models",
    
    # LLM 类
    "OpenAILLM",
    "QwenLLM",
    "VolcengineLLM",
    "SiliconLLM",
    "DeepSeekLLM",
    "ZhipuLLM",
    "MoonshotLLM",
    "MinimaxLLM",
    "TencentLLM",
    "QianfanLLM",
    "SparkLLM",
    
    # 函数
    "get_llm_provider",
    "get_llm_from_config",
    "LLM_REGISTRY",
]
