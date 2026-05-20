"""
模型配置管理模块 - 支持多模型、向量模型、语音模型配置
参考 Bisheng 平台的配置方式
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "model_config.json")


class ModelConfig(BaseModel):
    """单个模型配置"""
    model_config = {"protected_namespaces": ()}  # 禁用保护命名空间
    
    model_id: str  # 模型唯一标识
    model_name: str  # 显示名称
    model_type: str  # 类型: llm, embedding, speech, vision
    is_enabled: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    description: Optional[str] = None


class ProviderConfig(BaseModel):
    """提供商配置"""
    provider_id: str  # 提供商ID
    provider_name: str  # 显示名称
    api_key: str = ""
    api_base: Optional[str] = None
    is_enabled: bool = True
    models: List[ModelConfig] = []  # 该提供商下的所有模型
    
    # 链接信息
    api_key_url: str = ""
    doc_url: str = ""
    models_url: str = ""


class EmbeddingConfig(BaseModel):
    """向量模型配置"""
    model_config = {"protected_namespaces": ()}
    
    provider: str  # 提供商: openai, huggingface, etc.
    model_name: str  # 模型名称
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    is_enabled: bool = True
    dimension: Optional[int] = None  # 向量维度


class SpeechConfig(BaseModel):
    """语音模型配置"""
    model_config = {"protected_namespaces": ()}
    
    provider: str  # 提供商
    model_name: str  # 模型名称
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    is_enabled: bool = True
    type: str = "tts"  # tts(语音合成) 或 asr(语音识别)


class GlobalConfig(BaseModel):
    """全局配置"""
    llm_providers: Dict[str, ProviderConfig] = {}  # LLM提供商配置
    embedding_models: Dict[str, EmbeddingConfig] = {}  # 向量模型配置
    speech_models: Dict[str, SpeechConfig] = {}  # 语音模型配置
    default_llm_provider: Optional[str] = None
    default_llm_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_speech_model: Optional[str] = None


# 默认配置模板
DEFAULT_PROVIDERS = {
    "openai": {
        "provider_id": "openai",
        "provider_name": "OpenAI",
        "api_key_url": "https://platform.openai.com/api-keys",
        "doc_url": "https://platform.openai.com/docs",
        "models_url": "https://platform.openai.com/docs/models",
        "api_base": "https://api.openai.com/v1",
        "models": [
            {"model_id": "gpt-4", "model_name": "GPT-4", "model_type": "llm"},
            {"model_id": "gpt-4-turbo", "model_name": "GPT-4 Turbo", "model_type": "llm"},
            {"model_id": "gpt-3.5-turbo", "model_name": "GPT-3.5 Turbo", "model_type": "llm"},
            {"model_id": "text-embedding-3-small", "model_name": "Embedding Small", "model_type": "embedding"},
            {"model_id": "text-embedding-3-large", "model_name": "Embedding Large", "model_type": "embedding"},
            {"model_id": "tts-1", "model_name": "TTS-1", "model_type": "speech"},
            {"model_id": "whisper-1", "model_name": "Whisper", "model_type": "speech"},
        ]
    },
    "qwen": {
        "provider_id": "qwen",
        "provider_name": "阿里百炼 (Qwen)",
        "api_key_url": "https://bailian.console.aliyun.com/#/api-key",
        "doc_url": "https://help.aliyun.com/zh/dashscope/",
        "models_url": "https://dashscope.console.aliyun.com/model",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            {"model_id": "qwen-plus", "model_name": "通义千问 Plus", "model_type": "llm"},
            {"model_id": "qwen-turbo", "model_name": "通义千问 Turbo", "model_type": "llm"},
            {"model_id": "qwen-max", "model_name": "通义千问 Max", "model_type": "llm"},
            {"model_id": "text-embedding-v2", "model_name": "Embedding V2", "model_type": "embedding"},
            {"model_id": "text-embedding-v3", "model_name": "Embedding V3", "model_type": "embedding"},
        ]
    },
    "volcengine": {
        "provider_id": "volcengine",
        "provider_name": "火山引擎 (Volcengine)",
        "api_key_url": "https://console.volcengine.com/ark",
        "doc_url": "https://www.volcengine.com/docs/82379",
        "models_url": "https://console.volcengine.com/ark/region:ark+cn-beijing/model",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "models": [
            {"model_id": "doubao-pro-32k", "model_name": "豆包 Pro 32K", "model_type": "llm"},
            {"model_id": "doubao-pro-128k", "model_name": "豆包 Pro 128K", "model_type": "llm"},
            {"model_id": "doubao-lite-32k", "model_name": "豆包 Lite 32K", "model_type": "llm"},
            {"model_id": "doubao-embedding", "model_name": "豆包 Embedding", "model_type": "embedding"},
            {"model_id": "doubao-embedding-large", "model_name": "豆包 Embedding Large", "model_type": "embedding"},
            {"model_id": "doubao-tts", "model_name": "豆包 TTS", "model_type": "speech"},
            {"model_id": "doubao-asr", "model_name": "豆包 ASR", "model_type": "speech"},
        ]
    },
    "tencent": {
        "provider_id": "tencent",
        "provider_name": "腾讯混元 (Tencent)",
        "api_key_url": "https://console.cloud.tencent.com/cam/capi",
        "doc_url": "https://cloud.tencent.com/document/product/1729",
        "models_url": "https://cloud.tencent.com/document/product/1729/97732",
        "api_base": "https://hunyuan.tencentcloudapi.com",
        "models": [
            {"model_id": "hunyuan-lite", "model_name": "混元 Lite", "model_type": "llm"},
            {"model_id": "hunyuan-standard", "model_name": "混元 Standard", "model_type": "llm"},
            {"model_id": "hunyuan-pro", "model_name": "混元 Pro", "model_type": "llm"},
            {"model_id": "hunyuan-turbo", "model_name": "混元 Turbo", "model_type": "llm"},
        ]
    },
    "qianfan": {
        "provider_id": "qianfan",
        "provider_name": "百度千帆 (Qianfan)",
        "api_key_url": "https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application",
        "doc_url": "https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html",
        "models_url": "https://console.bce.baidu.com/qianfan/ais/console/onlineService",
        "api_base": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
        "models": [
            {"model_id": "ERNIE-4.0-8K", "model_name": "文心一言 4.0", "model_type": "llm"},
            {"model_id": "ERNIE-3.5-8K", "model_name": "文心一言 3.5", "model_type": "llm"},
            {"model_id": "ERNIE-Speed-8K", "model_name": "文心 Speed", "model_type": "llm"},
            {"model_id": "Embedding-V1", "model_name": "百度 Embedding", "model_type": "embedding"},
        ]
    },
    "spark": {
        "provider_id": "spark",
        "provider_name": "讯飞星火 (Spark)",
        "api_key_url": "https://console.xfyun.cn/services/bm3",
        "doc_url": "https://www.xfyun.cn/doc/spark/Web.html",
        "models_url": "https://console.xfyun.cn/services/bm3",
        "api_base": "wss://spark-api.xf-yun.com",
        "models": [
            {"model_id": "spark-v4.0", "model_name": "星火 V4.0", "model_type": "llm"},
            {"model_id": "spark-v3.5", "model_name": "星火 V3.5", "model_type": "llm"},
            {"model_id": "spark-v3.0", "model_name": "星火 V3.0", "model_type": "llm"},
        ]
    },
    "minimax": {
        "provider_id": "minimax",
        "provider_name": "Minimax",
        "api_key_url": "https://platform.minimaxi.com/user-center/basic-information/interface-key",
        "doc_url": "https://platform.minimaxi.com/document/guides",
        "models_url": "https://platform.minimaxi.com/document/models",
        "api_base": "https://api.minimax.chat/v1",
        "models": [
            {"model_id": "abab6.5s", "model_name": "abab 6.5s", "model_type": "llm"},
            {"model_id": "abab6.5", "model_name": "abab 6.5", "model_type": "llm"},
            {"model_id": "abab5.5", "model_name": "abab 5.5", "model_type": "llm"},
        ]
    },
    "moonshot": {
        "provider_id": "moonshot",
        "provider_name": "月之暗面 (Moonshot)",
        "api_key_url": "https://platform.moonshot.cn/console/api-keys",
        "doc_url": "https://platform.moonshot.cn/docs/intro",
        "models_url": "https://platform.moonshot.cn/docs/intro#%E6%A8%A1%E5%9E%8B%E8%AF%B4%E6%98%8E",
        "api_base": "https://api.moonshot.cn/v1",
        "models": [
            {"model_id": "moonshot-v1-8k", "model_name": "Moonshot 8K", "model_type": "llm"},
            {"model_id": "moonshot-v1-32k", "model_name": "Moonshot 32K", "model_type": "llm"},
            {"model_id": "moonshot-v1-128k", "model_name": "Moonshot 128K", "model_type": "llm"},
        ]
    },
    "zhipu": {
        "provider_id": "zhipu",
        "provider_name": "智谱AI (Zhipu)",
        "api_key_url": "https://open.bigmodel.cn/usercenter/apikeys",
        "doc_url": "https://open.bigmodel.cn/dev/howuse/glm-4",
        "models_url": "https://open.bigmodel.cn/modelcenter/square",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "models": [
            {"model_id": "glm-4", "model_name": "GLM-4", "model_type": "llm"},
            {"model_id": "glm-4-plus", "model_name": "GLM-4 Plus", "model_type": "llm"},
            {"model_id": "glm-4-flash", "model_name": "GLM-4 Flash", "model_type": "llm"},
            {"model_id": "embedding-2", "model_name": "智谱 Embedding", "model_type": "embedding"},
        ]
    },
    "silicon": {
        "provider_id": "silicon",
        "provider_name": "SiliconFlow",
        "api_key_url": "https://cloud.siliconflow.cn/account/ak",
        "doc_url": "https://docs.siliconflow.cn/",
        "models_url": "https://cloud.siliconflow.cn/models",
        "api_base": "https://api.siliconflow.cn/v1",
        "models": [
            {"model_id": "Qwen/Qwen2.5-72B-Instruct", "model_name": "Qwen2.5 72B", "model_type": "llm"},
            {"model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct", "model_name": "Llama 3.1 70B", "model_type": "llm"},
            {"model_id": "BAAI/bge-large-zh-v1.5", "model_name": "BGE Large Zh", "model_type": "embedding"},
            {"model_id": "BAAI/bge-m3", "model_name": "BGE M3", "model_type": "embedding"},
        ]
    },
    "deepseek": {
        "provider_id": "deepseek",
        "provider_name": "DeepSeek",
        "api_key_url": "https://platform.deepseek.com/api_keys",
        "doc_url": "https://platform.deepseek.com/api-docs/",
        "models_url": "https://platform.deepseek.com/api-docs/pricing",
        "api_base": "https://api.deepseek.com/v1",
        "models": [
            {"model_id": "deepseek-chat", "model_name": "DeepSeek Chat", "model_type": "llm"},
            {"model_id": "deepseek-coder", "model_name": "DeepSeek Coder", "model_type": "llm"},
        ]
    },
}


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = GlobalConfig()
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config = GlobalConfig(**data)
            except Exception as e:
                print(f"加载配置失败: {e}，使用默认配置")
                self._init_default_config()
        else:
            self._init_default_config()
    
    def _init_default_config(self):
        """初始化默认配置"""
        # 初始化LLM提供商
        for provider_id, provider_data in DEFAULT_PROVIDERS.items():
            models = [ModelConfig(**m) for m in provider_data.get("models", [])]
            self.config.llm_providers[provider_id] = ProviderConfig(
                provider_id=provider_id,
                provider_name=provider_data["provider_name"],
                api_key_url=provider_data["api_key_url"],
                doc_url=provider_data["doc_url"],
                models_url=provider_data["models_url"],
                api_base=provider_data.get("api_base"),
                models=models
            )
            
            # 从提供商配置中提取向量模型
            for model in models:
                if model.model_type == "embedding" and model.model_id not in self.config.embedding_models:
                    # 根据提供商设置默认维度
                    dimension = 1536  # 默认维度
                    if provider_id == "openai":
                        dimension = 1536
                    elif provider_id == "qwen":
                        dimension = 1536
                    elif provider_id == "volcengine":
                        dimension = 2048
                    elif provider_id == "qianfan":
                        dimension = 384
                    elif provider_id == "zhipu":
                        dimension = 2048
                    
                    self.config.embedding_models[model.model_id] = EmbeddingConfig(
                        provider=provider_id,
                        model_name=model.model_name,
                        is_enabled=model.is_enabled,
                        dimension=dimension
                    )
                elif model.model_type == "speech" and model.model_id not in self.config.speech_models:
                    self.config.speech_models[model.model_id] = SpeechConfig(
                        provider=provider_id,
                        model_name=model.model_name,
                        is_enabled=model.is_enabled
                    )
        
        # 初始化向量模型（从全局配置）
        for model_id, model_data in DEFAULT_EMBEDDING_MODELS.items():
            if model_id not in self.config.embedding_models:
                self.config.embedding_models[model_id] = EmbeddingConfig(**model_data)
    
    def save_config(self):
        """保存配置"""
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config.model_dump(), f, ensure_ascii=False, indent=2)
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """获取提供商配置"""
        return self.config.llm_providers.get(provider_id)
    
    def get_enabled_providers(self) -> List[ProviderConfig]:
        """获取已启用的提供商"""
        return [p for p in self.config.llm_providers.values() if p.is_enabled]
    
    def get_enabled_llm_models(self, provider_id: str) -> List[ModelConfig]:
        """获取已启用的LLM模型"""
        provider = self.config.llm_providers.get(provider_id)
        if not provider:
            return []
        return [m for m in provider.models if m.model_type == "llm" and m.is_enabled]
    
    def get_enabled_embedding_models(self, provider_id: str) -> List[ModelConfig]:
        """获取已启用的向量模型"""
        provider = self.config.llm_providers.get(provider_id)
        if not provider:
            return []
        return [m for m in provider.models if m.model_type == "embedding" and m.is_enabled]
    
    def update_provider_api_key(self, provider_id: str, api_key: str, api_base: Optional[str] = None):
        """更新提供商API Key"""
        if provider_id in self.config.llm_providers:
            self.config.llm_providers[provider_id].api_key = api_key
            if api_base:
                self.config.llm_providers[provider_id].api_base = api_base
            self.save_config()
    
    def update_model_enabled(self, provider_id: str, model_id: str, is_enabled: bool):
        """更新模型启用状态"""
        provider = self.config.llm_providers.get(provider_id)
        if provider:
            for model in provider.models:
                if model.model_id == model_id:
                    model.is_enabled = is_enabled
                    self.save_config()
                    break
    
    def add_custom_model(self, provider_id: str, model_id: str, model_name: str, model_type: str = "llm"):
        """添加自定义模型"""
        provider = self.config.llm_providers.get(provider_id)
        if provider:
            # 检查是否已存在
            for m in provider.models:
                if m.model_id == model_id:
                    return
            # 添加新模型
            provider.models.append(ModelConfig(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type
            ))
            self.save_config()
    
    def remove_custom_model(self, provider_id: str, model_id: str):
        """删除自定义模型"""
        provider = self.config.llm_providers.get(provider_id)
        if provider:
            provider.models = [m for m in provider.models if m.model_id != model_id]
            self.save_config()
    
    def get_embedding_model(self, model_id: str) -> Optional[EmbeddingConfig]:
        """获取向量模型配置"""
        return self.config.embedding_models.get(model_id)
    
    def get_enabled_embedding_models(self) -> Dict[str, EmbeddingConfig]:
        """获取所有启用的向量模型"""
        return {k: v for k, v in self.config.embedding_models.items() if v.is_enabled}
    
    def update_embedding_model(self, model_id: str, config: EmbeddingConfig):
        """更新向量模型配置"""
        self.config.embedding_models[model_id] = config
        self.save_config()
    
    def get_speech_model(self, model_id: str) -> Optional[SpeechConfig]:
        """获取语音模型配置"""
        return self.config.speech_models.get(model_id)
    
    def update_speech_model(self, model_id: str, config: SpeechConfig):
        """更新语音模型配置"""
        self.config.speech_models[model_id] = config
        self.save_config()


# 全局配置管理器实例
config_manager = ConfigManager()
