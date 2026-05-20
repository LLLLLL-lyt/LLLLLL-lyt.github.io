"""
LLM 提供商实现模块 - 适配新的配置系统
保留原有的 LLM 类实现，但使用新的配置管理器
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI, OpenAI
from app.llms.base import BaseLLM
from app.models.schemas import ChatMessage, LLMConfig
from loguru import logger

# 导入新的配置系统
from app.llms.config import config_manager, ModelType, ProviderType
from app.llms.factory import LLMFactory


class OpenAILLM(BaseLLM):
    """OpenAI 兼容格式的 LLM"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.api_base or "https://api.openai.com/v1"
        )
        self.aclient = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=self.api_base or "https://api.openai.com/v1"
        )
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )
        return {
            "answer": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )
        return {
            "answer": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        stream = await self.aclient.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# 各种提供商的 LLM 实现（继承 OpenAILLM，只需设置默认 api_base）
class QwenLLM(OpenAILLM):
    """阿里通义千问"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        super().__init__(config)


class VolcengineLLM(OpenAILLM):
    """火山引擎"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://ark.cn-beijing.volces.com/api/v3"
        super().__init__(config)


class SiliconLLM(OpenAILLM):
    """SiliconFlow"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://api.siliconflow.cn/v1"
        super().__init__(config)


class DeepSeekLLM(OpenAILLM):
    """DeepSeek"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://api.deepseek.com"
        super().__init__(config)


class ZhipuLLM(OpenAILLM):
    """智谱AI"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://open.bigmodel.cn/api/paas/v4"
        super().__init__(config)


class MoonshotLLM(OpenAILLM):
    """月之暗面"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://api.moonshot.cn/v1"
        super().__init__(config)


class MinimaxLLM(OpenAILLM):
    """MiniMax"""
    def __init__(self, config: LLMConfig):
        if not config.api_base:
            config.api_base = "https://api.minimax.chat/v1"
        super().__init__(config)


class TencentLLM(BaseLLM):
    """腾讯混元 - 使用官方SDK"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.hunyuan.v20230901 import hunyuan_client
            
            # 解析 API Key (格式: SecretId:SecretKey)
            if ":" in config.api_key:
                secret_id, secret_key = config.api_key.split(":", 1)
            else:
                secret_id = config.api_key
                secret_key = ""
            
            self.cred = credential.Credential(secret_id, secret_key)
            self.httpProfile = HttpProfile()
            self.httpProfile.endpoint = "hunyuan.tencentcloudapi.com"
            self.clientProfile = ClientProfile()
            self.clientProfile.httpProfile = self.httpProfile
            self.client = hunyuan_client.HunyuanClient(
                self.cred, "ap-guangzhou", self.clientProfile
            )
        except Exception as e:
            logger.error(f"Tencent LLM init failed: {e}")
            raise
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        try:
            from tencentcloud.hunyuan.v20230901 import models
            
            req = models.ChatCompletionsRequest()
            params = {
                "Model": self.model,
                "Messages": self.format_messages(messages),
            }
            
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["Temperature"] = temperature
            
            top_p = kwargs.get("top_p", self.top_p)
            if top_p is not None:
                params["TopP"] = top_p
            
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens:
                params["MaxTokens"] = max_tokens
            
            req.from_json_string(json.dumps(params))
            resp = self.client.ChatCompletions(req)
            
            return {
                "answer": resp.Choices[0].Message.Content,
                "usage": {
                    "prompt_tokens": resp.Usage.PromptTokens,
                    "completion_tokens": resp.Usage.CompletionTokens,
                    "total_tokens": resp.Usage.TotalTokens,
                }
            }
        except Exception as e:
            logger.error(f"Tencent chat failed: {e}")
            raise
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        return self.chat(messages, **kwargs)
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        try:
            from tencentcloud.hunyuan.v20230901 import models
            
            req = models.ChatCompletionsRequest()
            params = {
                "Model": self.model,
                "Messages": self.format_messages(messages),
                "Stream": True,
            }
            
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["Temperature"] = temperature
            
            top_p = kwargs.get("top_p", self.top_p)
            if top_p is not None:
                params["TopP"] = top_p
            
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens:
                params["MaxTokens"] = max_tokens
            
            req.from_json_string(json.dumps(params))
            
            for event in self.client.ChatCompletions(req):
                if event.Choices and len(event.Choices) > 0:
                    content = event.Choices[0].Delta.Content
                    if content:
                        yield content
        except Exception as e:
            logger.error(f"Tencent stream chat failed: {e}")
            raise
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        for chunk in self.stream_chat(messages, **kwargs):
            yield chunk


class QianfanLLM(BaseLLM):
    """百度千帆"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import qianfan
            
            # 解析 API Key (格式: AK:SK)
            if ":" in config.api_key:
                ak, sk = config.api_key.split(":", 1)
            else:
                ak = config.api_key
                sk = ""
            
            self.client = qianfan.ChatCompletion(ak=ak, sk=sk)
        except Exception as e:
            logger.error(f"Qianfan LLM init failed: {e}")
            raise
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        try:
            resp = self.client.do(
                model=self.model,
                messages=self.format_messages(messages),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
            )
            return {
                "answer": resp["result"],
                "usage": resp.get("usage", {})
            }
        except Exception as e:
            logger.error(f"Qianfan chat failed: {e}")
            raise
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        return self.chat(messages, **kwargs)
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        try:
            resp = self.client.do(
                model=self.model,
                messages=self.format_messages(messages),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                stream=True,
            )
            for r in resp:
                yield r["result"]
        except Exception as e:
            logger.error(f"Qianfan stream chat failed: {e}")
            raise
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        for chunk in self.stream_chat(messages, **kwargs):
            yield chunk


class SparkLLM(BaseLLM):
    """讯飞星火"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            # 解析 API Key (格式: APP_ID:API_SECRET:API_KEY)
            parts = config.api_key.split(":")
            if len(parts) >= 3:
                self.app_id, self.api_secret, self.api_key = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                self.app_id, self.api_secret = parts[0], parts[1]
                self.api_key = ""
            else:
                self.app_id = parts[0]
                self.api_secret = ""
                self.api_key = ""
            
            # 使用 OpenAI 兼容模式
            self.client = OpenAI(
                api_key=f"{self.api_secret}:{self.api_key}",
                base_url=config.api_base or "https://spark-api-open.xf-yun.com/v1"
            )
        except Exception as e:
            logger.error(f"Spark LLM init failed: {e}")
            raise
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
        )
        return {
            "answer": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        # 讯飞暂不支持异步，使用同步方式
        return self.chat(messages, **kwargs)
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        for chunk in self.stream_chat(messages, **kwargs):
            yield chunk


# LLM 注册表 - 提供商ID -> LLM类
LLM_REGISTRY = {
    "openai": OpenAILLM,
    "qwen": QwenLLM,
    "volcengine": VolcengineLLM,
    "tencent": TencentLLM,
    "qianfan": QianfanLLM,
    "spark": SparkLLM,
    "minimax": MinimaxLLM,
    "moonshot": MoonshotLLM,
    "zhipu": ZhipuLLM,
    "silicon": SiliconLLM,
    "deepseek": DeepSeekLLM,
}


def get_llm_provider(config: LLMConfig) -> BaseLLM:
    """
    根据配置获取 LLM 提供商实例
    
    Args:
        config: LLMConfig 配置对象
    
    Returns:
        BaseLLM 实例
    """
    provider_class = LLM_REGISTRY.get(config.provider)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {config.provider}")
    return provider_class(config)


def get_llm_from_config(provider_id: str, model_id: str, **kwargs) -> BaseLLM:
    """
    从新的配置系统创建 LLM 实例
    
    Args:
        provider_id: 提供商ID
        model_id: 模型ID
        **kwargs: 覆盖参数（temperature, max_tokens等）
    
    Returns:
        BaseLLM 实例
    """
    from app.llms.factory import get_llm_config_schema
    
    config = get_llm_config_schema(provider_id, model_id)
    if not config:
        raise ValueError(f"Model {model_id} not found in provider {provider_id}")
    
    # 应用覆盖参数
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    return get_llm_provider(config)


# 导出新的配置系统组件
__all__ = [
    # LLM 类
    "OpenAILLM", "QwenLLM", "VolcengineLLM", "SiliconLLM", 
    "DeepSeekLLM", "ZhipuLLM", "MoonshotLLM",
    "MinimaxLLM", "TencentLLM", "QianfanLLM", "SparkLLM",
    # 函数
    "get_llm_provider", "get_llm_from_config",
    # 配置系统
    "config_manager", "ModelType", "ProviderType",
    "LLMFactory", "EmbeddingFactory",
]

# 延迟导入以避免循环依赖
import json
