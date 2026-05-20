"""
Embedding 管理器 - 只支持在线 Embedding 模型
"""

from langchain_core.embeddings import Embeddings
from typing import Optional, Dict, Any
from app.llms import config_manager, ModelType
from loguru import logger
import requests


class VolcengineEmbeddings(Embeddings):
    """火山引擎 Embedding - 直接调用 API"""
    
    def __init__(self, model_id: str, api_key: str, api_base: str = "https://ark.cn-beijing.volces.com/api/v3"):
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        
        # 判断是否为多模态模型（包含 vision 关键字）
        if "vision" in model_id.lower() or "multimodal" in model_id.lower():
            self.endpoint = f"{self.api_base}/embeddings/multimodal"
            self.is_multimodal = True
        else:
            self.endpoint = f"{self.api_base}/embeddings"
            self.is_multimodal = False
    
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调用火山引擎 Embedding API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 根据模型类型构建不同的请求体
        if self.is_multimodal:
            # 多模态模型需要特殊格式
            payload = {
                "model": self.model_id,
                "input": [{"type": "text", "text": text} for text in texts]
            }
        else:
            # 普通文本模型
            payload = {
                "model": self.model_id,
                "input": texts
            }
        
        logger.info(f"调用火山引擎 Embedding API: {self.endpoint}")
        logger.info(f"模型类型: {'多模态' if self.is_multimodal else '文本'}, 请求文本数量: {len(texts)}")
        if texts:
            logger.info(f"第一个文本长度: {len(texts[0])}")
        
        response = requests.post(self.endpoint, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_msg = f"API 请求失败: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        result = response.json()
        logger.info(f"API 返回结果类型: {type(result)}")
        logger.info(f"API 返回结果 keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        
        # 检查返回格式
        data = result.get("data", [])
        logger.info(f"data 类型: {type(data)}")
        
        if not data:
            logger.error(f"API 返回数据为空，完整结果: {result}")
            raise Exception(f"API 返回数据为空: {result}")
        
        # 火山引擎多模态 API 可能返回字典格式，只包含单个 embedding
        if isinstance(data, dict):
            if "embedding" in data:
                # 单个嵌入向量 - 返回重复该向量的列表
                single_embedding = data["embedding"]
                logger.info(f"多模态 API 返回单个 embedding，将重复 {len(texts)} 次")
                return [single_embedding] * len(texts)
            else:
                logger.error(f"data 字典缺少 embedding 字段: {data}")
                raise Exception(f"API 返回格式错误，缺少 embedding 字段: {data}")
        
        # 如果是列表，按原逻辑处理
        if isinstance(data, list):
            embeddings = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
                    else:
                        logger.error(f"字典格式但无 embedding 字段: {item}")
                        raise Exception(f"API 返回格式错误，缺少 embedding 字段: {item}")
                elif isinstance(item, (list, tuple)):
                    embeddings.append(list(item))
                else:
                    logger.error(f"未知的返回格式 [{i}]: {type(item)}")
                    raise Exception(f"未知的返回格式: {item}")
            return embeddings
        
        logger.error(f"未知的 data 类型: {type(data)}")
        raise Exception(f"未知的 data 类型: {type(data)}")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"使用火山引擎 Embedding API 处理 {len(texts)} 个文本")
        return self._call_api(texts)
    
    def embed_query(self, text: str) -> list[float]:
        logger.info(f"使用火山引擎 Embedding API 处理查询")
        result = self._call_api([text])
        return result[0] if result else []


class OnlineEmbeddings(Embeddings):
    """在线 API Embedding 包装器"""
    
    def __init__(self, provider_id: str, model_id: str, api_key: str, api_base: Optional[str] = None):
        self.provider_id = provider_id
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
        
        # 根据提供商创建对应的 Embedding 客户端
        if provider_id == "volcengine":
            # 火山引擎使用自定义实现
            self._embeddings = VolcengineEmbeddings(
                model_id=model_id,
                api_key=api_key,
                api_base=api_base or "https://ark.cn-beijing.volces.com/api/v3"
            )
        else:
            # 其他提供商使用 langchain-openai
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                raise ImportError(
                    "使用在线 Embedding 需要安装 langchain-openai。"
                    "请运行: pip install langchain-openai"
                )
            
            if provider_id == "openai":
                self._embeddings = OpenAIEmbeddings(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base
                )
            elif provider_id == "zhipu":
                try:
                    from langchain_zhipu import ZhipuAIEmbeddings
                    self._embeddings = ZhipuAIEmbeddings(
                        model=model_id,
                        api_key=api_key
                    )
                except ImportError:
                    # 如果没有 langchain-zhipu，使用 OpenAI 兼容接口
                    self._embeddings = OpenAIEmbeddings(
                        model=model_id,
                        api_key=api_key,
                        base_url=api_base or "https://open.bigmodel.cn/api/paas/v4"
                    )
            elif provider_id == "qwen":
                try:
                    from langchain_community.embeddings import DashScopeEmbeddings
                    self._embeddings = DashScopeEmbeddings(
                        model=model_id,
                        dashscope_api_key=api_key
                    )
                except ImportError:
                    # 如果没有 DashScopeEmbeddings，使用 OpenAI 兼容接口
                    self._embeddings = OpenAIEmbeddings(
                        model=model_id,
                        api_key=api_key,
                        base_url=api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
            else:
                # 默认使用 OpenAI 兼容接口
                self._embeddings = OpenAIEmbeddings(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base
                )
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"使用 {self.provider_id} Embedding API 处理 {len(texts)} 个文本")
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        logger.info(f"使用 {self.provider_id} Embedding API 处理查询")
        return self._embeddings.embed_query(text)


class EmbeddingManager:
    """Embedding 管理器 - 只支持在线 Embedding"""
    
    def __init__(self):
        self._online_embeddings: Dict[str, Embeddings] = {}
    
    def get_embeddings_for_kb(self, embedding_model_id: str) -> Embeddings:
        """
        根据知识库配置的模型 ID 获取对应的 Embedding
        
        Args:
            embedding_model_id: 格式为 "provider_id/model_id"
        """
        if not embedding_model_id:
            raise ValueError("未配置 Embedding 模型，请在知识库配置中选择向量模型")
        
        # 检查缓存
        if embedding_model_id in self._online_embeddings:
            return self._online_embeddings[embedding_model_id]
        
        # 解析 provider_id 和 model_id
        if "/" not in embedding_model_id:
            raise ValueError(f"无效的 embedding_model_id 格式: {embedding_model_id}，应为 'provider_id/model_id'")
        
        provider_id, model_id = embedding_model_id.split("/", 1)
        
        # 从配置管理器获取服务器配置
        server = config_manager.get_server(provider_id)
        if not server or not server.api_key:
            raise ValueError(f"未找到 provider {provider_id} 的配置或 API Key，请先在设置中配置")
        
        # 查找模型配置
        model_config = None
        for m in server.models:
            if m.model_id == model_id and m.model_type == ModelType.EMBEDDING:
                model_config = m
                break
        
        if not model_config:
            raise ValueError(f"未找到模型 {model_id} 的 Embedding 配置，请检查模型配置")
        
        # 创建在线 Embedding
        logger.info(f"创建在线 Embedding: {provider_id}/{model_id}")
        online_embedding = OnlineEmbeddings(
            provider_id=provider_id,
            model_id=model_id,
            api_key=server.api_key,
            api_base=server.api_base
        )
        self._online_embeddings[embedding_model_id] = online_embedding
        return online_embedding


# 全局 Embedding 管理器实例
embedding_manager = EmbeddingManager()
