"""
Reranker 模块 - 支持在线 API 和禁用两种模式
"""

from typing import List, Optional
from langchain_core.documents import Document
from app.config import settings
from loguru import logger
import time

# Reranker 运行模式
class RerankerMode:
    DISABLED = "disabled"   # 禁用 Reranker
    ONLINE = "online"       # 使用在线 API（如 Cohere Rerank 等）

# 在线 API 支持检查
try:
    import httpx
    HAS_ONLINE_SUPPORT = True
except ImportError:
    HAS_ONLINE_SUPPORT = False


class Reranker:
    """
    重排序器 - 支持两种运行模式：

    - **disabled**: 跳过重排序，直接返回原始检索结果
    - **online**: 调用在线 Reranker API（如 Cohere、Jina 等）

    用法示例::

        # 切换模式
        reranker.set_mode("disabled")  # 禁用
        reranker.set_mode(RerankerMode.ONLINE)  # 在线 API

        # 正常调用（模式切换后无需改代码）
        results = reranker.rerank(query, documents)
    """
    
    # 在线 Reranker API 配置
    ONLINE_API_CONFIG = {
        "cohere": {
            "name": "Cohere Rerank",
            "url": "https://api.cohere.ai/v1/rerank",
            "default_model": "rerank-multilingual-v3.0",
        },
        "jina": {
            "name": "Jina Reranker",
            "url": "https://api.jina.ai/v1/rerank",
            "default_model": "jina-reranker-v2-base-multilingual",
        },
    }
    
    def __init__(self):
        self._mode: str = settings.RERANKER_MODE  # 默认从配置读取
        self._online_provider: str = settings.RERANKER_ONLINE_PROVIDER
        self._api_key: Optional[str] = None

        # 根据初始配置判断可用性
        self._check_capability()

    def _check_capability(self):
        """检查当前模式的可用性"""
        pass
    
    @property
    def mode(self) -> str:
        """当前运行的 Reranker 模式"""
        return self._mode
    
    @property
    def is_enabled(self) -> bool:
        """Reranker 是否启用"""
        return self._mode != RerankerMode.DISABLED
    
    def set_mode(self, mode: str,
                 online_provider: Optional[str] = None, api_key: Optional[str] = None) -> str:
        """
        动态切换 Reranker 模式

        Args:
            mode: 运行模式 ("disabled", "online")
            online_provider: 在线服务提供商（"cohere", "jina"，仅在 online 模式下有效）
            api_key: 在线 API Key（仅在 online 模式下有效）

        Returns:
            操作结果信息字符串
        """
        valid_modes = [RerankerMode.DISABLED, RerankerMode.ONLINE]
        if mode not in valid_modes:
            return f"❌ 无效的模式: {mode}，可选值: {valid_modes}"

        self._mode = mode

        if mode == RerankerMode.DISABLED:
            logger.info("Reranker 已切换为: 禁用模式")
            return "✅ Reranker 已禁用"

        elif mode == RerankerMode.ONLINE:
            if not HAS_ONLINE_SUPPORT:
                self._mode = RerankerMode.DISABLED
                return "❌ 在线模式不可用：请安装 httpx（pip install httpx）"

            if online_provider:
                if online_provider not in self.ONLINE_API_CONFIG:
                    available = list(self.ONLINE_API_CONFIG.keys())
                    return f"❌ 不支持的在线提供商: {online_provider}，可选: {available}"
                self._online_provider = online_provider

            if api_key:
                self._api_key = api_key

            provider_info = self.ONLINE_API_CONFIG.get(self._online_provider, {})
            logger.info(f"Reranker 已切换为: 在线模式 (提供商: {provider_info.get('name', self._online_provider)})")
            return f"✅ Reranker 已切换为在线模式，使用: {provider_info.get('name', self._online_provider)}"

        return ""

    def _rerank_online(
        self,
        query: str,
        documents: List[Document],
        top_n: int
    ) -> List[Document]:
        """调用在线 API 进行重排序"""
        provider_config = self.ONLINE_API_CONFIG.get(self._online_provider, {})
        url = provider_config.get("url", "")
        model = provider_config.get("default_model", "rerank-v3.0")
        api_key = self._api_key
        
        if not api_key:
            logger.warning(f"未设置 {self._online_provider} API Key，跳过在线重排序")
            return documents[:top_n]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_n,
        }
        
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # 解析返回结果（通用格式）
            results_indices = []
            if "results" in data:
                for item in data["results"]:
                    results_indices.append(item.get("index"))
            elif "data" in data:
                for item in data["data"]:
                    results_indices.append(item.get("index"))
            
            if results_indices:
                return [documents[i] for i in results_indices if i < len(documents)]
            else:
                logger.warning(f"{self._online_provider} API 返回结果为空，使用原始顺序")
                return documents[:top_n]
                
        except httpx.TimeoutException:
            logger.error(f"{self._online_provider} API 请求超时，跳过重排序")
            return documents[:top_n]
        except httpx.HTTPStatusError as e:
            logger.error(f"{self._online_provider} API 错误: {e.response.status_code}")
            return documents[:top_n]
        except Exception as e:
            logger.error(f"在线 Reranker 调用失败: {e}")
            return documents[:top_n]
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[Document]:
        """
        对检索结果进行重排序
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_n: 返回前 N 个文档
            use_reranker: 是否使用重排序（None 时跟随全局设置）
            
        Returns:
            重排序后的文档列表
        """
        top_n = top_n or settings.TOP_N
        
        # 如果明确禁用或全局已禁用
        if use_reranker is False or self._mode == RerankerMode.DISABLED:
            return documents[:top_n]
        
        if not documents:
            return documents
        
        start_time = time.time()
        logger.info(
            f"开始 Rerank | 模式={self._mode} | "
            f"输入文档数={len(documents)} | TopN={top_n}"
        )
        
        # 根据模式执行不同的重排序逻辑
        if self._mode == RerankerMode.ONLINE:
            result = self._rerank_online(query, documents, top_n)
        else:
            result = documents[:top_n]
        
        elapsed = time.time() - start_time
        logger.info(
            f"Rerank 完成 | {len(documents)} docs → {len(result)} docs | "
            f"耗时: {elapsed:.2f}秒 | 模式={self._mode}"
        )
        return result


# 全局单例
reranker = Reranker()
