"""
向量存储管理器 - 支持多知识库、多 Embedding 模型
"""

import os
import uuid
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from app.config import settings
from app.rag.embeddings import embedding_manager
from loguru import logger


class VectorStoreManager:
    """向量存储管理器 - 每个知识库使用自己的 Embedding 模型"""
    
    def __init__(self):
        self.base_dir = settings.VECTOR_STORE_DIR
        # 每个知识库对应一个向量存储
        self.vector_stores: Dict[str, VectorStore] = {}
        self.bm25_retrievers: Dict[str, BM25Retriever] = {}
        self.documents_cache: Dict[str, List[Document]] = {}
        # 记录每个知识库使用的 Embedding 模型
        self.kb_embedding_models: Dict[str, str] = {}
    
    def _get_vector_store_path(self, kb_id: str) -> str:
        return os.path.join(self.base_dir, kb_id)
    
    def _get_embeddings(self, kb_id: str, embedding_model_id: Optional[str] = None) -> Embeddings:
        """
        获取知识库对应的 Embedding 模型 - 只支持在线 Embedding
        
        Args:
            kb_id: 知识库 ID
            embedding_model_id: 可选，指定 Embedding 模型 ID（格式：provider_id/model_id）
        
        Returns:
            Embeddings 实例
        """
        # 优先使用指定的模型 ID
        model_id = embedding_model_id or self.kb_embedding_models.get(kb_id)
        
        logger.info(f"获取 Embedding 模型: kb_id={kb_id}, model_id={model_id}")
        
        if not model_id:
            raise ValueError(
                f"知识库 {kb_id} 未配置 Embedding 模型。"
                "请在创建知识库时选择向量模型，或在设置中配置 Embedding 模型。"
            )
        
        try:
            embeddings = embedding_manager.get_embeddings_for_kb(model_id)
            logger.info(f"成功获取 Embedding 模型: {model_id}")
            return embeddings
        except Exception as e:
            logger.error(f"获取 Embedding 模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_or_create_vector_store(
        self, 
        kb_id: str, 
        embedding_model_id: Optional[str] = None
    ) -> VectorStore:
        """
        获取或创建向量存储
        
        Args:
            kb_id: 知识库 ID
            embedding_model_id: Embedding 模型 ID（格式：provider_id/model_id）
        """
        # 如果知识库已存在且 Embedding 模型没变，直接返回
        if kb_id in self.vector_stores:
            current_model = self.kb_embedding_models.get(kb_id)
            if current_model == embedding_model_id or (not embedding_model_id and not current_model):
                return self.vector_stores[kb_id]
            # Embedding 模型变了，需要重新创建
            logger.warning(f"知识库 {kb_id} 的 Embedding 模型已变更，重新创建向量存储")
            del self.vector_stores[kb_id]
        
        # 记录使用的 Embedding 模型
        if embedding_model_id:
            self.kb_embedding_models[kb_id] = embedding_model_id
        
        persist_path = self._get_vector_store_path(kb_id)
        embeddings = self._get_embeddings(kb_id, embedding_model_id)
        
        vector_store = Chroma(
            collection_name=f"kb_{kb_id}",
            embedding_function=embeddings,
            persist_directory=persist_path,
        )
        self.vector_stores[kb_id] = vector_store
        
        if kb_id not in self.documents_cache:
            self.documents_cache[kb_id] = []
        
        logger.info(f"创建向量存储: {kb_id}, Embedding: {embedding_model_id or 'default'}")
        return vector_store
    
    def add_documents(
        self, 
        kb_id: str, 
        documents: List[Document], 
        embedding_model_id: Optional[str] = None
    ) -> List[str]:
        """
        添加文档到知识库
        
        Args:
            kb_id: 知识库 ID
            documents: 文档列表
            embedding_model_id: Embedding 模型 ID
        """
        if not documents:
            logger.warning(f"尝试添加空文档列表到知识库 {kb_id}")
            return []
        
        logger.info(f"开始添加 {len(documents)} 个文档到知识库 {kb_id}")
        
        vector_store = self.get_or_create_vector_store(kb_id, embedding_model_id)
        
        # 为每个文档生成 ID
        for doc in documents:
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = str(uuid.uuid4())
        
        logger.info(f"文档元数据示例: {documents[0].metadata if documents else 'N/A'}")
        
        try:
            ids = vector_store.add_documents(documents)
            logger.info(f"成功添加文档，返回 IDs: {ids}")
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # 初始化文档缓存
        if kb_id not in self.documents_cache:
            self.documents_cache[kb_id] = []
        self.documents_cache[kb_id].extend(documents)
        
        self._update_bm25_retriever(kb_id)
        
        logger.info(f"Added {len(documents)} chunks to knowledge base {kb_id}")
        return ids
    
    def _update_bm25_retriever(self, kb_id: str):
        """更新 BM25 检索器"""
        if kb_id in self.documents_cache and self.documents_cache[kb_id]:
            self.bm25_retrievers[kb_id] = BM25Retriever.from_documents(
                self.documents_cache[kb_id],
                k=settings.TOP_K
            )
    
    def get_hybrid_retriever(
        self, 
        kb_id: str, 
        top_k: Optional[int] = None,
        embedding_model_id: Optional[str] = None
    ):
        """
        获取混合检索器（向量检索 + BM25）
        
        Args:
            kb_id: 知识库 ID
            top_k: 返回结果数量
            embedding_model_id: Embedding 模型 ID
        """
        import time as _time
        total_start = _time.time()
        
        top_k = top_k or settings.TOP_K
        
        # 阶段1：创建/获取向量存储
        t1 = _time.time()
        vector_store = self.get_or_create_vector_store(kb_id, embedding_model_id)
        logger.info(f"[检索耗时] 向量存储初始化: {(_time.time()-t1):.2f}s")
        
        # 阶段2：创建稠密检索器
        t2 = _time.time()
        dense_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        logger.info(f"[检索耗时] 创建稠密检索器: {(_time.time()-t2):.2f}s")
        
        # 阶段3：BM25 检索器（可能需要从向量库加载全部文档来初始化）
        t3 = _time.time()
        if kb_id not in self.bm25_retrievers or not self.bm25_retrievers[kb_id]:
            logger.info(f"[检索耗时] BM25 未缓存，开始从向量库加载文档...")
            t3_inner = _time.time()
            docs = vector_store.get()
            logger.info(f"[检索耗时] vector_store.get() 完成: {(_time.time()-t3_inner):.2f}s")
            
            if docs and docs["documents"]:
                t3_build = _time.time()
                documents = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(docs["documents"], docs["metadatas"])
                ]
                logger.info(f"[检索耗时] 文档解析完成 ({len(documents)} 条): {(_time.time()-t3_build):.2f}s")
                
                self.documents_cache[kb_id] = documents
                t3_bm25 = _time.time()
                self.bm25_retrievers[kb_id] = BM25Retriever.from_documents(
                    documents, k=top_k
                )
                logger.info(f"[检索耗时] BM25 检索器构建完成: {(_time.time()-t3_bm25):.2f}s")
            else:
                logger.warning(f"[检索耗时] 知识库 {kb_id} 无文档")
        else:
            logger.info(f"[检索耗时] BM25 已命中缓存，跳过初始化")
        
        bm25_time = _time.time() - t3
        logger.info(f"[检索耗时] BM25 总计: {bm25_time:.2f}s")
        
        # 阶段4：组合混合检索器
        t4 = _time.time()
        if kb_id in self.bm25_retrievers:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, self.bm25_retrievers[kb_id]],
                weights=[0.6, 0.4]
            )
            logger.info(f"[检索耗时] 组合检索器创建: {(_time.time()-t4):.2f}s")
            logger.info(f"[检索总计] get_hybrid_retriever 总耗时: {(_time.time()-total_start):.2f}s")
            return ensemble_retriever
        
        logger.info(f"[检索总计] get_hybrid_retriever 总耗时(仅向量): {(_time.time()-total_start):.2f}s")
        return dense_retriever
    
    def delete_knowledge_base(self, kb_id: str):
        """删除知识库"""
        if kb_id in self.vector_stores:
            del self.vector_stores[kb_id]
        if kb_id in self.bm25_retrievers:
            del self.bm25_retrievers[kb_id]
        if kb_id in self.documents_cache:
            del self.documents_cache[kb_id]
        if kb_id in self.kb_embedding_models:
            del self.kb_embedding_models[kb_id]
        
        persist_path = self._get_vector_store_path(kb_id)
        if os.path.exists(persist_path):
            import shutil
            shutil.rmtree(persist_path)
        
        logger.info(f"Deleted knowledge base {kb_id}")
    
    def delete_documents(self, kb_id: str, file_id: str, embedding_model_id: Optional[str] = None):
        """
        从知识库中删除指定文件的所有分块
        
        Args:
            kb_id: 知识库 ID
            file_id: 文件 ID
            embedding_model_id: Embedding 模型 ID
        """
        vector_store = self.get_or_create_vector_store(kb_id, embedding_model_id)
        
        # 获取该文件的所有分块
        results = vector_store.get(where={"source": file_id})
        
        if not results or not results.get("ids"):
            logger.warning(f"未找到文件 {file_id} 的分块")
            return
        
        ids_to_delete = results["ids"]
        logger.info(f"准备删除 {len(ids_to_delete)} 个分块: {ids_to_delete}")
        
        # 删除分块
        vector_store.delete(ids=ids_to_delete)
        
        # 更新文档缓存
        if kb_id in self.documents_cache:
            self.documents_cache[kb_id] = [
                doc for doc in self.documents_cache[kb_id]
                if doc.metadata.get("source") != file_id
            ]
        
        # 更新 BM25 检索器
        self._update_bm25_retriever(kb_id)
        
        logger.info(f"已删除文件 {file_id} 的 {len(ids_to_delete)} 个分块")
    
    def get_document_count(self, kb_id: str) -> int:
        """获取知识库文档数量"""
        vector_store = self.get_or_create_vector_store(kb_id)
        result = vector_store.get()
        return len(result["ids"]) if result and result["ids"] else 0


# 全局向量存储管理器实例
vector_store_manager = VectorStoreManager()
