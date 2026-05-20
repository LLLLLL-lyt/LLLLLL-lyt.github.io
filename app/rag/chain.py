from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from app.rag.vector_store import vector_store_manager
from app.rag.knowledge_base import kb_manager
from app.rag.reranker import reranker
from app.llms.providers import get_llm_provider
from app.models.schemas import ChatMessage, LLMConfig
from loguru import logger

RAG_PROMPT_TEMPLATE = """你是一个专业的智能助手。请基于以下提供的上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答要求：
1. 基于提供的上下文信息回答，不要编造内容
2. 如果上下文中没有相关信息，请明确说明
3. 回答要简洁、准确、有条理
4. 请用中文回答"""

class RAGChain:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def _format_context(self, documents: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[文档 {i}]\n{doc.page_content}\n")
        return "\n".join(context_parts)
    
    def _get_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        sources = []
        for doc in documents:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        return sources
    
    def invoke(
        self,
        query: str,
        kb_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_rag: bool = True,
        use_reranker: Optional[bool] = None
    ) -> Dict[str, Any]:
        llm = get_llm_provider(llm_config)
        
        context_docs = []
        sources = []
        
        if use_rag and kb_id:
            try:
                # 获取知识库的 Embedding 模型配置
                embedding_model = kb_manager.get_kb_embedding_model(kb_id)
                logger.info(f"使用知识库 {kb_id} 的 Embedding 模型: {embedding_model}")
                
                retriever = vector_store_manager.get_hybrid_retriever(
                    kb_id, 
                    embedding_model_id=embedding_model
                )
                retrieved_docs = retriever.invoke(query)
                context_docs = reranker.rerank(query, retrieved_docs, use_reranker=use_reranker)
                sources = self._get_sources(context_docs)
                logger.info(f"Retrieved {len(context_docs)} documents for query")
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
        
        if context_docs:
            context = self._format_context(context_docs)
            prompt = self.prompt.format(context=context, question=query)
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = [ChatMessage(role="user", content=query)]
        
        result = llm.chat(messages)
        result["sources"] = sources
        return result
    
    async def ainvoke(
        self,
        query: str,
        kb_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_rag: bool = True,
        use_reranker: Optional[bool] = None
    ) -> Dict[str, Any]:
        llm = get_llm_provider(llm_config)
        
        context_docs = []
        sources = []
        
        if use_rag and kb_id:
            try:
                # 获取知识库的 Embedding 模型配置
                embedding_model = kb_manager.get_kb_embedding_model(kb_id)
                logger.info(f"使用知识库 {kb_id} 的 Embedding 模型: {embedding_model}")
                
                retriever = vector_store_manager.get_hybrid_retriever(
                    kb_id, 
                    embedding_model_id=embedding_model
                )
                retrieved_docs = retriever.invoke(query)
                context_docs = reranker.rerank(query, retrieved_docs, use_reranker=use_reranker)
                sources = self._get_sources(context_docs)
                logger.info(f"Retrieved {len(context_docs)} documents for query")
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
        
        if context_docs:
            context = self._format_context(context_docs)
            prompt = self.prompt.format(context=context, question=query)
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = [ChatMessage(role="user", content=query)]
        
        result = await llm.achat(messages)
        result["sources"] = sources
        return result
    
    def query(
        self,
        query: str,
        llm_config: Optional[LLMConfig] = None,
        kb_id: Optional[str] = None
    ) -> str:
        """简化的查询接口，返回字符串回答"""
        result = self.invoke(query, kb_id, llm_config, use_rag=kb_id is not None)
        return result.get("answer", "")
    
    def stream(
        self,
        query: str,
        kb_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_rag: bool = True,
        use_reranker: Optional[bool] = None
    ):
        import time as _time
        llm = get_llm_provider(llm_config)
        
        context_docs = []
        sources = []
        retrieval_total = 0.0
        
        if use_rag and kb_id:
            retrieval_start = _time.time()
            retrieval_total = 0.0  # ensure always defined
            
            try:
                # 获取知识库的 Embedding 模型配置
                embedding_model = kb_manager.get_kb_embedding_model(kb_id)
                logger.info(f"使用知识库 {kb_id} 的 Embedding 模型: {embedding_model}")
                
                retriever = vector_store_manager.get_hybrid_retriever(
                    kb_id, 
                    embedding_model_id=embedding_model
                )
                
                t_invoke = _time.time()
                retrieved_docs = retriever.invoke(query)
                logger.info(f"[阶段耗时] 检索完成 ({len(retrieved_docs)} 条): {(_time.time()-t_invoke):.2f}s")
                
                context_docs = reranker.rerank(query, retrieved_docs, use_reranker=use_reranker)
                
                sources = self._get_sources(context_docs)
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
            
            # 无论成功失败都记录 RAG 阶段总耗时
            retrieval_total = _time.time() - retrieval_start
            logger.info(f"[阶段耗时] RAG 总耗时: {retrieval_total:.2f}s | ⏱️ 进入 LLM 生成")
        
        if context_docs:
            context = self._format_context(context_docs)
            prompt = self.prompt.format(context=context, question=query)
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = [ChatMessage(role="user", content=query)]
        
        generation_start = _time.time()
        for chunk in llm.stream_chat(messages):
            yield chunk
        
        generation_total = _time.time() - generation_start
        
        yield f"\n\n__SOURCES__:{str(sources)}"
        # 输出性能指标
        yield f"\n\n__TIMING__:retrieval={retrieval_total:.2f}|generation={generation_total:.2f}"
    
    async def astream(
        self,
        query: str,
        kb_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_rag: bool = True,
        use_reranker: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        llm = get_llm_provider(llm_config)
        
        context_docs = []
        sources = []
        
        if use_rag and kb_id:
            try:
                retriever = vector_store_manager.get_hybrid_retriever(kb_id)
                retrieved_docs = retriever.invoke(query)
                context_docs = reranker.rerank(query, retrieved_docs, use_reranker=use_reranker)
                sources = self._get_sources(context_docs)
                logger.info(f"Retrieved {len(context_docs)} documents for query")
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
        
        if context_docs:
            context = self._format_context(context_docs)
            prompt = self.prompt.format(context=context, question=query)
            messages = [ChatMessage(role="user", content=prompt)]
        else:
            messages = [ChatMessage(role="user", content=query)]
        
        async for chunk in llm.astream_chat(messages):
            yield chunk
        
        yield f"\n\n__SOURCES__:{str(sources)}"

rag_chain = RAGChain()
