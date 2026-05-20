from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional
from app.models.schemas import ChatRequest, ChatResponse
from app.rag.chain import rag_chain
from app.llms.providers import LLM_REGISTRY
from app.config import settings
from loguru import logger

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        from app.models.schemas import LLMConfig
        
        llm_config = LLMConfig(
            provider=request.provider,
            api_key="",
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        
        if request.stream:
            return StreamingResponse(
                rag_chain.astream(
                    query=request.query,
                    kb_id=request.knowledge_base_id,
                    llm_config=llm_config,
                    use_rag=request.use_rag
                ),
                media_type="text/plain"
            )
        else:
            result = await rag_chain.ainvoke(
                query=request.query,
                kb_id=request.knowledge_base_id,
                llm_config=llm_config,
                use_rag=request.use_rag
            )
            return ChatResponse(
                answer=result["answer"],
                sources=result.get("sources"),
                usage=result.get("usage")
            )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        from app.models.schemas import LLMConfig
        
        llm_config = LLMConfig(
            provider=request.provider,
            api_key="",
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        
        async def generate():
            async for chunk in rag_chain.astream(
                query=request.query,
                kb_id=request.knowledge_base_id,
                llm_config=llm_config,
                use_rag=request.use_rag
            ):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Stream chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
