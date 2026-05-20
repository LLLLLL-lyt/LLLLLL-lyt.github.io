from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gradio as gr
import os
from app.config import settings
from app.api import chat, knowledge, models
from app.frontend.gradio_ui import create_main_ui
from app.rag.reranker import reranker
from loguru import logger

# 禁用 Gradio 分析统计，避免网络超时警告
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# 禁用 Hugging Face symlink 警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="瓜皮智能聊天助手 - 开源RAG聊天机器人"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=f"{settings.API_PREFIX}/chat", tags=["chat"])
app.include_router(knowledge.router, prefix=f"{settings.API_PREFIX}/knowledge", tags=["knowledge"])
app.include_router(models.router, prefix=f"{settings.API_PREFIX}/models", tags=["models"])

@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "gradio": "http://localhost:8000/gradio"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

demo = create_main_ui()

app = gr.mount_gradio_app(
    app,
    demo,
    path="/gradio",
    root_path="/gradio"
)

if __name__ == "__main__":
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"📡 FastAPI API: http://localhost:{settings.BACKEND_PORT}")
    logger.info(f"📚 API Docs:   http://localhost:{settings.BACKEND_PORT}/docs")
    logger.info(f"🎨 Gradio UI:   http://localhost:{settings.BACKEND_PORT}/gradio")
    logger.info("=" * 60)
    
    # Reranker 模式日志
    logger.info(f"🔄 Reranker 模式: {reranker.mode}")
    if reranker.mode == "disabled":
        logger.info("⏭️ Reranker 已禁用，跳过预加载")
    elif reranker.mode == "online":
        logger.info("☁️ 使用在线 Reranker API，无需本地预加载")
    
    uvicorn.run(
        app,
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        log_level="info"
    )
