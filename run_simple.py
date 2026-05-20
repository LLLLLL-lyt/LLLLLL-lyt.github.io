import gradio as gr
import sys
import os
from loguru import logger

# 禁用 Gradio 分析统计，避免网络超时警告
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/vector_stores", exist_ok=True)

logger.info("=" * 60)
logger.info("🚀 瓜皮智能聊天助手 - 简单启动模式")
logger.info("=" * 60)

# Reranker 模式日志
from app.rag.reranker import reranker
logger.info(f"🔄 Reranker 模式: {reranker.mode}")
if reranker.mode == "disabled":
    logger.info("⏭️ Reranker 已禁用，跳过预加载")
elif reranker.mode == "online":
    logger.info("️ 使用在线 Reranker API，无需本地预加载")

try:
    from app.frontend.gradio_ui import create_main_ui
    logger.info("✅ Gradio UI 加载成功")
    
    demo = create_main_ui()
    logger.info("🎉 Gradio 应用创建成功")
    logger.info("📡 访问地址: http://localhost:7860")
    logger.info("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
    
except Exception as e:
    logger.error(f"❌ 启动失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    input("\n按回车键退出...")
