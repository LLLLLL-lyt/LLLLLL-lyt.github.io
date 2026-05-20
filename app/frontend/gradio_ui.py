import gradio as gr
import os
import json
from typing import List, Dict, Any, Optional
from app.config import settings
# 使用新的配置系统
from app.llms import (
    config_manager,
    ModelType,
    get_available_llms,
    get_available_embeddings,
)
from app.models.schemas import LLMConfig
from app.rag.chain import rag_chain
from app.rag.vector_store import vector_store_manager
from app.rag.reranker import reranker, RerankerMode
from app.rag.splitter import splitter
from app.utils.helpers import generate_unique_id, get_file_md5, sanitize_filename
from datetime import datetime
from loguru import logger
# 导入新的设置界面
from app.frontend.settings_ui import create_settings_ui_new
# 导入新的知识库界面
from app.frontend.knowledge_ui import create_knowledge_base_ui, knowledge_bases, refresh_kb_list_data, refresh_kb_dropdown_data


def create_settings_ui():
    """系统设置界面 - 使用新的 Bisheng 风格配置"""
    return create_settings_ui_new()


# 获取已启用的模型 - 使用新的配置系统
def get_enabled_models(provider_id):
    """获取指定提供商下启用的LLM模型"""
    if not provider_id:
        return []
    
    # 重新加载配置以确保获取最新数据
    config_manager.load_from_file()
    
    server = config_manager.get_server(provider_id)
    if not server:
        return []
    
    models = []
    for model in server.models:
        if model.is_enabled and model.model_type == ModelType.LLM:
            models.append((model.model_name, model.model_id))
    return models


def get_configured_providers():
    """获取已配置API Key且有模型的提供商列表"""
    # 重新加载配置以确保获取最新数据
    config_manager.load_from_file()
    
    providers = []
    for server in config_manager.servers.values():
        # 只显示配置了API Key且有模型的提供商
        if server.api_key and server.models:
            # 检查是否有启用的LLM模型
            has_llm = any(m.is_enabled and m.model_type == ModelType.LLM for m in server.models)
            if has_llm:
                providers.append((server.provider_name, server.provider_id))
    return providers if providers else [("请先配置模型", "")]


def create_chat_ui():
    """聊天界面"""
    with gr.Column() as chat_column:
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="聊天",
                    height=500
                )
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的问题... (按回车发送，Shift+回车换行)",
                    scale=4,
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                    stop_btn = gr.Button("⏹ 停止生成", variant="stop", scale=1)
                    clear_btn = gr.Button("清空聊天", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("### 模型设置")
                
                # 获取已配置且启用的提供商 - 使用新的配置系统
                configured_providers = get_configured_providers()
                
                provider = gr.Dropdown(
                    label="提供商",
                    choices=configured_providers,
                    value=configured_providers[0][1] if configured_providers else None,
                    interactive=True
                )
                
                model = gr.Dropdown(
                    label="模型",
                    choices=[],
                    value="",
                    interactive=True
                )
                
                refresh_models_btn = gr.Button("🔄 刷新模型列表")
                
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1
                )
                
                top_p = gr.Slider(
                    label="Top P",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1
                )
                
                max_tokens = gr.Number(
                    label="Max Tokens",
                    value=2048,
                    minimum=1,
                    step=1
                )
                
                use_rag = gr.Checkbox(
                    label="使用 RAG 知识库",
                    value=True
                )
                
                knowledge_base = gr.Dropdown(
                    label="选择知识库",
                    choices=[],
                    value=None,
                    interactive=True
                )
                
                # Reranker 控件 - 开关 + 在线API模型选择
                use_reranker = gr.Checkbox(
                    label="使用重排序 (Reranker)",
                    value=False,
                    info="启用后对检索结果进行精排（使用在线 API）"
                )
                
                reranker_model = gr.Dropdown(
                    label="在线 Reranker 服务",
                    choices=[
                        ("Cohere Rerank (推荐)", "cohere"),
                        ("Jina Reranker (多语言)", "jina"),
                    ],
                    value="cohere",
                    interactive=True,
                    info="选择在线重排序服务提供商"
                )
                
                refresh_kb_btn = gr.Button("刷新知识库列表")
                
                def update_model_list_from_config(provider_id):
                    models = get_enabled_models(provider_id)
                    return gr.update(
                        choices=models,
                        value=models[0][1] if models else ""
                    )
                
                def refresh_kb_list():
                    """刷新知识库列表 - 从管理器重新读取最新数据"""
                    return gr.update(choices=refresh_kb_dropdown_data())
                
                provider.change(
                    update_model_list_from_config,
                    inputs=[provider],
                    outputs=[model]
                )
                
                refresh_models_btn.click(
                    update_model_list_from_config,
                    inputs=[provider],
                    outputs=[model]
                )
                
                refresh_kb_btn.click(
                    refresh_kb_list,
                    outputs=[knowledge_base]
                )
        
        def user(user_message, history):
            if history is None:
                history = []
            return "", history + [[user_message, None]]
        
        def bot(history, provider_id, model_name, temp, top_p_val, max_tok, use_rag_flag, kb_id, use_reranker_flag, reranker_model_sel):
            import time
            
            if not history or len(history) == 0:
                return history
            
            user_message = history[-1][0]
            
            # 检查模型是否已选择
            if not model_name:
                history[-1][1] = "❌ 请先选择模型! 点击右侧的 '🔄 刷新模型列表' 按钮加载可用模型。"
                yield history
                return
            
            # 从新的配置系统获取API配置
            server = config_manager.get_server(provider_id)
            if not server or not server.api_key:
                history[-1][1] = "❌ 请先在设置中配置 API Key!"
                yield history
                return
            
            # Reranker 使用在线 API 模式
            if use_reranker_flag and reranker_model_sel:
                reranker.set_mode("online", online_provider=reranker_model_sel)
            else:
                reranker.set_mode("disabled")
            
            try:
                llm_config = LLMConfig(
                    provider=provider_id,
                    model=model_name,
                    api_key=server.api_key,
                    api_base=server.api_base,
                    temperature=temp,
                    max_tokens=max_tok,
                    top_p=top_p_val
                )
                
                # 记录开始时间
                start_time = time.time()
                first_token_time = None
                total_chars = 0
                timing_data = {}  # 存储阶段耗时
                
                # 使用流式输出
                history[-1][1] = ""
                for chunk in rag_chain.stream(
                    user_message,
                    kb_id if use_rag_flag else None,
                    llm_config,
                    use_rag=use_rag_flag,
                    use_reranker=use_reranker_flag
                ):
                    # 过滤掉 sources 标记
                    if "__SOURCES__:" in chunk:
                        continue
                    
                    # 解析性能数据
                    if "__TIMING__:" in chunk:
                        timing_str = chunk.strip().replace("__TIMING__:", "")
                        for part in timing_str.split("|"):
                            if "=" in part:
                                key, val = part.split("=", 1)
                                timing_data[key.strip()] = float(val)
                        continue
                    
                    # 记录首 token 时间
                    if first_token_time is None and chunk.strip():
                        first_token_time = time.time()
                    
                    history[-1][1] += chunk
                    total_chars += len(chunk)
                    yield history
                
                # 计算统计信息
                end_time = time.time()
                total_time = end_time - start_time
                first_token_latency = (first_token_time - start_time) if first_token_time else 0
                
                # 构建响应统计
                rag_time = timing_data.get("retrieval", 0)
                gen_time = timing_data.get("generation", 0)
                
                stats_parts = []
                if rag_time > 0:
                    stats_parts.append(f"检索: {rag_time:.2f}s")
                if gen_time > 0:
                    stats_parts.append(f"生成: {gen_time:.2f}s")
                stats_parts.append(f"首字延迟: {first_token_latency:.2f}s")
                stats_parts.append(f"总耗时: {total_time:.2f}s")
                stats_parts.append(f"共 {total_chars} 字符")
                
                stats = "\n\n---\n⏱️ " + " | ".join(stats_parts)
                history[-1][1] += stats
                yield history
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"Chat error: {e}\n{error_detail}")
                history[-1][1] = f"❌ 错误: {str(e)}\n\n详细错误:\n```\n{error_detail[:500]}...\n```"
                yield history
        
        # 发送按钮点击
        submit_click_event = submit_btn.click(
            user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            bot,
            inputs=[chatbot, provider, model, temperature, top_p, max_tokens, use_rag, knowledge_base, use_reranker, reranker_model],
            outputs=[chatbot],
        )
        
        # 回车键触发发送（Shift+回车换行）
        msg.submit(
            user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            bot,
            inputs=[chatbot, provider, model, temperature, top_p, max_tokens, use_rag, knowledge_base, use_reranker, reranker_model],
            outputs=[chatbot],
        )
        
        # 停止按钮 - 取消正在运行的生成任务
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_click_event])
        
        # 清空聊天
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    # 返回组件引用供主界面使用
    return chat_column, provider, model, knowledge_base


def create_main_ui():
    """主界面"""
    with gr.Blocks(title="瓜皮智能聊天助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎉 瓜皮智能聊天助手")
        gr.Markdown("开源 RAG 智能聊天机器人 - 支持 12 家大语言模型")
        
        with gr.Tabs():
            with gr.Tab("💬 聊天"):
                chat_column, provider_dropdown, model_dropdown, knowledge_base = create_chat_ui()
            
            with gr.Tab("📚 知识库"):
                kb_column = create_knowledge_base_ui()
            
            with gr.Tab("⚙️ 设置"):
                gr.Markdown("# ⚙️ 系统设置")
                create_settings_ui()
        
        # 页面加载时自动加载模型列表和知识库列表
        def auto_load_models(provider_id):
            if provider_id:
                models = get_enabled_models(provider_id)
                return gr.update(choices=models, value=models[0][1] if models else "")
            return gr.update(choices=[], value="")
        
        def auto_load_kb_list():
            """页面加载时自动刷新知识库下拉框"""
            return gr.update(choices=refresh_kb_dropdown_data())
        
        demo.load(
            auto_load_models,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
        demo.load(
            auto_load_kb_list,
            outputs=[knowledge_base]
        )
    
    return demo
