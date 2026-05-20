"""
系统设置界面 - 参考 Bisheng 平台的配置方式
每个提供商可以配置多个模型，支持 LLM、Embedding、Reranker 等多种类型
"""

import gradio as gr
from typing import Dict, List, Tuple, Any
from app.llms import (
    config_manager,
    LLMServer,
    LLMModel,
    ModelType,
    ProviderType,
    DEFAULT_SERVER_TEMPLATES,
)
from app.config import settings
from app.rag.reranker import reranker, RerankerMode
from loguru import logger
import os


def _get_model_type_label(m_type: ModelType) -> str:
    """获取模型类型显示标签"""
    labels = {
        ModelType.LLM: "大语言模型",
        ModelType.EMBEDDING: "向量模型",
        ModelType.RERANK: "重排模型",
        ModelType.SPEECH_TTS: "语音合成",
        ModelType.SPEECH_ASR: "语音识别",
        ModelType.VISION: "视觉模型",
    }
    return labels.get(m_type, m_type.value)


def create_llm_settings_ui():
    """创建 LLM 模型配置界面 - 参考 Bisheng 的设计"""

    # ==================== 辅助函数 ====================

    def get_provider_choices() -> List[Tuple[str, str]]:
        return [(s.provider_name, s.provider_id) for s in config_manager.get_all_servers()]

    def get_provider_info_markdown(provider_id: str) -> str:
        server = config_manager.get_server(provider_id)
        if not server:
            return "### 请先选择提供商"
        links = []
        if server.api_key_url:
            links.append(f"[获取API Key]({server.api_key_url})")
        if server.doc_url:
            links.append(f"[查看文档]({server.doc_url})")
        if server.models_url:
            links.append(f"[模型列表]({server.models_url})")
        links_str = " | ".join(links) if links else ""
        return f"""### {server.provider_name}
{links_str}
**默认 API Base:** {server.api_base or '使用系统默认'}
**提供商类型:** {server.provider_type.value}
"""

    def load_provider_config(provider_id: str):
        """加载提供商配置"""
        server = config_manager.get_server(provider_id)
        if not server:
            return "提供商不存在", "", "", []

        info = get_provider_info_markdown(provider_id)
        model_table = _build_model_table(server)
        return info, server.api_key or "", server.api_base or "", model_table

    def _build_model_table(server: LLMServer) -> List[List]:
        """构建模型表格数据: [模型ID, 模型名称, 类型, 启用状态]"""
        data = []
        for model in server.models:
            data.append([
                model.model_id,
                model.model_name,
                _get_model_type_label(model.model_type),
                "启用" if model.is_enabled else "禁用",
            ])
        return data

    def save_provider_config(provider_id: str, api_key: str, api_base: str):
        """保存提供商的 API 配置"""
        if not provider_id:
            return "请先选择提供商"
        server = config_manager.get_server(provider_id)
        if not server:
            return "提供商不存在"
        config_manager.update_server(
            provider_id,
            api_key=api_key if api_key else None,
            api_base=api_base if api_base else None,
        )
        config_manager.save_to_file()
        return f"✅ {server.provider_name} 配置已保存！"

    def refresh_model_table(provider_id: str):
        """刷新模型表格"""
        server = config_manager.get_server(provider_id)
        if not server:
            return []
        return _build_model_table(server)

    def toggle_model(provider_id: str, model_id: str):
        """切换模型启用/禁用状态"""
        if not provider_id or not model_id:
            return "请选择提供商和模型", []
        server = config_manager.get_server(provider_id)
        if not server:
            return "提供商不存在", []
        model = server.get_model(model_id)
        if not model:
            return "模型不存在", []
        model.is_enabled = not model.is_enabled
        model.updated_at = __import__('datetime').datetime.now()
        config_manager.save_to_file()
        status = f"✅ {model.model_name} 已{'启用' if model.is_enabled else '禁用'}"
        return status, _build_model_table(server)

    def delete_model(provider_id: str, model_id: str):
        """删除模型"""
        if not provider_id or not model_id:
            return "请选择提供商和模型", []
        server = config_manager.get_server(provider_id)
        if not server:
            return "提供商不存在", []
        model = server.get_model(model_id)
        if not model:
            return "模型不存在", []
        model_name = model.model_name
        server.remove_model(model_id)
        config_manager.save_to_file()
        return f"✅ 已删除模型: {model_name}", _build_model_table(server)

    def add_model(provider_id: str, model_id: str, model_name: str, model_type: str):
        """添加新模型"""
        if not provider_id or not model_id:
            return "请填写模型ID", []
        server = config_manager.get_server(provider_id)
        if not server:
            return "提供商不存在", []
        if server.get_model(model_id):
            return f"模型 {model_id} 已存在", _build_model_table(server)
        try:
            m_type = ModelType(model_type)
        except ValueError:
            return "无效的模型类型", _build_model_table(server)
        new_model = LLMModel(
            model_id=model_id,
            model_name=model_name or model_id,
            model_type=m_type,
            is_enabled=True,
        )
        server.add_model(new_model)
        config_manager.save_to_file()
        return f"✅ 已添加模型: {model_name or model_id}", _build_model_table(server)

    # ==================== 构建界面 ====================

    with gr.Column() as llm_column:
        gr.Markdown("## 大语言模型配置")
        gr.Markdown("配置各厂商的 API Key 和模型，一个配置可以包含多种类型的模型")

        with gr.Row():
            # 左侧：提供商选择 + 添加模型
            with gr.Column(scale=1):
                gr.Markdown("### 选择提供商")
                provider_dropdown = gr.Dropdown(
                    label="提供商",
                    choices=get_provider_choices(),
                    value=get_provider_choices()[0][1] if get_provider_choices() else None,
                )
                load_btn = gr.Button("加载配置", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### 添加模型")
                add_model_id = gr.Textbox(label="模型ID", placeholder="如: gpt-4")
                add_model_name = gr.Textbox(label="模型名称", placeholder="如: GPT-4")
                add_model_type = gr.Dropdown(
                    label="模型类型",
                    choices=[
                        (ModelType.LLM.value, ModelType.LLM.value),
                        (ModelType.EMBEDDING.value, ModelType.EMBEDDING.value),
                        (ModelType.RERANK.value, ModelType.RERANK.value),
                        (ModelType.SPEECH_TTS.value, ModelType.SPEECH_TTS.value),
                        (ModelType.SPEECH_ASR.value, ModelType.SPEECH_ASR.value),
                        (ModelType.VISION.value, ModelType.VISION.value),
                    ],
                    value=ModelType.LLM.value,
                )
                add_btn = gr.Button("添加模型", variant="secondary")
                add_status = gr.Textbox(label="状态", interactive=False)

            # 右侧：配置详情 + 模型列表
            with gr.Column(scale=2):
                provider_info = gr.Markdown("点击'加载配置'查看提供商信息")

                gr.Markdown("### API 配置")
                with gr.Row():
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="请输入 API Key")
                    api_base_input = gr.Textbox(label="API Base (可选)", placeholder="使用默认地址")

                with gr.Row():
                    save_btn = gr.Button("保存配置", variant="primary")
                    reset_btn = gr.Button("重置", variant="secondary")
                status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### 模型列表")
                model_table = gr.Dataframe(
                    headers=["模型ID", "模型名称", "类型", "状态"],
                    label="已配置模型",
                    interactive=False,
                    value=[],
                )

                with gr.Row():
                    toggle_btn = gr.Button("切换启用/禁用", variant="secondary")
                    delete_btn = gr.Button("删除选中模型", variant="stop")
                action_status = gr.Textbox(label="操作结果", interactive=False)

        # ==================== 事件绑定 ====================

        load_btn.click(
            load_provider_config,
            inputs=[provider_dropdown],
            outputs=[provider_info, api_key_input, api_base_input, model_table],
        )

        save_btn.click(
            save_provider_config,
            inputs=[provider_dropdown, api_key_input, api_base_input],
            outputs=[status],
        )

        add_btn.click(
            add_model,
            inputs=[provider_dropdown, add_model_id, add_model_name, add_model_type],
            outputs=[add_status, model_table],
        )

        # 点击表格行时，将模型ID填入操作区域
        def on_model_select(evt: gr.SelectData, provider_id: str):
            if not evt or not evt.value:
                return "", ""
            row = evt.index[0] if evt.index else 0
            server = config_manager.get_server(provider_id)
            if not server:
                return "", ""
            if row < len(server.models):
                model = server.models[row]
                return model.model_id, model.model_id
            return "", ""

        model_table.select(
            on_model_select,
            inputs=[provider_dropdown],
            outputs=[add_model_id, delete_btn],  # 这里简化处理
        )

        toggle_btn.click(
            lambda pid, mid: toggle_model(pid, mid),
            inputs=[provider_dropdown, add_model_id],
            outputs=[action_status, model_table],
        )

        delete_btn.click(
            lambda pid, mid: delete_model(pid, mid),
            inputs=[provider_dropdown, add_model_id],
            outputs=[action_status, model_table],
        )

        # 默认加载第一个提供商
        if get_provider_choices():
            first_provider = get_provider_choices()[0][1]
            # 通过JS自动触发加载
            provider_dropdown.value = first_provider
            # 用户需要手动点击"加载配置"按钮

    return llm_column


def create_model_list_ui():
    """创建模型列表界面"""

    def refresh_model_list():
        config_manager.load_from_file()
        data = []
        for provider_id, server in config_manager.servers.items():
            if not server.api_key:
                continue
            for model in server.models:
                data.append([
                    server.provider_name,
                    model.model_name,
                    model.model_id,
                    _get_model_type_label(model.model_type),
                    "启用" if model.is_enabled else "禁用",
                    "已配置" if server.api_key else "未配置",
                ])
        if not data:
            return [["暂无配置", "请先配置API Key并添加模型", "-", "-", "-", "-"]]
        return data

    with gr.Column() as model_list_column:
        gr.Markdown("## 模型列表")
        gr.Markdown("显示所有已配置 API Key 的提供商的模型")
        refresh_btn = gr.Button("刷新列表", variant="secondary")
        model_table = gr.Dataframe(
            headers=["提供商", "模型名称", "模型ID", "类型", "状态", "API配置"],
            label="模型列表",
            interactive=False,
            value=refresh_model_list(),
        )
        refresh_btn.click(refresh_model_list, outputs=[model_table])

    return model_list_column


def create_embedding_settings_ui():
    """创建向量模型配置界面"""

    def get_embedding_choices() -> List[Tuple[str, str]]:
        choices = []
        for provider_id, server in config_manager.servers.items():
            if not server.is_enabled or not server.api_key:
                continue
            for model in server.models:
                if model.model_type == ModelType.EMBEDDING and model.is_enabled:
                    full_id = f"{provider_id}/{model.model_id}"
                    display = f"{server.provider_name} - {model.model_name}"
                    choices.append((display, full_id))
        return choices

    def set_default_embedding(full_model_id: str) -> str:
        if not full_model_id or "/" not in full_model_id:
            return "请选择有效的模型"
        provider_id, model_id = full_model_id.split("/", 1)
        config_manager.set_default_embedding(provider_id, model_id)
        config_manager.save_to_file()
        return f"✅ 已设置默认向量模型: {model_id}"

    with gr.Column() as embedding_column:
        gr.Markdown("## 向量模型配置")
        gr.Markdown("用于文本向量化和知识库检索")

        embedding_choices = get_embedding_choices()

        if embedding_choices:
            gr.Markdown("### 设置默认向量模型")
            default_embedding = gr.Dropdown(
                label="默认向量模型",
                choices=embedding_choices,
                value=embedding_choices[0][1] if embedding_choices else None,
            )
            set_default_btn = gr.Button("设为默认", variant="primary")
            status = gr.Textbox(label="状态", interactive=False)
            set_default_btn.click(set_default_embedding, inputs=[default_embedding], outputs=[status])
        else:
            gr.Markdown("### 暂无可用向量模型")
            gr.Markdown("请先配置LLM提供商的API Key，并在模型配置中启用Embedding模型。")

    return embedding_column


def create_reranker_settings_ui():
    """创建 Reranker 重排序配置界面"""

    def get_current_mode_info():
        mode = reranker.mode
        mode_labels = {
            RerankerMode.DISABLED: ("禁用", "不使用重排序，直接返回检索结果（速度最快）"),
            RerankerMode.ONLINE: ("在线 API", f"使用在线服务: {reranker._online_provider}"),
        }
        label, desc = mode_labels.get(mode, (mode, ""))
        status_text = f"### 当前模式: **{label}**\n\n{desc}"
        return status_text, mode

    def apply_reranker_config(mode, online_provider, api_key):
        result = reranker.set_mode(
            mode=mode,
            online_provider=online_provider if online_provider else None,
            api_key=api_key if api_key else None,
        )
        status_info, current_mode = get_current_mode_info()
        return result + "\n\n" + status_info, current_mode

    with gr.Column() as reranker_column:
        gr.Markdown("## 重排序器 (Reranker) 配置")
        gr.Markdown("对检索结果进行精排，提升相关性但会增加延迟。")

        gr.Markdown("""
| 模式 | 说明 | 延迟 | 推荐场景 |
|------|------|------|----------|
| **disabled** | 禁用，跳过重排序 | 最快 | 测试/快速响应 |
| **online** | 调用 Cohere/Jina API | 低 (~0.2s) | 有 API Key，追求效果 |
        """)

        reranker_status = gr.Markdown(get_current_mode_info()[0])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 模式设置")
                reranker_mode = gr.Radio(
                    label="运行模式",
                    choices=[
                        ("禁用 (Disabled)", RerankerMode.DISABLED),
                        ("在线 API (Online)", RerankerMode.ONLINE),
                    ],
                    value=reranker.mode,
                )
                online_provider = gr.Dropdown(
                    label="在线服务提供商",
                    choices=[("Cohere Rerank", "cohere"), ("Jina Reranker", "jina")],
                    value=reranker._online_provider,
                    info="选择在线 Reranker 服务提供商",
                )
                online_api_key = gr.Textbox(
                    label="在线 API Key",
                    type="password",
                    placeholder="输入 API Key",
                    info="API Key 保存在内存中，重启后需重新填写",
                )
                apply_btn = gr.Button("应用配置", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 使用说明")
                gr.Markdown("""
#### 在线模式（推荐）
- **Cohere**: 免费额度足够日常 [获取Key](https://dashboard.cohere.com/api-keys)
- **Jina**: 中文多语言支持好 [获取Key](https://jina.ai/api-key/)
- 需要: `pip install httpx`
                """)

        config_status = gr.Textbox(label="操作结果", interactive=False, lines=3)

        apply_btn.click(
            apply_reranker_config,
            inputs=[reranker_mode, online_provider, online_api_key],
            outputs=[config_status, reranker_mode],
        )

    return reranker_column


def create_rag_settings_ui():
    """创建 RAG 检索参数配置界面"""

    def load_rag_config():
        """加载当前 RAG 配置"""
        return (
            settings.TOP_K,
            settings.TOP_N if hasattr(settings, 'TOP_N') else 3,
            settings.CHUNK_SIZE if hasattr(settings, 'CHUNK_SIZE') else 500,
            settings.CHUNK_OVERLAP if hasattr(settings, 'CHUNK_OVERLAP') else 100,
        )

    def save_rag_config(top_k: int, top_n: int, chunk_size: int, chunk_overlap: int) -> str:
        """保存 RAG 配置到环境变量"""
        if top_k < 1 or top_k > 20:
            return "TOP_K 范围: 1-20"
        if top_n < 1 or top_n > 20:
            return "TOP_N 范围: 1-20"
        if chunk_size < 100 or chunk_size > 2000:
            return "CHUNK_SIZE 范围: 100-2000"
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            return "CHUNK_OVERLAP 必须小于 CHUNK_SIZE"

        os.environ["TOP_K"] = str(top_k)
        os.environ["TOP_N"] = str(top_n)
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)

        # 更新 settings 对象
        settings.TOP_K = top_k
        if hasattr(settings, 'TOP_N'):
            settings.TOP_N = top_n
        if hasattr(settings, 'CHUNK_SIZE'):
            settings.CHUNK_SIZE = chunk_size
        if hasattr(settings, 'CHUNK_OVERLAP'):
            settings.CHUNK_OVERLAP = chunk_overlap

        return f"✅ RAG 检索参数已保存\n- TOP_K: {top_k}\n- TOP_N: {top_n}\n- CHUNK_SIZE: {chunk_size}\n- CHUNK_OVERLAP: {chunk_overlap}"

    with gr.Column() as rag_column:
        gr.Markdown("## RAG 检索参数配置")
        gr.Markdown("配置检索和文档分块的参数")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 检索参数")
                top_k_input = gr.Slider(
                    label="TOP_K (检索返回数量)",
                    minimum=1,
                    maximum=20,
                    value=settings.TOP_K,
                    step=1,
                    info="向量检索时返回的文档数量，默认 5",
                )
                top_n_input = gr.Slider(
                    label="TOP_N (重排后保留数量)",
                    minimum=1,
                    maximum=20,
                    value=getattr(settings, 'TOP_N', 3),
                    step=1,
                    info="Reranker 精排后保留的文档数量，默认 3",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 分块参数")
                chunk_size_input = gr.Slider(
                    label="CHUNK_SIZE (文档分块大小)",
                    minimum=100,
                    maximum=2000,
                    value=getattr(settings, 'CHUNK_SIZE', 500),
                    step=50,
                    info="每个文本块的字符数，默认 500",
                )
                chunk_overlap_input = gr.Slider(
                    label="CHUNK_OVERLAP (分块重叠)",
                    minimum=0,
                    maximum=500,
                    value=getattr(settings, 'CHUNK_OVERLAP', 100),
                    step=10,
                    info="相邻文本块重叠的字符数，默认 100",
                )

        with gr.Row():
            save_rag_btn = gr.Button("保存 RAG 参数", variant="primary")
            reset_rag_btn = gr.Button("重置为默认", variant="secondary")
        rag_status = gr.Textbox(label="状态", interactive=False)

        save_rag_btn.click(
            save_rag_config,
            inputs=[top_k_input, top_n_input, chunk_size_input, chunk_overlap_input],
            outputs=[rag_status],
        )

        reset_rag_btn.click(
            lambda: (5, 3, 500, 100),
            outputs=[top_k_input, top_n_input, chunk_size_input, chunk_overlap_input],
        )

    return rag_column


def create_settings_ui_new():
    """新的系统设置界面"""
    with gr.Column() as settings_column:
        gr.Markdown("# 系统设置")

        with gr.Tabs():
            with gr.TabItem("模型配置"):
                create_llm_settings_ui()
            with gr.TabItem("模型列表"):
                create_model_list_ui()
            with gr.TabItem("向量模型"):
                create_embedding_settings_ui()
            with gr.TabItem("RAG 检索"):
                create_rag_settings_ui()
            with gr.TabItem("重排序 (Reranker)"):
                create_reranker_settings_ui()

    return settings_column
