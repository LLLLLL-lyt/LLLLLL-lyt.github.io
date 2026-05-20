"""
知识库管理界面 - 参考 Bisheng 设计
包含分块策略配置和内容预览功能
"""

import gradio as gr
import os
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
from app.llms import config_manager, ModelType
from app.rag.splitter import DocumentSplitter
from app.rag.knowledge_base import kb_manager
from app.utils.helpers import generate_unique_id, get_file_md5, sanitize_filename
from langchain_core.documents import Document

# 知识库存储 - 使用管理器
knowledge_bases: Dict[str, Dict[str, Any]] = kb_manager.get_all_kbs()

# 默认分块策略
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", " ", ""]


def get_embedding_choices() -> List[Tuple[str, str]]:
    """获取可用的向量模型选项"""
    config_manager.load_from_file()
    choices = []
    for provider_id, server in config_manager.servers.items():
        if not server.api_key:
            continue
        for model in server.models:
            if model.model_type == ModelType.EMBEDDING and model.is_enabled:
                full_id = f"{provider_id}/{model.model_id}"
                display = f"{server.provider_name} - {model.model_name}"
                choices.append((display, full_id))
    return choices if choices else [("请先配置Embedding模型", "")]


def refresh_kb_list_data() -> List[List]:
    """刷新知识库列表数据 - 从管理器重新加载"""
    data = []
    # 重新从管理器获取最新数据
    all_kbs = kb_manager.get_all_kbs()
    for kb_id, kb in all_kbs.items():
        data.append([
            kb_id,
            kb["name"],
            kb.get("description", ""),
            kb.get("embedding_model", "默认"),
            len(kb.get("documents", [])),
            kb["created_at"].strftime("%Y-%m-%d %H:%M"),
            kb.get("chunk_size", DEFAULT_CHUNK_SIZE),
            kb.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        ])
    return data


def refresh_kb_dropdown_data() -> List[Tuple[str, str]]:
    """刷新知识库下拉框数据 - 从管理器重新加载"""
    choices = [("请选择知识库", "")]
    # 重新从管理器获取最新数据
    all_kbs = kb_manager.get_all_kbs()
    for kb_id, kb in all_kbs.items():
        choices.append((kb["name"], kb_id))
    return choices


def get_initial_doc_choices(kb_id: str = None) -> List[Tuple[str, str]]:
    """获取初始文档下拉框选项 - 使用指定知识库的文档"""
    empty_choices = [("请选择文档", "")]
    all_kbs = kb_manager.get_all_kbs()
    if not all_kbs:
        return empty_choices
    
    # 使用传入的 kb_id 或第一个知识库
    if not kb_id:
        kb_id = list(all_kbs.keys())[0]
    
    kb = kb_manager.get_kb(kb_id)
    if not kb or not kb.get("documents"):
        return empty_choices
    
    doc_choices = list(empty_choices)
    seen_file_ids = set()
    for doc in kb.get("documents", []):
        file_id = doc.get("file_id", "")
        if file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        doc_choices.append((doc["filename"], file_id))
    
    return doc_choices


def get_initial_kb_id() -> str:
    """获取初始知识库 ID - 返回第一个知识库的 ID"""
    all_kbs = kb_manager.get_all_kbs()
    if all_kbs:
        return list(all_kbs.keys())[0]
    return ""


def get_initial_doc_table() -> list:
    """获取初始文档表格数据 - 使用第一个知识库的文档"""
    all_kbs = kb_manager.get_all_kbs()
    if not all_kbs:
        return []
    
    kb_id = list(all_kbs.keys())[0]
    kb = kb_manager.get_kb(kb_id)
    if not kb or not kb.get("documents"):
        return []
    
    doc_data = []
    for doc in kb.get("documents", []):
        doc_data.append([
            doc["filename"],
            doc["chunk_count"],
            doc["char_count"],
            doc["created_at"].strftime("%Y-%m-%d %H:%M")
        ])
    
    return doc_data


def preview_document_split(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    separator1: str,
    separator2: str,
    separator3: str,
    max_preview: int = 5
) -> Tuple[str, str]:
    """
    预览文档分块结果
    返回: (预览文本, 统计信息)
    """
    if not file_path or not os.path.exists(file_path):
        return "请先上传文件", ""
    
    try:
        # 构建分隔符列表
        separators = [s for s in [separator1, separator2, separator3] if s.strip()]
        if not separators:
            separators = DEFAULT_SEPARATORS
        
        # 创建临时 splitter
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # 加载文档
        documents = splitter.load_document(file_path)
        
        # 分块
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        chunks = text_splitter.split_documents(documents)
        
        # 生成预览
        preview_texts = []
        for i, chunk in enumerate(chunks[:max_preview]):
            preview_texts.append(f"=== 分块 {i+1} ===\n{chunk.page_content[:500]}{'...' if len(chunk.page_content) > 500 else ''}\n")
        
        if len(chunks) > max_preview:
            preview_texts.append(f"\n... 还有 {len(chunks) - max_preview} 个分块 ...")
        
        preview = "\n".join(preview_texts)
        stats = f"总字符数: {sum(len(d.page_content) for d in documents)} | 分块数: {len(chunks)} | 平均每块: {sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0} 字符"
        
        return preview, stats
    except Exception as e:
        return f"预览失败: {str(e)}", ""


def create_knowledge_base_ui():
    """知识库管理界面 - Bisheng 风格"""
    
    with gr.Column() as kb_column:
        gr.Markdown("# 📚 知识库管理")
        
        with gr.Tabs():
            # ========== 创建知识库标签页 ==========
            with gr.TabItem("➕ 创建知识库"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 基本信息")
                        kb_name = gr.Textbox(
                            label="知识库名称 *",
                            placeholder="请输入知识库名称"
                        )
                        kb_desc = gr.Textbox(
                            label="知识库描述",
                            placeholder="请输入知识库描述（可选）",
                            lines=2
                        )
                        
                        gr.Markdown("### 向量模型配置")
                        select_embedding = gr.Dropdown(
                            label="选择向量模型 *",
                            choices=[("请先刷新", "")],
                            value="",
                            interactive=True
                        )
                        refresh_embedding_btn = gr.Button("🔄 刷新向量模型列表", variant="secondary")
                        
                        gr.Markdown("### 分块策略配置")
                        with gr.Accordion("📐 分块参数设置", open=True):
                            chunk_size = gr.Slider(
                                label="分块大小 (Chunk Size)",
                                minimum=100,
                                maximum=4000,
                                step=100,
                                value=DEFAULT_CHUNK_SIZE,
                                info="每个分块的最大字符数"
                            )
                            chunk_overlap = gr.Slider(
                                label="分块重叠 (Chunk Overlap)",
                                minimum=0,
                                maximum=500,
                                step=50,
                                value=DEFAULT_CHUNK_OVERLAP,
                                info="相邻分块之间的重叠字符数"
                            )
                            
                            gr.Markdown("**分隔符设置** (按优先级排序)")
                            separator1 = gr.Textbox(
                                label="分隔符 1 (最高优先级)",
                                value="\\n\\n",
                                info="例如: 段落分隔符"
                            )
                            separator2 = gr.Textbox(
                                label="分隔符 2",
                                value="\\n",
                                info="例如: 换行符"
                            )
                            separator3 = gr.Textbox(
                                label="分隔符 3",
                                value="。",
                                info="例如: 句号"
                            )
                        
                        create_btn = gr.Button("✅ 创建知识库", variant="primary")
                        create_status = gr.Textbox(label="状态", interactive=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 分块效果预览")
                        gr.Markdown("上传文件并点击预览按钮查看分块效果")
                        
                        preview_file = gr.File(
                            label="选择要预览的文件",
                            file_types=[".pdf", ".txt", ".docx", ".md"]
                        )
                        preview_btn = gr.Button("👁️ 预览分块效果", variant="secondary")
                        preview_stats = gr.Textbox(label="统计信息", interactive=False)
                        preview_result = gr.Textbox(
                            label="分块预览 (前5个分块)",
                            lines=20,
                            interactive=False
                        )
            
            # ========== 知识库列表标签页 ==========
            with gr.TabItem("📋 知识库列表"):
                kb_list = gr.Dataframe(
                    headers=["ID", "名称", "描述", "向量模型", "文档数", "创建时间", "分块大小", "重叠"],
                    label="知识库列表",
                    interactive=True,
                    value=refresh_kb_list_data()
                )
                
                with gr.Row():
                    delete_kb_select = gr.Dropdown(
                        label="选择要删除的知识库",
                        choices=refresh_kb_dropdown_data(),
                        value="",
                        interactive=True,
                        scale=3
                    )
                    delete_btn = gr.Button("🗑️ 删除知识库", variant="stop", scale=1)
                
                delete_status = gr.Textbox(label="删除状态", interactive=False)
                refresh_list_btn = gr.Button("🔄 刷新列表", variant="secondary")
            
            # ========== 上传文档标签页 ==========
            with gr.TabItem("📤 上传文档"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 选择知识库")
                        select_kb = gr.Dropdown(
                            label="知识库",
                            choices=refresh_kb_dropdown_data(),
                            value="",
                            interactive=True
                        )
                        
                        gr.Markdown("### 上传文件")
                        upload_files = gr.File(
                            label="选择文件",
                            file_types=[".pdf", ".txt", ".docx", ".md"],
                            file_count="multiple"
                        )
                        overwrite_check = gr.Checkbox(
                            label="覆盖已有文档",
                            value=False,
                            info="如果勾选，同名文档将被替换"
                        )
                        upload_btn = gr.Button("📤 上传并处理", variant="primary")
                        upload_status = gr.Textbox(label="处理结果", interactive=False, lines=5)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 文档列表")
                        doc_list = gr.Dataframe(
                            headers=["文件名", "分块数", "字符数", "上传时间"],
                            label="已上传文档",
                            interactive=False
                        )
            
            # ========== 文档管理标签页 ==========
            with gr.TabItem("📄 文档管理"):
                # 初始化时获取第一个知识库的文档
                init_kb_id = get_initial_kb_id()
                init_doc_choices = get_initial_doc_choices(init_kb_id)
                init_doc_table = get_initial_doc_table()
                
                logger.info(f"文档管理初始化: kb_id={init_kb_id}, doc_choices={len(init_doc_choices)}, doc_table={len(init_doc_table)}")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 选择知识库")
                        manage_kb_select = gr.Dropdown(
                            label="知识库",
                            choices=refresh_kb_dropdown_data(),
                            value=init_kb_id if init_kb_id else "",
                            interactive=True
                        )
                        refresh_docs_btn = gr.Button("🔄 刷新文档列表", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 文档列表")
                        manage_doc_list = gr.Dataframe(
                            headers=["文件名", "分块数", "字符数", "上传时间"],
                            label="已上传文档",
                            interactive=False,
                            value=init_doc_table
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 删除文档")
                        delete_doc_select = gr.Dropdown(
                            label="选择要删除的文档",
                            choices=init_doc_choices,
                            value="",
                            interactive=True
                        )
                        delete_doc_btn = gr.Button("🗑️ 删除文档", variant="stop")
                        delete_doc_status = gr.Textbox(label="删除状态", interactive=False)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 文档预览")
                        preview_doc_select = gr.Dropdown(
                            label="选择要预览的文档",
                            choices=init_doc_choices,
                            value="",
                            interactive=True
                        )
                        preview_doc_btn = gr.Button("👁️ 预览文档", variant="secondary")
                        preview_doc_content = gr.Textbox(
                            label="文档内容预览",
                            lines=20,
                            interactive=False
                        )
        
        # ========== 事件绑定 ==========
        
        # 刷新向量模型列表
        def refresh_embedding_dropdown():
            choices = get_embedding_choices()
            return gr.update(choices=choices, value=choices[0][1] if choices else "")
        
        refresh_embedding_btn.click(refresh_embedding_dropdown, outputs=[select_embedding])
        
        # 预览分块效果
        preview_btn.click(
            preview_document_split,
            inputs=[preview_file, chunk_size, chunk_overlap, separator1, separator2, separator3],
            outputs=[preview_result, preview_stats]
        )
        
        # 创建知识库
        def create_kb(name, desc, embedding_model, size, overlap, sep1, sep2, sep3):
            if not name:
                return "❌ 请输入知识库名称", refresh_kb_list_data(), refresh_kb_dropdown_data()
            if not embedding_model:
                return "❌ 请选择向量模型", refresh_kb_list_data(), refresh_kb_dropdown_data()
            
            # 检查名称是否已存在
            all_kbs = kb_manager.get_all_kbs()
            for kb_id, kb in all_kbs.items():
                if kb.get("name") == name:
                    return f"❌ 知识库名称 '{name}' 已存在，请使用其他名称", refresh_kb_list_data(), refresh_kb_dropdown_data()
            
            kb_id = generate_unique_id()
            separators = [s for s in [sep1, sep2, sep3] if s.strip()]
            
            kb_config = {
                "id": kb_id,
                "name": name,
                "description": desc,
                "embedding_model": embedding_model,
                "chunk_size": size,
                "chunk_overlap": overlap,
                "separators": separators if separators else DEFAULT_SEPARATORS,
                "documents": [],
                "created_at": datetime.now()
            }
            
            # 保存到知识库管理器
            kb_manager.add_kb(kb_id, kb_config)
            # 更新本地引用
            knowledge_bases[kb_id] = kb_config
            
            return f"✅ 知识库 '{name}' 创建成功!", refresh_kb_list_data(), refresh_kb_dropdown_data()
        
        create_btn.click(
            create_kb,
            inputs=[kb_name, kb_desc, select_embedding, chunk_size, chunk_overlap, separator1, separator2, separator3],
            outputs=[create_status, kb_list, select_kb]
        )
        
        # 刷新知识库列表
        refresh_list_btn.click(refresh_kb_list_data, outputs=[kb_list])
        
        # 删除知识库
        def delete_kb(kb_id):
            # 处理 kb_id 可能是各种格式的情况
            print(f"[DEBUG] 原始 kb_id: {type(kb_id)} = {kb_id}")
            
            # 提取实际的 ID 值
            while isinstance(kb_id, (list, tuple, set)):
                if not kb_id:
                    kb_id = ""
                    break
                kb_id = kb_id[0]
            
            kb_id = str(kb_id).strip()
            print(f"[DEBUG] 转换后 kb_id: {kb_id}")
            
            if not kb_id:
                return "❌ 请选择要删除的知识库", refresh_kb_list_data(), refresh_kb_dropdown_data(), refresh_kb_dropdown_data()
            
            # 检查知识库是否存在
            kb = kb_manager.get_kb(kb_id)
            if not kb:
                print(f"[DEBUG] 知识库不存在: {kb_id}")
                return f"❌ 知识库不存在 (ID: {kb_id})", refresh_kb_list_data(), refresh_kb_dropdown_data(), refresh_kb_dropdown_data()
            
            try:
                # 删除知识库配置
                kb_manager.delete_kb(kb_id)
                
                # 更新本地引用
                if kb_id in knowledge_bases:
                    del knowledge_bases[kb_id]
                
                # 尝试删除向量存储目录
                vector_store_dir = Path(__file__).parent.parent.parent / "data" / "vector_stores" / kb_id
                if vector_store_dir.exists():
                    shutil.rmtree(vector_store_dir)
                    logger.info(f"已删除向量存储目录: {vector_store_dir}")
                
                return f"✅ 知识库 '{kb.get('name')}' 已删除", refresh_kb_list_data(), refresh_kb_dropdown_data(), refresh_kb_dropdown_data()
            except Exception as e:
                return f"❌ 删除失败: {str(e)}", refresh_kb_list_data(), refresh_kb_dropdown_data(), refresh_kb_dropdown_data()
        
        delete_btn.click(
            delete_kb,
            inputs=[delete_kb_select],
            outputs=[delete_status, kb_list, select_kb, delete_kb_select]
        )
        
        # 上传文档
        def upload_and_process(kb_id, files, overwrite=False):
            if not kb_id:
                return "❌ 请先选择知识库", None
            if not files:
                return "❌ 请选择文件", None
            
            # 处理 kb_id 可能是元组的情况
            if isinstance(kb_id, (list, tuple)):
                if len(kb_id) == 2 and isinstance(kb_id[1], str):
                    kb_id = kb_id[1]
                else:
                    kb_id = kb_id[0]
            
            # 从管理器获取最新知识库数据
            kb = kb_manager.get_kb(kb_id)
            if not kb:
                return "❌ 知识库不存在", None
            
            results = []
            all_docs = []
            
            # 获取知识库配置的 Embedding 模型
            embedding_model = kb.get("embedding_model", "")
            
            for file in files if isinstance(files, list) else [files]:
                try:
                    file_path = file.name
                    file_name = os.path.basename(file_path)
                    safe_filename = sanitize_filename(file_name)
                    
                    # 检查文档是否已存在
                    existing_docs = [d for d in kb.get("documents", []) if d.get("filename") == safe_filename]
                    if existing_docs and not overwrite:
                        results.append(f"⚠️ {safe_filename}: 文档已存在，请使用覆盖选项重新上传")
                        continue
                    
                    # 如果存在且要覆盖，先删除旧文档
                    if existing_docs and overwrite:
                        from app.rag.vector_store import vector_store_manager
                        for old_doc in existing_docs:
                            old_file_id = old_doc.get("file_id")
                            if old_file_id:
                                try:
                                    vector_store_manager.delete_documents(kb_id, old_file_id)
                                    kb["documents"] = [d for d in kb.get("documents", []) if d.get("file_id") != old_file_id]
                                    logger.info(f"已删除旧文档: {safe_filename}")
                                except Exception as e:
                                    logger.warning(f"删除旧文档失败: {e}")
                    
                    # 使用知识库的分块配置
                    splitter = DocumentSplitter(
                        chunk_size=kb.get("chunk_size", DEFAULT_CHUNK_SIZE),
                        chunk_overlap=kb.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                    )
                    
                    # 加载并分块
                    print(f"[DEBUG] 开始处理文件: {file_path}")
                    documents = splitter.load_and_split(file_path)
                    print(f"[DEBUG] 文档分块完成: {safe_filename}, 共 {len(documents)} 个分块")
                    
                    if not documents:
                        raise ValueError(f"文档 {safe_filename} 分块后为空，请检查文件内容")
                    
                    # 添加元数据
                    for doc in documents:
                        doc.metadata["kb_id"] = kb_id
                        doc.metadata["source"] = safe_filename
                        doc.metadata["upload_time"] = datetime.now().isoformat()
                    
                    # 添加到向量存储（使用知识库配置的 Embedding 模型）
                    from app.rag.vector_store import vector_store_manager
                    # 确保 embedding_model 不为空
                    if not embedding_model:
                        raise ValueError("知识库未配置 Embedding 模型，请重新创建知识库并选择向量模型")
                    
                    vector_store_manager.add_documents(
                        kb_id, 
                        documents, 
                        embedding_model_id=embedding_model
                    )
                    
                    # 保存文档信息
                    doc_info = {
                        "filename": safe_filename,
                        "file_id": get_file_md5(safe_filename.encode()),
                        "chunk_count": len(documents),
                        "char_count": sum(len(d.page_content) for d in documents),
                        "created_at": datetime.now()
                    }
                    kb["documents"].append(doc_info)
                    all_docs.extend(documents)
                    
                    # 保存到知识库管理器
                    kb_manager.update_kb(kb_id, {"documents": kb["documents"]})
                    
                    # 同时更新本地缓存
                    if kb_id in knowledge_bases:
                        knowledge_bases[kb_id]["documents"] = kb["documents"]
                    
                    if existing_docs and overwrite:
                        results.append(f"✅ {safe_filename}: 已覆盖，{len(documents)} 个分块")
                    else:
                        results.append(f"✅ {safe_filename}: {len(documents)} 个分块")
                except Exception as e:
                    results.append(f"❌ {file.name}: 处理失败 - {str(e)}")
            
            # 重新从管理器获取最新数据（因为 add_document 会保存到文件）
            kb = kb_manager.get_kb(kb_id)
            
            # 更新文档列表
            doc_data = []
            for doc in kb.get("documents", []) if kb else []:
                doc_data.append([
                    doc["filename"],
                    doc["chunk_count"],
                    doc["char_count"],
                    doc["created_at"].strftime("%Y-%m-%d %H:%M")
                ])
            
            return "\n".join(results), doc_data
        
        upload_btn.click(
            upload_and_process,
            inputs=[select_kb, upload_files, overwrite_check],
            outputs=[upload_status, doc_list]
        )
        
        # 选择知识库时更新文档列表
        def load_kb_docs(kb_id):
            # 处理 kb_id 可能是列表或元组的情况
            if isinstance(kb_id, (list, tuple)):
                if not kb_id:
                    return []
                # 如果是元组 (label, value)，取第二个值
                if len(kb_id) == 2 and isinstance(kb_id[1], str):
                    kb_id = kb_id[1]
                else:
                    kb_id = kb_id[0]
            
            if not kb_id:
                return []
            
            # 从管理器获取最新知识库数据
            kb = kb_manager.get_kb(kb_id)
            if not kb:
                return []
            doc_data = []
            for doc in kb.get("documents", []):
                doc_data.append([
                    doc["filename"],
                    doc["chunk_count"],
                    doc["char_count"],
                    doc["created_at"].strftime("%Y-%m-%d %H:%M")
                ])
            return doc_data
        
        select_kb.change(load_kb_docs, inputs=[select_kb], outputs=[doc_list])
        
        # ========== 文档管理功能 ==========
        
        # 刷新文档列表
        def refresh_manage_docs(kb_id):
            # 处理 kb_id 可能是列表或元组的情况
            if isinstance(kb_id, (list, tuple)):
                if len(kb_id) == 2 and isinstance(kb_id[1], str):
                    kb_id = kb_id[1]
                else:
                    kb_id = kb_id[0] if kb_id else ""
            
            # 使用统一的空选项格式
            empty_choices = [("请选择文档", "")]
            
            if not kb_id:
                return [], empty_choices, empty_choices
            
            kb = kb_manager.get_kb(kb_id)
            if not kb:
                return [], empty_choices, empty_choices
            
            doc_data = []
            doc_choices = list(empty_choices)  # 从空选项开始
            
            # 使用 file_id 去重
            seen_file_ids = set()
            for doc in kb.get("documents", []):
                file_id = doc.get("file_id", "")
                if file_id in seen_file_ids:
                    continue  # 跳过重复文档
                seen_file_ids.add(file_id)
                
                doc_data.append([
                    doc["filename"],
                    doc["chunk_count"],
                    doc["char_count"],
                    doc["created_at"].strftime("%Y-%m-%d %H:%M")
                ])
                doc_choices.append((doc["filename"], file_id))
            
            logger.info(f"刷新文档列表: kb_id={kb_id}, 文档数={len(doc_choices)-1}")
            
            return doc_data, doc_choices, doc_choices
        
        refresh_docs_btn.click(
            refresh_manage_docs,
            inputs=[manage_kb_select],
            outputs=[manage_doc_list, delete_doc_select, preview_doc_select]
        )
        
        # 管理知识库选择变化时刷新
        manage_kb_select.change(
            refresh_manage_docs,
            inputs=[manage_kb_select],
            outputs=[manage_doc_list, delete_doc_select, preview_doc_select]
        )
        
        # 点击知识库列表时进入文档管理
        def kb_list_click(evt: gr.SelectData):
            """点击知识库列表时，提取知识库ID并切换到文档管理"""
            empty_choices = [("请选择文档", "")]
            if not evt or not evt.value:
                return "", [], empty_choices, empty_choices
            
            # evt.value 是选中的单元格值，evt.index 是 (行, 列)
            # 我们需要获取该行的第一列（ID）
            row_index = evt.index[0] if evt.index else 0
            
            # 从知识库管理器获取所有知识库
            all_kbs = kb_manager.get_all_kbs()
            kb_ids = list(all_kbs.keys())
            
            if row_index >= len(kb_ids):
                return "", [], empty_choices, empty_choices
            
            kb_id = kb_ids[row_index]
            
            # 刷新文档列表
            return kb_id, *refresh_manage_docs(kb_id)
        
        kb_list.select(
            kb_list_click,
            inputs=None,
            outputs=[manage_kb_select, manage_doc_list, delete_doc_select, preview_doc_select]
        )
        
        # 删除文档
        def delete_document(kb_id, file_id):
            logger.info(f"删除文档 - 原始参数: kb_id={kb_id} (type={type(kb_id).__name__}), file_id={file_id} (type={type(file_id).__name__})")
            
            # 处理 kb_id - Gradio dropdown 返回的是选择的 value（即 kb_id 字符串）
            if isinstance(kb_id, (list, tuple)):
                kb_id = str(kb_id[1]) if len(kb_id) > 1 and kb_id[1] else str(kb_id[0]) if kb_id else ""
            elif kb_id:
                kb_id = str(kb_id)
            else:
                kb_id = ""
            
            # 处理 file_id - Gradio dropdown 返回的是选择的 value（即 file_id 字符串）
            if isinstance(file_id, (list, tuple)):
                file_id = str(file_id[1]) if len(file_id) > 1 and file_id[1] else str(file_id[0]) if file_id else ""
            elif file_id:
                file_id = str(file_id)
            else:
                file_id = ""
            
            logger.info(f"删除文档 - 处理后: kb_id={kb_id}, file_id={file_id}")
            
            empty_choices = [("请选择文档", "")]
            
            if not kb_id or not file_id:
                return "❌ 请选择知识库和文档", [], empty_choices, empty_choices
            
            kb = kb_manager.get_kb(kb_id)
            if not kb:
                return "知识库不存在", [], empty_choices, empty_choices
            
            # 查找文档
            doc_to_delete = None
            for doc in kb.get("documents", []):
                if doc["file_id"] == file_id:
                    doc_to_delete = doc
                    break
            
            if not doc_to_delete:
                return "❌ 文档不存在", [], empty_choices, empty_choices
            
            try:
                # 从向量存储中删除文档
                from app.rag.vector_store import vector_store_manager
                vector_store_manager.delete_documents(kb_id, file_id)
                
                # 从知识库配置中删除文档
                kb["documents"] = [d for d in kb.get("documents", []) if d["file_id"] != file_id]
                kb_manager.update_kb(kb_id, {"documents": kb["documents"]})
                
                # 更新本地缓存
                if kb_id in knowledge_bases:
                    knowledge_bases[kb_id]["documents"] = kb["documents"]
                
                # 刷新下拉框选项
                refreshed = refresh_manage_docs(kb_id)
                doc_data = refreshed[0]
                doc_choices = refreshed[1]
                
                return f"✅ 文档 '{doc_to_delete['filename']}' 已删除", doc_data, doc_choices, doc_choices
            except Exception as e:
                import traceback
                logger.error(f"删除文档失败: {e}\n{traceback.format_exc()}")
                return f"❌ 删除失败: {str(e)}", [], empty_choices, empty_choices
        
        delete_doc_btn.click(
            delete_document,
            inputs=[manage_kb_select, delete_doc_select],
            outputs=[delete_doc_status, manage_doc_list, delete_doc_select, preview_doc_select]
        )
        
        # 预览文档
        def preview_document(kb_id, file_id):
            # 处理 kb_id 可能是列表或元组的情况
            if isinstance(kb_id, (list, tuple)):
                if len(kb_id) == 2 and isinstance(kb_id[1], str):
                    kb_id = kb_id[1]
                else:
                    kb_id = kb_id[0] if kb_id else ""
            if isinstance(file_id, (list, tuple)):
                if len(file_id) == 2 and isinstance(file_id[1], str):
                    file_id = file_id[1]
                else:
                    file_id = file_id[0] if file_id else ""
            
            if not kb_id or not file_id:
                return "请选择知识库和文档"
            
            try:
                # 从向量存储中检索文档的所有分块
                from app.rag.vector_store import vector_store_manager
                from app.rag.embeddings import embedding_manager
                
                kb = kb_manager.get_kb(kb_id)
                if not kb:
                    return "❌ 知识库不存在"
                
                embedding_model = kb.get("embedding_model", "")
                if not embedding_model:
                    return "❌ 知识库未配置向量模型"
                
                # 获取向量存储
                vector_store = vector_store_manager.get_or_create_vector_store(kb_id, embedding_model)
                
                # 检索该文档的所有分块
                results = vector_store.get(where={"source": file_id})
                
                if not results or not results.get("documents"):
                    return "❌ 未找到文档内容"
                
                # 组合所有分块内容
                chunks = results["documents"]
                preview = f"=== 文档预览 ===\n文件名: {file_id}\n分块数: {len(chunks)}\n\n"
                for i, chunk in enumerate(chunks[:10]):  # 最多预览10个分块
                    preview += f"--- 分块 {i+1} ---\n{chunk[:500]}{'...' if len(chunk) > 500 else ''}\n\n"
                
                if len(chunks) > 10:
                    preview += f"\n... 还有 {len(chunks) - 10} 个分块 ..."
                
                return preview
            except Exception as e:
                import traceback
                return f"❌ 预览失败: {str(e)}\n{traceback.format_exc()[:500]}"
        
        preview_doc_btn.click(
            preview_document,
            inputs=[manage_kb_select, preview_doc_select],
            outputs=[preview_doc_content]
        )
    
    return kb_column
