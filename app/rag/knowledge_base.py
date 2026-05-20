"""
知识库管理模块 - 统一管理知识库配置
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from loguru import logger

# 知识库存储文件路径 - 使用绝对路径
BASE_DIR = Path(__file__).parent.parent.parent  # app/rag/knowledge_base.py -> app/rag -> app -> project_root
KB_CONFIG_FILE = BASE_DIR / "data" / "knowledge_bases.json"


class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self):
        self._knowledge_bases: Dict[str, Dict[str, Any]] = {}
        self._load_from_file()
    
    def _load_from_file(self):
        """从文件加载知识库配置"""
        if KB_CONFIG_FILE.exists():
            try:
                with open(KB_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 转换日期字符串为 datetime 对象
                    for kb_id, kb in data.items():
                        if "created_at" in kb and isinstance(kb["created_at"], str):
                            kb["created_at"] = datetime.fromisoformat(kb["created_at"])
                        if "documents" in kb:
                            for doc in kb["documents"]:
                                if "created_at" in doc and isinstance(doc["created_at"], str):
                                    doc["created_at"] = datetime.fromisoformat(doc["created_at"])
                    self._knowledge_bases = data
                logger.info(f"加载了 {len(self._knowledge_bases)} 个知识库")
            except Exception as e:
                logger.error(f"加载知识库配置失败: {e}")
                self._knowledge_bases = {}
        else:
            self._knowledge_bases = {}
    
    def _save_to_file(self):
        """保存知识库配置到文件"""
        try:
            # 确保目录存在
            KB_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"保存知识库配置到: {KB_CONFIG_FILE}")
            
            # 转换 datetime 为字符串
            data = {}
            for kb_id, kb in self._knowledge_bases.items():
                kb_copy = kb.copy()
                if "created_at" in kb_copy and isinstance(kb_copy["created_at"], datetime):
                    kb_copy["created_at"] = kb_copy["created_at"].isoformat()
                if "documents" in kb_copy:
                    kb_copy["documents"] = []
                    for doc in kb["documents"]:
                        doc_copy = doc.copy()
                        if "created_at" in doc_copy and isinstance(doc_copy["created_at"], datetime):
                            doc_copy["created_at"] = doc_copy["created_at"].isoformat()
                        kb_copy["documents"].append(doc_copy)
                data[kb_id] = kb_copy
            
            with open(KB_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存 {len(data)} 个知识库配置")
        except Exception as e:
            logger.error(f"保存知识库配置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_kb(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """获取知识库配置"""
        return self._knowledge_bases.get(kb_id)
    
    def get_all_kbs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有知识库"""
        return self._knowledge_bases.copy()
    
    def get_kb_embedding_model(self, kb_id: str) -> Optional[str]:
        """获取知识库的 Embedding 模型 ID"""
        kb = self._knowledge_bases.get(kb_id)
        if kb:
            return kb.get("embedding_model")
        return None
    
    def add_kb(self, kb_id: str, kb_config: Dict[str, Any]):
        """添加知识库"""
        self._knowledge_bases[kb_id] = kb_config
        self._save_to_file()
    
    def update_kb(self, kb_id: str, updates: Dict[str, Any]):
        """更新知识库配置"""
        if kb_id in self._knowledge_bases:
            self._knowledge_bases[kb_id].update(updates)
            self._save_to_file()
    
    def delete_kb(self, kb_id: str):
        """删除知识库"""
        if kb_id in self._knowledge_bases:
            del self._knowledge_bases[kb_id]
            self._save_to_file()
    
    def add_document(self, kb_id: str, doc_info: Dict[str, Any]):
        """添加文档到知识库"""
        if kb_id in self._knowledge_bases:
            if "documents" not in self._knowledge_bases[kb_id]:
                self._knowledge_bases[kb_id]["documents"] = []
            self._knowledge_bases[kb_id]["documents"].append(doc_info)
            self._save_to_file()


# 全局知识库管理器实例
kb_manager = KnowledgeBaseManager()
