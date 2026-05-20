"""
模型配置管理模块 - 参考 Bisheng 平台设计
核心设计：
1. LLMServer: 服务提供商配置（API Key、Base URL等）
2. LLMModel: 具体模型配置（模型名称、类型、参数等）
3. 支持多种模型类型：llm、embedding、speech、vision
4. 统一的配置管理和工厂方法
"""

from typing import Dict, List, Optional, Any, Type, Callable
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import datetime
import json
import os
from pathlib import Path
from loguru import logger

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent / "data"
CONFIG_FILE = CONFIG_DIR / "llm_config.json"

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent / "data"
CONFIG_FILE = CONFIG_DIR / "llm_config.json"


class ModelType(str, Enum):
    """模型类型枚举"""
    LLM = "llm"           # 大语言模型
    EMBEDDING = "embedding"  # 向量模型
    SPEECH_TTS = "speech_tts"  # 语音合成
    SPEECH_ASR = "speech_asr"  # 语音识别
    VISION = "vision"     # 视觉模型
    RERANK = "rerank"     # 重排序模型


class ProviderType(str, Enum):
    """提供商类型枚举 - 对应不同的API格式"""
    OPENAI = "openai"           # OpenAI 标准格式
    AZURE_OPENAI = "azure_openai"  # Azure OpenAI
    QWEN = "qwen"               # 阿里通义千问
    VOLCENGINE = "volcengine"   # 火山引擎
    TENCENT = "tencent"         # 腾讯混元
    QIANFAN = "qianfan"         # 百度千帆
    SPARK = "spark"             # 讯飞星火
    MINIMAX = "minimax"         # MiniMax
    MOONSHOT = "moonshot"       # 月之暗面
    ZHIPU = "zhipu"             # 智谱AI
    DEEPSEEK = "deepseek"       # DeepSeek
    SILICON = "silicon"         # SiliconFlow


class ModelStatus(int, Enum):
    """模型状态"""
    NORMAL = 0      # 正常
    ABNORMAL = 1    # 异常
    UNKNOWN = 2     # 未知（未测试）


class LLMModel(BaseModel):
    """
    模型配置 - 对应 Bisheng 的 LLMModel
    每个模型属于一个服务提供商
    """
    model_config = ConfigDict(protected_namespaces=())
    
    # 基础信息
    model_id: str = Field(..., description="模型唯一标识，如 gpt-4, qwen-plus")
    model_name: str = Field(..., description="模型显示名称")
    model_type: ModelType = Field(default=ModelType.LLM, description="模型类型")
    
    # 状态
    is_enabled: bool = Field(default=True, description="是否启用")
    status: ModelStatus = Field(default=ModelStatus.UNKNOWN, description="模型状态")
    status_msg: Optional[str] = Field(default=None, description="状态信息")
    
    # 模型参数（调用时传入）
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    temperature: Optional[float] = Field(default=0.7, description="温度参数")
    top_p: Optional[float] = Field(default=None, description="top_p参数")
    
    # 扩展配置
    config: Dict[str, Any] = Field(default_factory=dict, description="额外配置参数")
    
    # 元数据
    description: Optional[str] = Field(default=None, description="模型描述")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class LLMServer(BaseModel):
    """
    服务提供商配置 - 对应 Bisheng 的 LLMServer
    一个提供商下可以有多个模型
    """
    model_config = ConfigDict(protected_namespaces=())
    
    # 基础信息
    provider_id: str = Field(..., description="提供商ID，如 openai, qwen")
    provider_name: str = Field(..., description="提供商显示名称")
    provider_type: ProviderType = Field(..., description="提供商类型，决定API格式")
    
    # API 配置
    api_key: Optional[str] = Field(default=None, description="API Key")
    api_base: Optional[str] = Field(default=None, description="API Base URL")
    api_version: Optional[str] = Field(default=None, description="API版本（Azure等需要）")
    
    # 状态
    is_enabled: bool = Field(default=True, description="是否启用")
    
    # 限流配置
    rate_limit: Optional[int] = Field(default=None, description="每分钟最大请求数")
    daily_limit: Optional[int] = Field(default=None, description="每日最大请求数")
    
    # 链接信息（前端展示用）
    api_key_url: Optional[str] = Field(default=None, description="获取API Key的链接")
    doc_url: Optional[str] = Field(default=None, description="文档链接")
    models_url: Optional[str] = Field(default=None, description="模型列表链接")
    
    # 模型列表
    models: List[LLMModel] = Field(default_factory=list, description="该提供商下的模型列表")
    
    # 扩展配置
    config: Dict[str, Any] = Field(default_factory=dict, description="额外配置")
    
    # 元数据
    description: Optional[str] = Field(default=None, description="提供商描述")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def get_enabled_models(self, model_type: Optional[ModelType] = None) -> List[LLMModel]:
        """获取启用的模型列表"""
        models = [m for m in self.models if m.is_enabled]
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models
    
    def get_model(self, model_id: str) -> Optional[LLMModel]:
        """根据ID获取模型"""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None
    
    def add_model(self, model: LLMModel) -> bool:
        """添加模型"""
        if self.get_model(model.model_id):
            return False
        self.models.append(model)
        self.updated_at = datetime.now()
        return True
    
    def remove_model(self, model_id: str) -> bool:
        """移除模型"""
        for i, model in enumerate(self.models):
            if model.model_id == model_id:
                self.models.pop(i)
                self.updated_at = datetime.now()
                return True
        return False
    
    def update_model(self, model_id: str, **kwargs) -> bool:
        """更新模型配置"""
        model = self.get_model(model_id)
        if not model:
            return False
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
        model.updated_at = datetime.now()
        self.updated_at = datetime.now()
        return True


class LLMConfigManager(BaseModel):
    """
    全局配置管理器
    管理所有服务提供商和模型配置
    """
    model_config = ConfigDict(protected_namespaces=())
    
    # 服务提供商列表
    servers: Dict[str, LLMServer] = Field(default_factory=dict, description="服务提供商配置")
    
    # 默认配置
    default_llm_provider: Optional[str] = Field(default=None, description="默认LLM提供商")
    default_llm_model: Optional[str] = Field(default=None, description="默认LLM模型")
    default_embedding_provider: Optional[str] = Field(default=None, description="默认Embedding提供商")
    default_embedding_model: Optional[str] = Field(default=None, description="默认Embedding模型")
    
    # 版本
    version: str = Field(default="1.0.0")
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def get_server(self, provider_id: str) -> Optional[LLMServer]:
        """获取服务提供商"""
        return self.servers.get(provider_id)
    
    def add_server(self, server: LLMServer) -> bool:
        """添加服务提供商"""
        if server.provider_id in self.servers:
            return False
        self.servers[server.provider_id] = server
        self.updated_at = datetime.now()
        return True
    
    def remove_server(self, provider_id: str) -> bool:
        """移除服务提供商"""
        if provider_id not in self.servers:
            return False
        del self.servers[provider_id]
        self.updated_at = datetime.now()
        return True
    
    def update_server(self, provider_id: str, **kwargs) -> bool:
        """更新服务提供商配置"""
        server = self.servers.get(provider_id)
        if not server:
            return False
        for key, value in kwargs.items():
            if hasattr(server, key):
                setattr(server, key, value)
        server.updated_at = datetime.now()
        self.updated_at = datetime.now()
        return True
    
    def get_all_servers(self, enabled_only: bool = False) -> List[LLMServer]:
        """获取所有服务提供商"""
        servers = list(self.servers.values())
        if enabled_only:
            servers = [s for s in servers if s.is_enabled]
        return servers
    
    def get_all_models(self, model_type: Optional[ModelType] = None, enabled_only: bool = False) -> List[tuple[str, LLMModel]]:
        """
        获取所有模型
        返回: [(provider_id, model), ...]
        """
        result = []
        for provider_id, server in self.servers.items():
            if enabled_only and not server.is_enabled:
                continue
            for model in server.models:
                if enabled_only and not model.is_enabled:
                    continue
                if model_type and model.model_type != model_type:
                    continue
                result.append((provider_id, model))
        return result
    
    def get_model(self, provider_id: str, model_id: str) -> Optional[LLMModel]:
        """获取指定模型"""
        server = self.servers.get(provider_id)
        if not server:
            return None
        return server.get_model(model_id)
    
    def find_model(self, model_id: str) -> Optional[tuple[str, LLMModel]]:
        """在所有提供商中查找模型"""
        for provider_id, server in self.servers.items():
            model = server.get_model(model_id)
            if model:
                return (provider_id, model)
        return None
    
    def set_default_llm(self, provider_id: str, model_id: str):
        """设置默认LLM"""
        self.default_llm_provider = provider_id
        self.default_llm_model = model_id
        self.updated_at = datetime.now()
    
    def set_default_embedding(self, provider_id: str, model_id: str):
        """设置默认Embedding"""
        self.default_embedding_provider = provider_id
        self.default_embedding_model = model_id
        self.updated_at = datetime.now()
    
    def get_default_llm(self) -> Optional[tuple[str, str]]:
        """获取默认LLM配置 (provider_id, model_id)"""
        if self.default_llm_provider and self.default_llm_model:
            return (self.default_llm_provider, self.default_llm_model)
        return None
    
    def get_default_embedding(self) -> Optional[tuple[str, str]]:
        """获取默认Embedding配置 (provider_id, model_id)"""
        if self.default_embedding_provider and self.default_embedding_model:
            return (self.default_embedding_provider, self.default_embedding_model)
        return None
    
    def save_to_file(self, filepath: Optional[str] = None):
        """保存配置到文件"""
        filepath = filepath or str(CONFIG_FILE)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: Optional[str] = None) -> "LLMConfigManager":
        """从文件加载配置"""
        filepath = filepath or str(CONFIG_FILE)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 转换字符串类型的 model_type 为枚举
                for server_data in data.get('servers', {}).values():
                    for model_data in server_data.get('models', []):
                        if isinstance(model_data.get('model_type'), str):
                            try:
                                model_data['model_type'] = ModelType(model_data['model_type'])
                            except ValueError:
                                model_data['model_type'] = ModelType.LLM
                    if isinstance(server_data.get('provider_type'), str):
                        try:
                            server_data['provider_type'] = ProviderType(server_data['provider_type'])
                        except ValueError:
                            server_data['provider_type'] = ProviderType.OPENAI
                return cls(**data)
        return cls()


# 预定义的提供商模板 - 只保留基本信息，模型需要用户手动添加
DEFAULT_SERVER_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "openai": {
        "provider_id": "openai",
        "provider_name": "OpenAI",
        "provider_type": ProviderType.OPENAI,
        "api_base": "https://api.openai.com/v1",
        "api_key_url": "https://platform.openai.com/api-keys",
        "doc_url": "https://platform.openai.com/docs",
        "models_url": "https://platform.openai.com/docs/models",
        "models": []
    },
    "qwen": {
        "provider_id": "qwen",
        "provider_name": "阿里百炼 (Qwen)",
        "provider_type": ProviderType.QWEN,
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_url": "https://bailian.console.aliyun.com/#/api-key",
        "doc_url": "https://help.aliyun.com/zh/dashscope/",
        "models_url": "https://dashscope.console.aliyun.com/model",
        "models": []
    },
    "volcengine": {
        "provider_id": "volcengine",
        "provider_name": "火山引擎 (Volcengine)",
        "provider_type": ProviderType.VOLCENGINE,
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_url": "https://console.volcengine.com/ark",
        "doc_url": "https://www.volcengine.com/docs/82379",
        "models_url": "https://console.volcengine.com/ark/region:ark+cn-beijing/model",
        "models": []
    },
    "tencent": {
        "provider_id": "tencent",
        "provider_name": "腾讯混元 (Tencent)",
        "provider_type": ProviderType.TENCENT,
        "api_base": "https://hunyuan.tencentcloudapi.com",
        "api_key_url": "https://console.cloud.tencent.com/cam/capi",
        "doc_url": "https://cloud.tencent.com/document/product/1729",
        "models_url": "https://cloud.tencent.com/document/product/1729/97732",
        "models": []
    },
    "qianfan": {
        "provider_id": "qianfan",
        "provider_name": "百度千帆 (Qianfan)",
        "provider_type": ProviderType.QIANFAN,
        "api_base": "https://qianfan.baidubce.com/v2",
        "api_key_url": "https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application",
        "doc_url": "https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html",
        "models_url": "https://console.bce.baidu.com/qianfan/ais/console/onlineService",
        "models": []
    },
    "spark": {
        "provider_id": "spark",
        "provider_name": "讯飞星火 (Spark)",
        "provider_type": ProviderType.SPARK,
        "api_base": "wss://spark-api.xf-yun.com",
        "api_key_url": "https://console.xfyun.cn/services/bm3",
        "doc_url": "https://www.xfyun.cn/doc/spark/Web.html",
        "models_url": "https://console.xfyun.cn/services/bm3",
        "models": []
    },
    "minimax": {
        "provider_id": "minimax",
        "provider_name": "MiniMax",
        "provider_type": ProviderType.MINIMAX,
        "api_base": "https://api.minimax.chat/v1",
        "api_key_url": "https://platform.minimaxi.com/user-center/basic-information/interface-key",
        "doc_url": "https://platform.minimaxi.com/document/guides",
        "models_url": "https://platform.minimaxi.com/document/models",
        "models": []
    },
    "moonshot": {
        "provider_id": "moonshot",
        "provider_name": "月之暗面 (Moonshot)",
        "provider_type": ProviderType.MOONSHOT,
        "api_base": "https://api.moonshot.cn/v1",
        "api_key_url": "https://platform.moonshot.cn/console/api-keys",
        "doc_url": "https://platform.moonshot.cn/docs/intro",
        "models_url": "https://platform.moonshot.cn/docs/intro#%E6%A8%A1%E5%9E%8B%E8%AF%B4%E6%98%8E",
        "models": []
    },
    "zhipu": {
        "provider_id": "zhipu",
        "provider_name": "智谱AI (Zhipu)",
        "provider_type": ProviderType.ZHIPU,
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_url": "https://open.bigmodel.cn/usercenter/apikeys",
        "doc_url": "https://open.bigmodel.cn/dev/howuse/glm-4",
        "models_url": "https://open.bigmodel.cn/modelcenter/square",
        "models": []
    },
    "deepseek": {
        "provider_id": "deepseek",
        "provider_name": "DeepSeek",
        "provider_type": ProviderType.DEEPSEEK,
        "api_base": "https://api.deepseek.com",
        "api_key_url": "https://platform.deepseek.com/api_keys",
        "doc_url": "https://platform.deepseek.com/docs",
        "models_url": "https://platform.deepseek.com/docs/models",
        "models": []
    },
    "silicon": {
        "provider_id": "silicon",
        "provider_name": "SiliconFlow",
        "provider_type": ProviderType.SILICON,
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key_url": "https://cloud.siliconflow.cn/account/ak",
        "doc_url": "https://docs.siliconflow.cn/",
        "models_url": "https://cloud.siliconflow.cn/models",
        "models": []
    },
}


def create_config_manager() -> LLMConfigManager:
    """
    创建配置管理器
    如果配置文件存在则加载，否则使用默认模板创建
    同时兼容旧的配置文件
    """
    manager = LLMConfigManager.load_from_file()
    
    # 如果没有配置，使用默认模板初始化
    if not manager.servers:
        for template in DEFAULT_SERVER_TEMPLATES.values():
            server = LLMServer(**template)
            manager.add_server(server)
        manager.save_to_file()
    
    # 尝试从旧配置迁移
    _migrate_from_old_config(manager)
    
    return manager


def _migrate_from_old_config(manager: LLMConfigManager):
    """从旧的配置文件迁移配置"""
    import json
    
    # 旧配置文件路径
    old_config_paths = [
        Path(__file__).parent.parent / "guapi_config.json",
        Path(__file__).parent.parent.parent / "guapi_config.json",
        Path("guapi_config.json"),
    ]
    
    for old_config_path in old_config_paths:
        if old_config_path.exists():
            try:
                with open(old_config_path, 'r', encoding='utf-8') as f:
                    old_config = json.load(f)
                
                logger.info(f"从旧配置迁移: {old_config_path}")
                
                # 迁移火山引擎配置
                if old_config.get('volcengine_api_key'):
                    server = manager.get_server('volcengine')
                    if server:
                        manager.update_server(
                            'volcengine',
                            api_key=old_config.get('volcengine_api_key'),
                            api_base=old_config.get('volcengine_api_base', 'https://ark.cn-beijing.volces.com/api/v3')
                        )
                        # 添加模型
                        if old_config.get('volcengine_model'):
                            model_id = old_config['volcengine_model']
                            if not server.get_model(model_id):
                                server.add_model(LLMModel(
                                    model_id=model_id,
                                    model_name=model_id,
                                    model_type=ModelType.LLM,
                                    is_enabled=True
                                ))
                        manager.save_to_file()
                        logger.info("火山引擎配置迁移成功")
                
                # 可以添加其他提供商的迁移逻辑
                
                break  # 成功迁移后退出
            except Exception as e:
                logger.error(f"迁移旧配置失败: {e}")


# 全局配置管理器实例
config_manager = create_config_manager()
