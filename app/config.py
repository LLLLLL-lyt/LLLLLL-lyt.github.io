from pydantic_settings import BaseSettings
from typing import Dict, Any, List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "瓜皮智能聊天助手"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    SECRET_KEY: str = "guapi-chat-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    DEBUG: bool = True
    
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
    VECTOR_STORE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_stores")
    DATABASE_URL: str = "sqlite:///./data/guapi.db"
    
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
    
    # Reranker 配置
    RERANKER_MODE: str = "disabled"  # 运行模式: disabled(禁用), online(在线API)
    RERANKER_ONLINE_PROVIDER: str = "cohere"  # 在线API提供商: cohere, jina
    
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    TOP_K: int = 5      # 检索返回数量
    TOP_N: int = 3      # Rerank 后保留数量
    
    LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
        "openai": {
            "name": "OpenAI",
            "api_key_url": "https://platform.openai.com/api-keys",
            "doc_url": "https://platform.openai.com/docs",
            "models_url": "https://platform.openai.com/docs/models",
            "api_base": "https://api.openai.com/v1",
            "models": [
                "gpt-4", "gpt-4-turbo-preview", "gpt-4-1106-preview",
                "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"
            ]
        },
        "qwen": {
            "name": "阿里百炼 (Qwen)",
            "api_key_url": "https://bailian.console.aliyun.com/#/api-key",
            "doc_url": "https://help.aliyun.com/zh/dashscope/",
            "models_url": "https://dashscope.console.aliyun.com/model",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": [
                "qwen-plus", "qwen-turbo", "qwen-max", "qwen-max-longcontext",
                "qwen-vl-plus", "qwen-vl-max"
            ]
        },
        "volcengine": {
            "name": "火山引擎 (Volcengine)",
            "api_key_url": "https://console.volcengine.com/ark",
            "doc_url": "https://www.volcengine.com/docs/82379",
            "models_url": "https://console.volcengine.com/ark/region:ark+cn-beijing/model",
            "api_base": "https://ark.cn-beijing.volces.com/api/v3",
            "models": [
                "ep-20241203100915-47j7z", "ep-20241203101021-8k5q5",
                "doubao-pro-32k", "doubao-pro-128k", "doubao-lite-32k"
            ]
        },
        "tencent": {
            "name": "腾讯混元 (Tencent Hunyuan)",
            "api_key_url": "https://console.cloud.tencent.com/cam/capi",
            "doc_url": "https://cloud.tencent.com/document/product/1729",
            "models_url": "https://cloud.tencent.com/document/product/1729/97732",
            "api_base": "https://hunyuan.tencentcloudapi.com",
            "models": [
                "hunyuan-lite", "hunyuan-standard", "hunyuan-pro", "hunyuan-turbo"
            ]
        },
        "qianfan": {
            "name": "百度千帆 (Qianfan)",
            "api_key_url": "https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application",
            "doc_url": "https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html",
            "models_url": "https://console.bce.baidu.com/qianfan/ais/console/onlineService",
            "api_base": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
            "models": [
                "ERNIE-4.0-8K", "ERNIE-3.5-8K", "ERNIE-Speed-8K",
                "ERNIE-Lite-8K", "ERNIE-Tiny-8K"
            ]
        },
        "spark": {
            "name": "讯飞星火 (Spark)",
            "api_key_url": "https://console.xfyun.cn/services/bm3",
            "doc_url": "https://www.xfyun.cn/doc/spark/Web.html",
            "models_url": "https://console.xfyun.cn/services/bm3",
            "api_base": "wss://spark-api.xf-yun.com",
            "models": [
                "spark-v4.0", "spark-v3.5", "spark-v3.0", "spark-v2.0", "spark-v1.5"
            ]
        },
        "minimax": {
            "name": "Minimax",
            "api_key_url": "https://platform.minimaxi.com/user-center/basic-information/interface-key",
            "doc_url": "https://platform.minimaxi.com/document/guides",
            "models_url": "https://platform.minimaxi.com/document/models",
            "api_base": "https://api.minimax.chat/v1",
            "models": [
                "abab6.5s-chat", "abab6-chat", "abab5.5s-chat"
            ]
        },
        "moonshot": {
            "name": "月之暗面 (Moonshot)",
            "api_key_url": "https://platform.moonshot.cn/console/api-keys",
            "doc_url": "https://platform.moonshot.cn/docs",
            "models_url": "https://platform.moonshot.cn/docs/intro#%E6%A8%A1%E5%9E%8B%E5%88%97%E4%BB%8B",
            "api_base": "https://api.moonshot.cn/v1",
            "models": [
                "moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"
            ]
        },
        "zhipu": {
            "name": "智谱AI (Zhipu)",
            "api_key_url": "https://open.bigmodel.cn/usercenter/apikeys",
            "doc_url": "https://open.bigmodel.cn/dev/api",
            "models_url": "https://open.bigmodel.cn/modelcenter",
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "models": [
                "glm-4", "glm-4v", "glm-3-turbo", "codegeex-4"
            ]
        },
        "silicon": {
            "name": "SiliconFlow",
            "api_key_url": "https://cloud.siliconflow.cn/account/ak",
            "doc_url": "https://docs.siliconflow.cn/",
            "models_url": "https://cloud.siliconflow.cn/models",
            "api_base": "https://api.siliconflow.cn/v1",
            "models": [
                "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
                "deepseek-ai/DeepSeek-V3", "THUDM/glm-4-9b-chat"
            ]
        },
        "deepseek": {
            "name": "DeepSeek",
            "api_key_url": "https://platform.deepseek.com/api_keys",
            "doc_url": "https://platform.deepseek.com/docs",
            "models_url": "https://platform.deepseek.com/docs/models",
            "api_base": "https://api.deepseek.com",
            "models": [
                "deepseek-chat", "deepseek-coder"
            ]
        }
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
