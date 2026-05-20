# 瓜皮智能聊天助手 GuaPi-Chat-Assistant

一个功能完整的开源RAG（检索增强生成）智能聊天助手，纯Python实现，一个命令即可启动！

## 🌟 功能特性

### 🤖 多模型支持
- **12家厂商集成**：OpenAI、阿里百炼、火山引擎、腾讯混元、百度千帆、讯飞星火、Minimax、月之暗面、智谱AI、SiliconFlow、DeepSeek、vLLM本地模型
- 统一API接口，无缝切换
- 支持流式输出和同步调用

### 📚 RAG知识库
- BGE嵌入模型 + Chroma向量库
- BM25+密集向量混合检索
- BGE重排序模型
- 多知识库管理
- 文件支持：PDF、DOCX、TXT、MD、XLSX等

### 🎨 Gradio界面
- **纯Python前端**，无需Node.js
- 现代化UI，支持Tab页切换
- 聊天界面、知识库管理、系统设置
- 参数调节滑块、文件上传

### 🔧 技术特性
- Python + FastAPI后端
- Gradio前端，一体化部署
- 模块化、插件化架构
- MD5加密文件名
- 同步invoke + 异步stream流式输出
- **一个命令即可启动**

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 8GB+ 内存（推荐16GB+）

### 方法一：一键启动脚本（推荐）

#### Windows
```bash
start.bat
```

#### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

### 方法二：手动启动

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动（一个命令搞定！）
python -m app.main
```

启动后访问：
- 🎨 **Gradio UI**: http://localhost:7860
- 📡 **FastAPI API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

### 方法三：Docker部署
```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 📖 使用说明

### 1. 配置API Key
1. 打开「⚙️ 设置」标签页
2. 选择模型提供商
3. 输入对应的API Key
4. 配置会自动保存到 `guapi_config.json`

### 2. 创建知识库
1. 打开「📚 知识库」标签页
2. 输入名称和描述，点击「创建知识库」
3. 选择知识库，点击「上传文件」
4. 选择文件（支持多选），点击「上传并处理」
5. 等待文件处理完成

### 3. 开始聊天
1. 打开「💬 聊天」标签页
2. 右侧选择模型提供商和模型
3. 调整参数（Temperature、Top P）
4. 勾选「使用 RAG 知识库」，选择知识库
5. 输入消息，按「发送」或回车

## 📁 项目结构

```
GuaPi-Chat-Assistant/
├── app/                          # 后端代码
│   ├── api/                      # API路由
│   │   ├── chat.py              # 聊天接口
│   │   ├── knowledge.py         # 知识库接口
│   │   └── models.py            # 模型接口
│   ├── core/                     # 核心模块
│   ├── llms/                     # LLM集成
│   │   ├── base.py              # 基类
│   │   └── providers.py         # 12家厂商实现
│   ├── rag/                      # RAG模块
│   │   ├── embeddings.py        # 嵌入模型
│   │   ├── splitter.py          # 文档切分
│   │   ├── vector_store.py      # 向量存储
│   │   ├── reranker.py          # 重排序
│   │   └── chain.py             # RAG链
│   ├── frontend/                 # Gradio前端
│   │   └── gradio_ui.py         # Gradio UI实现
│   ├── models/                   # 数据模型
│   ├── utils/                    # 工具函数
│   ├── config.py                 # 配置文件
│   └── main.py                   # 入口（同时启动FastAPI+Gradio）
├── data/                         # 数据目录
│   ├── uploads/                 # 上传文件
│   └── vector_stores/           # 向量库
├── docs/                         # 文档
├── requirements.txt              # Python依赖
├── Dockerfile                    # Docker镜像
├── docker-compose.yml            # Docker Compose
├── start.bat                     # Windows启动脚本
├── start.sh                      # Linux/Mac启动脚本
├── guapi_config.json             # 配置文件（自动生成）
└── README.md
```

## 🎯 支持的模型提供商

| 提供商 | 模型示例 | API Key获取 |
|--------|----------|-------------|
| OpenAI | gpt-4, gpt-3.5-turbo | [platform.openai.com](https://platform.openai.com/api-keys) |
| 阿里百炼 | qwen-plus, qwen-max | [bailian.console.aliyun.com](https://bailian.console.aliyun.com/#/api-key) |
| 火山引擎 | doubao-pro-32k | [console.volcengine.com](https://console.volcengine.com/ark) |
| 腾讯混元 | hunyuan-lite, hunyuan-pro | [console.cloud.tencent.com](https://console.cloud.tencent.com/cam/capi) |
| 百度千帆 | ERNIE-4.0-8K | [console.bce.baidu.com](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application) |
| 讯飞星火 | spark-v4.0, spark-v3.5 | [console.xfyun.cn](https://console.xfyun.cn/services/bm3) |
| Minimax | abab6.5s-chat | [platform.minimaxi.com](https://platform.minimaxi.com/user-center/basic-information/interface-key) |
| 月之暗面 | moonshot-v1-128k | [platform.moonshot.cn](https://platform.moonshot.cn/console/api-keys) |
| 智谱AI | glm-4, glm-3-turbo | [open.bigmodel.cn](https://open.bigmodel.cn/usercenter/apikeys) |
| SiliconFlow | Qwen2.5-72B-Instruct | [cloud.siliconflow.cn](https://cloud.siliconflow.cn/account/ak) |
| DeepSeek | deepseek-chat | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| vLLM本地 | Qwen2.5-7B-Instruct | 本地部署 |

## 🔧 配置说明

主要配置在 `app/config.py` 中：

```python
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
TOP_N = 3
```

## 📝 更新日志

### v2.0.0 (Gradio版本)
- ✅ 改用Gradio纯Python前端
- ✅ 一个命令同时启动FastAPI + Gradio
- ✅ 删除Vue3依赖，无需Node.js
- ✅ 更简单的部署流程

### v1.0.0
- ✅ 完整的RAG流程
- ✅ 12家模型提供商集成
- ✅ Vue3现代化界面
- ✅ 多知识库管理
- ✅ 混合检索 + 重排序
- ✅ 流式输出支持

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- 参考 [Bisheng](https://github.com/dataelement/bisheng) 开源项目架构
- 使用 [LangChain](https://github.com/langchain-ai/langchain) 框架
- 使用 [Gradio](https://github.com/gradio-app/gradio) UI框架
- 感谢所有开源贡献者

---

**瓜皮智能聊天助手** - 纯Python，一键启动！🎉
