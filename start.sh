#!/bin/bash

echo "========================================"
echo "    瓜皮智能聊天助手 - 启动脚本"
echo "========================================"
echo ""

echo "[1/2] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.9+"
    exit 1
fi
echo "Python环境检查通过"
echo ""

echo "[2/2] 安装依赖并启动服务..."
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""
echo "========================================"
echo "    🎉 服务启动成功！"
echo "========================================"
echo ""
echo "📡 FastAPI API: http://localhost:8000"
echo "📚 API Docs:   http://localhost:8000/docs"
echo "🎨 Gradio UI:  http://localhost:7860"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python3 -m app.main
