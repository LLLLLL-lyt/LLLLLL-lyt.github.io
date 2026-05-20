import sys
print("=" * 60)
print("Testing imports...")
print("=" * 60)

steps = [
    ("FastAPI", "from fastapi import FastAPI"),
    ("Gradio", "import gradio as gr"),
    ("Settings", "from app.config import settings"),
    ("LangChain", "import langchain"),
    ("ChromaDB", "import chromadb"),
    ("SentenceTransformers", "import sentence_transformers"),
    ("Transformers", "import transformers"),
    ("PyTorch", "import torch"),
    ("Gradio UI", "from app.frontend.gradio_ui import create_main_ui"),
    ("Main App", "from app.main import app"),
]

for name, import_stmt in steps:
    try:
        print(f"\n[{name}] Testing...")
        exec(import_stmt)
        print(f"✅ {name} imported successfully!")
    except Exception as e:
        print(f"❌ {name} failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Import test complete!")
print("=" * 60)
