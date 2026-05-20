import sys
sys.path.insert(0, r'e:\xiangmu\gp')

print("=" * 60)
print("Testing new LLM config system...")
print("=" * 60)

try:
    print("\n[1] Testing config module import...")
    from app.llms.config import (
        LLMConfigManager, LLMServer, LLMModel,
        ModelType, ProviderType, config_manager,
        DEFAULT_SERVER_TEMPLATES
    )
    print("✅ Config module imported successfully!")
    print(f"   - Providers: {list(config_manager.servers.keys())}")
    
    print("\n[2] Testing factory module import...")
    from app.llms.factory import (
        LLMFactory, EmbeddingFactory,
        get_available_llms, get_available_embeddings
    )
    print("✅ Factory module imported successfully!")
    
    print("\n[3] Testing providers module import...")
    from app.llms.providers import (
        OpenAILLM, QwenLLM, get_llm_provider, get_llm_from_config
    )
    print("✅ Providers module imported successfully!")
    
    print("\n[4] Testing main llms module import...")
    from app.llms import config_manager, ModelType, ProviderType
    print("✅ Main llms module imported successfully!")
    
    print("\n[5] Testing config data...")
    print(f"   - Total servers: {len(config_manager.servers)}")
    for pid, server in config_manager.servers.items():
        print(f"   - {pid}: {server.provider_name} ({len(server.models)} models)")
    
    print("\n[6] Testing get_available_llms...")
    llms = get_available_llms()
    print(f"   - Available LLMs: {len(llms)}")
    for llm in llms[:3]:
        print(f"     * {llm['full_name']}")
    
    print("\n[7] Testing get_available_embeddings...")
    embeddings = get_available_embeddings()
    print(f"   - Available Embeddings: {len(embeddings)}")
    for emb in embeddings[:3]:
        print(f"     * {emb['full_name']}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
