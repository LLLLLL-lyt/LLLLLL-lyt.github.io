from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.models.schemas import ProviderInfo
from app.config import settings
from loguru import logger

router = APIRouter()

@router.get("/providers", response_model=List[ProviderInfo])
async def list_providers():
    providers = []
    for provider_id, provider_config in settings.LLM_PROVIDERS.items():
        providers.append(ProviderInfo(
            id=provider_id,
            name=provider_config["name"],
            api_key_url=provider_config["api_key_url"],
            doc_url=provider_config["doc_url"],
            models_url=provider_config["models_url"],
            models=provider_config["models"]
        ))
    return providers

@router.get("/providers/{provider_id}", response_model=ProviderInfo)
async def get_provider(provider_id: str):
    if provider_id not in settings.LLM_PROVIDERS:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    provider_config = settings.LLM_PROVIDERS[provider_id]
    return ProviderInfo(
        id=provider_id,
        name=provider_config["name"],
        api_key_url=provider_config["api_key_url"],
        doc_url=provider_config["doc_url"],
        models_url=provider_config["models_url"],
        models=provider_config["models"]
    )

@router.get("/providers/{provider_id}/models")
async def list_models(provider_id: str) -> Dict[str, Any]:
    if provider_id not in settings.LLM_PROVIDERS:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    provider_config = settings.LLM_PROVIDERS[provider_id]
    return {
        "provider": provider_id,
        "models": provider_config["models"]
    }
