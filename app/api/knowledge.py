from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional
import os
import shutil
from datetime import datetime
from app.models.schemas import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate, KnowledgeBaseInfo,
    FileUploadResponse, DocumentInfo
)
from app.config import settings
from app.utils.helpers import get_file_md5, generate_unique_id, sanitize_filename, format_size
from app.rag.splitter import splitter
from app.rag.vector_store import vector_store_manager
from loguru import logger

router = APIRouter()

knowledge_bases: dict = {}

@router.post("/knowledge-bases", response_model=KnowledgeBaseInfo)
async def create_knowledge_base(data: KnowledgeBaseCreate):
    kb_id = generate_unique_id()
    knowledge_bases[kb_id] = {
        "id": kb_id,
        "name": data.name,
        "description": data.description,
        "created_at": datetime.now(),
        "documents": []
    }
    return KnowledgeBaseInfo(
        id=kb_id,
        name=data.name,
        description=data.description,
        created_at=knowledge_bases[kb_id]["created_at"],
        document_count=0,
        size=0
    )

@router.get("/knowledge-bases", response_model=List[KnowledgeBaseInfo])
async def list_knowledge_bases():
    result = []
    for kb_id, kb in knowledge_bases.items():
        doc_count = len(kb["documents"])
        total_size = sum(doc.get("size", 0) for doc in kb["documents"])
        result.append(KnowledgeBaseInfo(
            id=kb_id,
            name=kb["name"],
            description=kb["description"],
            created_at=kb["created_at"],
            document_count=doc_count,
            size=total_size
        ))
    return result

@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseInfo)
async def get_knowledge_base(kb_id: str):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    kb = knowledge_bases[kb_id]
    doc_count = len(kb["documents"])
    total_size = sum(doc.get("size", 0) for doc in kb["documents"])
    
    return KnowledgeBaseInfo(
        id=kb_id,
        name=kb["name"],
        description=kb["description"],
        created_at=kb["created_at"],
        document_count=doc_count,
        size=total_size
    )

@router.put("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseInfo)
async def update_knowledge_base(kb_id: str, data: KnowledgeBaseUpdate):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    if data.name:
        knowledge_bases[kb_id]["name"] = data.name
    if data.description is not None:
        knowledge_bases[kb_id]["description"] = data.description
    
    kb = knowledge_bases[kb_id]
    doc_count = len(kb["documents"])
    total_size = sum(doc.get("size", 0) for doc in kb["documents"])
    
    return KnowledgeBaseInfo(
        id=kb_id,
        name=kb["name"],
        description=kb["description"],
        created_at=kb["created_at"],
        document_count=doc_count,
        size=total_size
    )

@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    vector_store_manager.delete_knowledge_base(kb_id)
    del knowledge_bases[kb_id]
    
    return {"message": "Knowledge base deleted successfully"}

@router.post("/knowledge-bases/{kb_id}/upload", response_model=FileUploadResponse)
async def upload_file(kb_id: str, file: UploadFile = File(...)):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    content = await file.read()
    file_md5 = get_file_md5(content)
    safe_filename = sanitize_filename(file.filename or "unknown")
    file_id = f"{file_md5}_{safe_filename}"
    
    save_path = os.path.join(settings.UPLOAD_DIR, file_id)
    
    with open(save_path, "wb") as f:
        f.write(content)
    
    file_size = len(content)
    
    try:
        documents = splitter.load_and_split(save_path)
        for doc in documents:
            doc.metadata["source"] = safe_filename
            doc.metadata["file_id"] = file_id
            doc.metadata["kb_id"] = kb_id
        
        vector_store_manager.add_documents(kb_id, documents)
        
        doc_info = {
            "id": generate_unique_id(),
            "filename": safe_filename,
            "file_id": file_id,
            "size": file_size,
            "chunk_count": len(documents),
            "created_at": datetime.now()
        }
        knowledge_bases[kb_id]["documents"].append(doc_info)
        
        logger.info(f"File {safe_filename} processed: {len(documents)} chunks")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=safe_filename,
            size=file_size,
            status="success"
        )
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@router.get("/knowledge-bases/{kb_id}/documents", response_model=List[DocumentInfo])
async def list_documents(kb_id: str):
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    docs = []
    for doc in knowledge_bases[kb_id]["documents"]:
        docs.append(DocumentInfo(
            id=doc["id"],
            filename=doc["filename"],
            file_id=doc["file_id"],
            knowledge_base_id=kb_id,
            created_at=doc["created_at"],
            chunk_count=doc.get("chunk_count", 0)
        ))
    return docs
