"""主路由聚合"""

from fastapi import APIRouter
from src.api import documents
from src.api import chat

api_router = APIRouter()
api_router.include_router(documents.router)
api_router.include_router(chat.router)
