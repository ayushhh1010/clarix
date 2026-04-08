"""
Pydantic schemas for API request / response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, EmailStr


# ── Auth ────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=128)


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    avatar_url: str | None = None
    oauth_provider: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Repository ──────────────────────────────────────────────

class RepoUploadRequest(BaseModel):
    url: str = Field(..., description="Git clone URL of the repository")


class RepoResponse(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    status: str
    file_count: int = 0
    chunk_count: int = 0
    ingestion_progress: int = 0
    ingestion_total_chunks: int = 0
    ingestion_cached_chunks: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RepoFileNode(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
    children: Optional[list["RepoFileNode"]] = None


# ── Chat ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    repo_id: str = Field(..., description="Repository ID to query against")
    message: str = Field(..., min_length=1, description="User message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID; omit to start new")


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    sources: list[dict] = Field(default_factory=list)


# ── Agent ───────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    repo_id: str
    task: str = Field(..., min_length=1, description="Natural language description of the agent task")
    conversation_id: Optional[str] = None


class AgentStepResponse(BaseModel):
    step: str
    agent: str
    content: str


class AgentRunResponse(BaseModel):
    conversation_id: str
    steps: list[AgentStepResponse]
    final_answer: str


# ── Conversation ────────────────────────────────────────────

class ConversationResponse(BaseModel):
    id: str
    repo_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata_json: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── File Content ────────────────────────────────────────────

class FileContentResponse(BaseModel):
    content: str
    language: str

    model_config = {"from_attributes": True}


# ── Pagination ──────────────────────────────────────────────

from typing import TypeVar, Generic

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    per_page: int
    has_more: bool

    model_config = {"from_attributes": True}
