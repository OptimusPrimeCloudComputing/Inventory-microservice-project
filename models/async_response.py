"""Models for asynchronous operations and status polling."""

from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class OperationStatus(str, Enum):
    """Status enum for async operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncOperationResponse(BaseModel):
    """Response model for 202 Accepted operations."""
    operation_id: str = Field(
        ...,
        description="Unique identifier for the async operation",
        json_schema_extra={"example": "op-123e4567-e89b-12d3-a456-426614174000"}
    )
    status: OperationStatus = Field(
        default=OperationStatus.PENDING,
        description="Current status of the operation"
    )
    message: str = Field(
        ...,
        description="Human-readable message about the operation",
        json_schema_extra={"example": "Product update initiated"}
    )
    status_url: str = Field(
        ...,
        description="URL to poll for operation status",
        json_schema_extra={"example": "/operations/op-123e4567-e89b-12d3-a456-426614174000"}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "operation_id": "op-123e4567-e89b-12d3-a456-426614174000",
                    "status": "pending",
                    "message": "Product update initiated",
                    "status_url": "/operations/op-123e4567-e89b-12d3-a456-426614174000"
                }
            ]
        }
    }


class OperationStatusResponse(BaseModel):
    """Response model for polling operation status."""
    operation_id: str = Field(
        ...,
        description="Unique identifier for the async operation"
    )
    status: OperationStatus = Field(
        ...,
        description="Current status of the operation"
    )
    message: str = Field(
        ...,
        description="Human-readable status message"
    )
    progress_percent: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Estimated progress percentage (0-100)"
    )
    created_at: datetime = Field(
        ...,
        description="When the operation was initiated"
    )
    updated_at: datetime = Field(
        ...,
        description="When the operation status was last updated"
    )
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Result data if operation completed successfully"
    )
    error: Optional[Dict[str, Any]] = Field(
        None,
        description="Error details if operation failed"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "operation_id": "op-123e4567-e89b-12d3-a456-426614174000",
                    "status": "completed",
                    "message": "Product update completed successfully",
                    "progress_percent": 100,
                    "created_at": "2025-11-22T10:00:00Z",
                    "updated_at": "2025-11-22T10:00:05Z",
                    "result": {
                        "id": "11111111-1111-4111-8111-111111111111",
                        "sku": "PROD-12345",
                        "name": "Updated Product Name"
                    },
                    "error": None
                }
            ]
        }
    }
