"""Async operations manager for tracking background tasks."""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum


class OperationStatus(str, Enum):
    """Status enum for async operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AsyncOperation:
    """Represents a single async operation."""
    operation_id: str
    status: OperationStatus
    message: str
    created_at: datetime
    updated_at: datetime
    progress_percent: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    task: Optional[asyncio.Task] = None


class AsyncOperationManager:
    """Manages async operations with in-memory storage."""
    
    def __init__(self):
        self._operations: Dict[str, AsyncOperation] = {}
        self._lock = asyncio.Lock()
    
    async def create_operation(
        self,
        message: str,
        coroutine: Coroutine
    ) -> str:
        """
        Create a new async operation and start executing the coroutine.
        
        Args:
            message: Human-readable message describing the operation
            coroutine: The coroutine to execute asynchronously
            
        Returns:
            operation_id: Unique identifier for tracking the operation
        """
        operation_id = f"op-{uuid.uuid4()}"
        now = datetime.utcnow()
        
        operation = AsyncOperation(
            operation_id=operation_id,
            status=OperationStatus.PENDING,
            message=message,
            created_at=now,
            updated_at=now,
            progress_percent=0
        )
        
        async with self._lock:
            self._operations[operation_id] = operation
        
        # Start the background task
        task = asyncio.create_task(self._execute_operation(operation_id, coroutine))
        operation.task = task
        
        return operation_id
    
    async def _execute_operation(self, operation_id: str, coroutine: Coroutine):
        """Execute the coroutine and update operation status."""
        operation = self._operations[operation_id]
        
        try:
            async with self._lock:
                operation.status = OperationStatus.IN_PROGRESS
                operation.updated_at = datetime.utcnow()
                operation.progress_percent = 10
                operation.message = f"{operation.message}... (in progress)"
            
            # Execute the coroutine
            result = await coroutine
            
            async with self._lock:
                operation.status = OperationStatus.COMPLETED
                operation.updated_at = datetime.utcnow()
                operation.progress_percent = 100
                operation.result = result
                operation.message = f"{operation.message} (completed)"
        
        except Exception as e:
            async with self._lock:
                operation.status = OperationStatus.FAILED
                operation.updated_at = datetime.utcnow()
                operation.error = {
                    "type": type(e).__name__,
                    "message": str(e)
                }
                operation.message = f"{operation.message} (failed)"
    
    async def get_operation(self, operation_id: str) -> Optional[AsyncOperation]:
        """Get the status of an operation."""
        async with self._lock:
            return self._operations.get(operation_id)
    
    async def update_progress(
        self,
        operation_id: str,
        progress_percent: int,
        message: Optional[str] = None
    ):
        """Update the progress of an ongoing operation."""
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation:
                operation.progress_percent = max(0, min(100, progress_percent))
                operation.updated_at = datetime.utcnow()
                if message:
                    operation.message = message


# Global instance
async_operation_manager = AsyncOperationManager()
