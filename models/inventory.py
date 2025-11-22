from __future__ import annotations

from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from models.gen_response import Link


class InventoryBase(BaseModel):
    product_id: UUID = Field(
        ...,
        description="Reference to the Product ID.",
        json_schema_extra={"example": "11111111-1111-4111-8111-111111111111"},
    )
    quantity: int = Field(
        ...,
        description="Current stock quantity (unlimited capacity for online store).",
        ge=0,
        json_schema_extra={"example": 150},
    )
    warehouse_location: Optional[str] = Field(
        None,
        description="Physical location in warehouse or fulfillment center (optional for drop-shipped items).",
        max_length=100,
        json_schema_extra={"example": "A-12-05"},
    )
    reorder_level: Optional[int] = Field(
        None,
        description="Minimum quantity before reordering is needed.",
        ge=0,
        json_schema_extra={"example": 20},
    )
    reorder_quantity: Optional[int] = Field(
        None,
        description="Quantity to reorder when stock falls below reorder level.",
        ge=0,
        json_schema_extra={"example": 100},
    )
    reserved_quantity: int = Field(
        default=0,
        description="Quantity reserved for pending orders or items in shopping carts (not available for sale).",
        ge=0,
        json_schema_extra={"example": 10},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_id": "11111111-1111-4111-8111-111111111111",
                    "quantity": 150,
                    "warehouse_location": "A-12-05",
                    "reorder_level": 20,
                    "reorder_quantity": 100,
                    "reserved_quantity": 10,
                }
            ]
        }
    }


class InventoryCreate(InventoryBase):
    """Creation payload for an Inventory record."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_id": "22222222-2222-4222-8222-222222222222",
                    "quantity": 200,
                    "warehouse_location": "B-03-12",
                    "reorder_level": 30,
                    "reorder_quantity": 150,
                    "reserved_quantity": 0,
                },
                {
                    "product_id": "33333333-3333-4333-8333-333333333333",
                    "quantity": 5000,
                    "warehouse_location": None,  # Drop-shipped item
                    "reorder_level": 100,
                    "reorder_quantity": 1000,
                    "reserved_quantity": 0,
                }
            ]
        }
    }


class InventoryReplace(BaseModel):
    """Full replacement for an Inventory record; all fields required."""
    product_id: UUID = Field(
        ...,
        description="Reference to the Product ID.",
        json_schema_extra={"example": "11111111-1111-4111-8111-111111111111"},
    )
    quantity: int = Field(
        ...,
        description="Current stock quantity (unlimited capacity for online store).",
        ge=0,
        json_schema_extra={"example": 150},
    )
    warehouse_location: Optional[str] = Field(
        None,
        description="Physical location in warehouse or fulfillment center (optional for drop-shipped items).",
        max_length=100,
        json_schema_extra={"example": "A-12-05"},
    )
    reorder_level: Optional[int] = Field(
        None,
        description="Minimum quantity before reordering is needed.",
        ge=0,
        json_schema_extra={"example": 20},
    )
    reorder_quantity: Optional[int] = Field(
        None,
        description="Quantity to reorder when stock falls below reorder level.",
        ge=0,
        json_schema_extra={"example": 100},
    )
    reserved_quantity: int = Field(
        default=0,
        description="Quantity reserved for pending orders or items in shopping carts (not available for sale).",
        ge=0,
        json_schema_extra={"example": 10},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_id": "11111111-1111-4111-8111-111111111111",
                    "quantity": 150,
                    "warehouse_location": "A-12-05",
                    "reorder_level": 20,
                    "reorder_quantity": 100,
                    "reserved_quantity": 10
                }
            ]
        }
    }


class InventoryUpdate(BaseModel):
    """Partial update for Inventory; supply only fields to change."""
    product_id: Optional[UUID] = Field(
        None,
        description="Reference to the Product ID.",
        json_schema_extra={"example": "11111111-1111-4111-8111-111111111111"},
    )
    quantity: Optional[int] = Field(
        None,
        ge=0,
        json_schema_extra={"example": 175},
    )
    warehouse_location: Optional[str] = Field(
        None,
        max_length=100,
        json_schema_extra={"example": "A-15-08"},
    )
    reorder_level: Optional[int] = Field(
        None,
        ge=0,
        json_schema_extra={"example": 25},
    )
    reorder_quantity: Optional[int] = Field(
        None,
        ge=0,
        json_schema_extra={"example": 120},
    )
    reserved_quantity: Optional[int] = Field(
        None,
        ge=0,
        json_schema_extra={"example": 15},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"quantity": 180},
                {"warehouse_location": "C-01-03"},
                {"quantity": 90, "reserved_quantity": 20},
            ]
        }
    }


class InventoryRead(InventoryBase):
    """Server representation returned to clients."""
    id: UUID = Field(
        default_factory=uuid4,
        description="Server-generated Inventory record ID.",
        json_schema_extra={"example": "33333333-3333-4333-8333-333333333333"},
    )
    available_quantity: int = Field(
        ...,
        description="Calculated available quantity (quantity - reserved_quantity). This is what customers can actually purchase.",
        json_schema_extra={"example": 140},
    )
    needs_reorder: bool = Field(
        ...,
        description="True if current quantity is at or below reorder level.",
        json_schema_extra={"example": False},
    )
    last_restocked_at: Optional[datetime] = Field(
        None,
        description="Last time inventory was restocked (UTC).",
        json_schema_extra={"example": "2025-09-28T14:30:00Z"},
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp (UTC).",
        json_schema_extra={"example": "2025-09-30T10:20:30Z"},
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp (UTC).",
        json_schema_extra={"example": "2025-09-30T12:00:00Z"},
    )
    links: List[Link] = Field(
        ...,
        description="HATEOAS links for inventory",
        json_schema_extra={
            "examples": [
                {
                    "rel": "self",
                    "href": "https://api.example.com/inventory/33333333-3333-4333-8333-333333333333"
                },
                {
                    "rel": "product",
                    "href": "https://api.example.com/products/11111111-1111-4111-8111-111111111111"
                }
            ]
        }
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "33333333-3333-4333-8333-333333333333",
                    "product_id": "11111111-1111-4111-8111-111111111111",
                    "quantity": 150,
                    "warehouse_location": "A-12-05",
                    "reorder_level": 20,
                    "reorder_quantity": 100,
                    "reserved_quantity": 10,
                    "available_quantity": 140,
                    "needs_reorder": False,
                    "last_restocked_at": "2025-09-28T14:30:00Z",
                    "created_at": "2025-09-30T10:20:30Z",
                    "updated_at": "2025-09-30T12:00:00Z",
                    "links": [
                        {
                            "rel": "self",
                            "href": "https://api.example.com/inventory/33333333-3333-4333-8333-333333333333"
                        },
                        {
                            "rel": "product",
                            "href": "https://api.example.com/products/11111111-1111-4111-8111-111111111111"
                        }
                    ]
                }
            ]
        }
    }


class InventoryAdjustment(BaseModel):
    """Payload for adjusting inventory quantity (add/remove stock)."""
    adjustment: int = Field(
        ...,
        description="Quantity to add (positive) or remove (negative) from inventory. No upper limit for restocks.",
        json_schema_extra={"example": 50},
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for the adjustment (restock, sale, damage, return, etc.).",
        max_length=200,
        json_schema_extra={"example": "Restock from supplier"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"adjustment": 100, "reason": "Restock from supplier"},
                {"adjustment": -5, "reason": "Damaged items removed"},
                {"adjustment": 10, "reason": "Customer return processed"},
                {"adjustment": 1000, "reason": "Bulk order from manufacturer"},
            ]
        }
    }

class InventoryResponse(BaseModel):
    # allows for a message to be included when an inventory is created
    message: str = "New product created"
    product: InventoryRead
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "New product created",
                    "product": {
                        "id": "33333333-3333-4333-8333-333333333333",
                        "product_id": "11111111-1111-4111-8111-111111111111",
                        "quantity": 150,
                        "warehouse_location": "A-12-05",
                        "reorder_level": 20,
                        "reorder_quantity": 100,
                        "reserved_quantity": 10,
                        "available_quantity": 140,
                        "needs_reorder": False,
                        "last_restocked_at": "2025-09-28T14:30:00Z",
                        "created_at": "2025-09-30T10:20:30Z",
                        "updated_at": "2025-09-30T12:00:00Z",
                        "links": [
                            {
                                "rel": "self",
                                "href": "https://api.example.com/inventory/33333333-3333-4333-8333-333333333333"
                            },
                            {
                                "rel": "product",
                                "href": "https://api.example.com/products/11111111-1111-4111-8111-111111111111"
                            }
                        ]
                    },
                }
            ]
        }
    }