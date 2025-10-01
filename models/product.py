from __future__ import annotations

from typing import Optional, Annotated
from uuid import UUID, uuid4
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, StringConstraints

# SKU pattern: alphanumeric with hyphens (e.g., PROD-12345)
SKUType = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9\-]{3,20}$")]


class ProductBase(BaseModel):
    sku: SKUType = Field(
        ...,
        description="Stock Keeping Unit - unique product identifier (uppercase alphanumeric with hyphens).",
        json_schema_extra={"example": "PROD-12345"},
    )
    name: str = Field(
        ...,
        description="Product name.",
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "Wireless Mouse"},
    )
    description: Optional[str] = Field(
        None,
        description="Detailed product description.",
        max_length=1000,
        json_schema_extra={"example": "Ergonomic wireless mouse with USB receiver"},
    )
    price: Decimal = Field(
        ...,
        description="Product price in USD.",
        ge=0,
        decimal_places=2,
        json_schema_extra={"example": "29.99"},
    )
    category: Optional[str] = Field(
        None,
        description="Product category.",
        max_length=100,
        json_schema_extra={"example": "Electronics"},
    )
    brand: Optional[str] = Field(
        None,
        description="Product brand or manufacturer.",
        max_length=100,
        json_schema_extra={"example": "TechCorp"},
    )
    is_active: bool = Field(
        default=True,
        description="Whether the product is currently active/available.",
        json_schema_extra={"example": True},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sku": "PROD-12345",
                    "name": "Wireless Mouse",
                    "description": "Ergonomic wireless mouse with USB receiver",
                    "price": "29.99",
                    "category": "Electronics",
                    "brand": "TechCorp",
                    "is_active": True,
                }
            ]
        }
    }


class ProductCreate(ProductBase):
    """Creation payload for a Product."""
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sku": "KEYB-5678",
                    "name": "Mechanical Keyboard",
                    "description": "RGB backlit mechanical keyboard with blue switches",
                    "price": "89.99",
                    "category": "Electronics",
                    "brand": "TechCorp",
                    "is_active": True,
                }
            ]
        }
    }


class ProductUpdate(BaseModel):
    """Partial update for a Product; supply only fields to change."""
    sku: Optional[SKUType] = Field(
        None,
        description="Stock Keeping Unit.",
        json_schema_extra={"example": "PROD-54321"},
    )
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "Updated Product Name"},
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        json_schema_extra={"example": "Updated product description"},
    )
    price: Optional[Decimal] = Field(
        None,
        ge=0,
        decimal_places=2,
        json_schema_extra={"example": "39.99"},
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        json_schema_extra={"example": "Office Supplies"},
    )
    brand: Optional[str] = Field(
        None,
        max_length=100,
        json_schema_extra={"example": "NewBrand"},
    )
    is_active: Optional[bool] = Field(
        None,
        json_schema_extra={"example": False},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"price": "34.99"},
                {"name": "Premium Wireless Mouse", "price": "49.99"},
                {"is_active": False},
            ]
        }
    }


class ProductRead(ProductBase):
    """Server representation returned to clients."""
    id: UUID = Field(
        default_factory=uuid4,
        description="Server-generated Product ID.",
        json_schema_extra={"example": "11111111-1111-4111-8111-111111111111"},
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "sku": "PROD-12345",
                    "name": "Wireless Mouse",
                    "description": "Ergonomic wireless mouse with USB receiver",
                    "price": "29.99",
                    "category": "Electronics",
                    "brand": "TechCorp",
                    "is_active": True,
                    "created_at": "2025-09-30T10:20:30Z",
                    "updated_at": "2025-09-30T12:00:00Z",
                }
            ]
        }
    }