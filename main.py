from __future__ import annotations

import os
import socket
from datetime import datetime
from decimal import Decimal

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Path

from models.product import ProductCreate, ProductRead, ProductUpdate
from models.inventory import (
    InventoryCreate,
    InventoryRead,
    InventoryUpdate,
    InventoryAdjustment,
)
from models.health import Health

port = int(os.environ.get("FASTAPIPORT", 8000))

# -----------------------------------------------------------------------------
# Fake in-memory "databases"
# -----------------------------------------------------------------------------
products: Dict[UUID, ProductRead] = {}
inventories: Dict[UUID, InventoryRead] = {}

app = FastAPI(
    title="E-commerce Inventory Management API",
    description="FastAPI microservice for managing products and inventory for an online store",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Health endpoints
# -----------------------------------------------------------------------------


def make_health(echo: Optional[str], path_echo: Optional[str] = None) -> Health:
    return Health(
        status=200,
        status_message="OK",
        timestamp=datetime.utcnow().isoformat() + "Z",
        ip_address=socket.gethostbyname(socket.gethostname()),
        echo=echo,
        path_echo=path_echo,
    )


@app.get("/health", response_model=Health)
def get_health_no_path(echo: str | None = Query(None, description="Optional echo string")):
    return make_health(echo=echo, path_echo=None)


@app.get("/health/{path_echo}", response_model=Health)
def get_health_with_path(
        path_echo: str = Path(..., description="Required echo in the URL path"),
        echo: str | None = Query(None, description="Optional echo string"),
):
    return make_health(echo=echo, path_echo=path_echo)


# -----------------------------------------------------------------------------
# Product endpoints
# -----------------------------------------------------------------------------


@app.post("/products", response_model=ProductRead, status_code=201)
def create_product(product: ProductCreate):
    """Create a new product."""
    # Check if SKU already exists
    for p in products.values():
        if p.sku == product.sku:
            raise HTTPException(status_code=400, detail=f"Product with SKU '{product.sku}' already exists")

    new_product = ProductRead(**product.model_dump())
    products[new_product.id] = new_product
    return new_product


@app.get("/products", response_model=List[ProductRead])
def list_products(
        sku: Optional[str] = Query(None, description="Filter by SKU"),
        name: Optional[str] = Query(None, description="Filter by name (contains)"),
        category: Optional[str] = Query(None, description="Filter by category"),
        brand: Optional[str] = Query(None, description="Filter by brand"),
        is_active: Optional[bool] = Query(None, description="Filter by active status"),
        min_price: Optional[Decimal] = Query(None, description="Minimum price"),
        max_price: Optional[Decimal] = Query(None, description="Maximum price"),
):
    """List all products with optional filters."""
    results = list(products.values())

    if sku:
        results = [p for p in results if p.sku == sku]
    if name:
        results = [p for p in results if name.lower() in p.name.lower()]
    if category:
        results = [p for p in results if p.category and category.lower() in p.category.lower()]
    if brand:
        results = [p for p in results if p.brand and brand.lower() in p.brand.lower()]
    if is_active is not None:
        results = [p for p in results if p.is_active == is_active]
    if min_price is not None:
        results = [p for p in results if p.price >= min_price]
    if max_price is not None:
        results = [p for p in results if p.price <= max_price]

    return results


@app.get("/products/{product_id}", response_model=ProductRead)
def get_product(product_id: UUID = Path(..., description="Product ID")):
    """Get a specific product by ID."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    return products[product_id]


@app.patch("/products/{product_id}", response_model=ProductRead)
def update_product(
        product_id: UUID = Path(..., description="Product ID"),
        product_update: ProductUpdate = None,
):
    """Update a product (partial update)."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    existing = products[product_id]
    update_data = product_update.model_dump(exclude_unset=True)

    # Check if SKU is being changed and if new SKU already exists
    if "sku" in update_data and update_data["sku"] != existing.sku:
        for p in products.values():
            if p.sku == update_data["sku"] and p.id != product_id:
                raise HTTPException(status_code=400, detail=f"Product with SKU '{update_data['sku']}' already exists")

    # Update fields
    for field, value in update_data.items():
        setattr(existing, field, value)

    existing.updated_at = datetime.utcnow()
    products[product_id] = existing
    return existing


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: UUID = Path(..., description="Product ID")):
    """Delete a product."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    # Check if product has inventory
    for inv in inventories.values():
        if inv.product_id == product_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete product with existing inventory. Remove inventory first.",
            )

    del products[product_id]
    return None


# -----------------------------------------------------------------------------
# Inventory endpoints
# -----------------------------------------------------------------------------


@app.post("/inventory", response_model=InventoryRead, status_code=201)
def create_inventory(inventory: InventoryCreate):
    """Create a new inventory record for a product."""
    # Check if product exists
    if inventory.product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    # Check if inventory already exists for this product
    for inv in inventories.values():
        if inv.product_id == inventory.product_id:
            raise HTTPException(
                status_code=400,
                detail="Inventory record already exists for this product",
            )

    # Create inventory record
    available = inventory.quantity - inventory.reserved_quantity
    needs_reorder = (
        inventory.quantity <= inventory.reorder_level
        if inventory.reorder_level is not None
        else False
    )

    new_inventory = InventoryRead(
        **inventory.model_dump(),
        available_quantity=available,
        needs_reorder=needs_reorder,
        last_restocked_at=datetime.utcnow(),
    )
    inventories[new_inventory.id] = new_inventory
    return new_inventory


@app.get("/inventory", response_model=List[InventoryRead])
def list_inventory(
        product_id: Optional[UUID] = Query(None, description="Filter by product ID"),
        warehouse_location: Optional[str] = Query(None, description="Filter by warehouse location"),
        needs_reorder: Optional[bool] = Query(None, description="Filter items needing reorder"),
        low_stock: Optional[bool] = Query(None, description="Show items with quantity < 10"),
):
    """List all inventory records with optional filters."""
    results = list(inventories.values())

    if product_id:
        results = [inv for inv in results if inv.product_id == product_id]
    if warehouse_location:
        results = [inv for inv in results if
                   inv.warehouse_location and warehouse_location.lower() in inv.warehouse_location.lower()]
    if needs_reorder is not None:
        results = [inv for inv in results if inv.needs_reorder == needs_reorder]
    if low_stock:
        results = [inv for inv in results if inv.quantity < 10]

    return results


@app.get("/inventory/{inventory_id}", response_model=InventoryRead)
def get_inventory(inventory_id: UUID = Path(..., description="Inventory ID")):
    """Get a specific inventory record by ID."""
    if inventory_id not in inventories:
        raise HTTPException(status_code=404, detail="Inventory record not found")
    return inventories[inventory_id]


@app.get("/inventory/product/{product_id}", response_model=InventoryRead)
def get_inventory_by_product(product_id: UUID = Path(..., description="Product ID")):
    """Get inventory record for a specific product."""
    for inv in inventories.values():
        if inv.product_id == product_id:
            return inv
    raise HTTPException(status_code=404, detail="Inventory record not found for this product")


@app.patch("/inventory/{inventory_id}", response_model=InventoryRead)
def update_inventory(
        inventory_id: UUID = Path(..., description="Inventory ID"),
        inventory_update: InventoryUpdate = None,
):
    """Update an inventory record (partial update)."""
    if inventory_id not in inventories:
        raise HTTPException(status_code=404, detail="Inventory record not found")

    existing = inventories[inventory_id]
    update_data = inventory_update.model_dump(exclude_unset=True)

    # Update fields
    for field, value in update_data.items():
        setattr(existing, field, value)

    # Recalculate computed fields
    existing.available_quantity = existing.quantity - existing.reserved_quantity
    existing.needs_reorder = (
        existing.quantity <= existing.reorder_level
        if existing.reorder_level is not None
        else False
    )
    existing.updated_at = datetime.utcnow()

    inventories[inventory_id] = existing
    return existing


@app.post("/inventory/{inventory_id}/adjust", response_model=InventoryRead)
def adjust_inventory(
        inventory_id: UUID = Path(..., description="Inventory ID"),
        adjustment: InventoryAdjustment = None,
):
    """Adjust inventory quantity (add or remove stock)."""
    if inventory_id not in inventories:
        raise HTTPException(status_code=404, detail="Inventory record not found")

    existing = inventories[inventory_id]
    new_quantity = existing.quantity + adjustment.adjustment

    if new_quantity < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient stock. Current: {existing.quantity}, Adjustment: {adjustment.adjustment}",
        )

    existing.quantity = new_quantity
    existing.available_quantity = existing.quantity - existing.reserved_quantity
    existing.needs_reorder = (
        existing.quantity <= existing.reorder_level
        if existing.reorder_level is not None
        else False
    )

    # Update last_restocked_at only for positive adjustments
    if adjustment.adjustment > 0:
        existing.last_restocked_at = datetime.utcnow()

    existing.updated_at = datetime.utcnow()
    inventories[inventory_id] = existing
    return existing


@app.delete("/inventory/{inventory_id}", status_code=204)
def delete_inventory(inventory_id: UUID = Path(..., description="Inventory ID")):
    """Delete an inventory record."""
    if inventory_id not in inventories:
        raise HTTPException(status_code=404, detail="Inventory record not found")

    del inventories[inventory_id]
    return None


# -----------------------------------------------------------------------------
# Inventory statistics endpoints
# -----------------------------------------------------------------------------


@app.get("/inventory/stats/summary")
def get_inventory_summary():
    """Get summary statistics for all inventory."""
    total_products = len(inventories)
    total_quantity = sum(inv.quantity for inv in inventories.values())
    total_reserved = sum(inv.reserved_quantity for inv in inventories.values())
    total_available = sum(inv.available_quantity for inv in inventories.values())
    items_needing_reorder = sum(1 for inv in inventories.values() if inv.needs_reorder)
    low_stock_items = sum(1 for inv in inventories.values() if inv.quantity < 10)

    return {
        "total_products": total_products,
        "total_quantity": total_quantity,
        "total_reserved": total_reserved,
        "total_available": total_available,
        "items_needing_reorder": items_needing_reorder,
        "low_stock_items": low_stock_items,
    }


# -----------------------------------------------------------------------------
# Run the app
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)