from __future__ import annotations

import os
import socket
from datetime import datetime
from decimal import Decimal

from typing import Dict, List, Optional
import uuid
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response, status, Request, Query, Path
from fastapi_pagination import add_pagination, paginate, set_page, set_params
from fastapi_pagination.limit_offset import LimitOffsetParams, LimitOffsetPage

from models.product import ProductCreate, ProductRead, ProductUpdate, ProductResponse
from models.inventory import (
    InventoryCreate,
    InventoryRead,
    InventoryUpdate,
    InventoryAdjustment,
    InventoryResponse,
)
from models.gen_response import PaginateResponse
from models.health import Health
from db import get_connection
import pymysql


def _to_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return v


def row_to_inventory_read(row: dict) -> InventoryRead:
    """
    Convert a SQL row from the `inventory` table (DictCursor) into InventoryRead.
    Assumes SELECT included computed columns:
      - available_quantity
      - needs_reorder
    """
    if not row:
        return None

    available = row.get("available_quantity")
    if available is None and row.get("quantity") is not None and row.get("reserved_quantity") is not None:
        available = _to_int(row["quantity"]) - \
            _to_int(row["reserved_quantity"])

    needs = row.get("needs_reorder")
    if needs is None:
        rl = row.get("reorder_level")
        needs = bool(rl is not None and _to_int(
            row["quantity"]) <= _to_int(rl))
    else:
        needs = bool(needs)

    return InventoryRead(
        id=UUID(str(row["inventory_id"])),
        product_id=UUID(str(row["product_id"])),
        quantity=_to_int(row["quantity"]),
        warehouse_location=row.get("warehouse_location"),
        reorder_level=_to_int(row.get("reorder_level")),
        reorder_quantity=_to_int(row.get("reorder_quantity")),
        reserved_quantity=_to_int(row.get("reserved_quantity", 0)),
        available_quantity=_to_int(available),
        needs_reorder=needs,
        last_restocked_at=row.get("last_restocked_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )

def row_to_product_read(row: dict) -> ProductRead:
    """
    Convert a SQL row from the `products` table (DictCursor) into ProductRead.
    """
    if not row:
        return None

    return ProductRead(
        id=UUID(str(row["product_id"])),
        sku=row.get("sku"),
        name=row.get("name"),
        description=row.get("description"),
        price=Decimal(row.get("price")) if row.get("price") is not None else None,
        category=row.get("category"),
        brand=row.get("brand"),
        is_active=bool(row.get("is_active")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


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

add_pagination(app)


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
        path_echo: str = Path(...,
                              description="Required echo in the URL path"),
        echo: str | None = Query(None, description="Optional echo string"),
):
    return make_health(echo=echo, path_echo=path_echo)


# -----------------------------------------------------------------------------
# Product endpoints
# -----------------------------------------------------------------------------


@app.post("/products", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(product: ProductCreate, response: Response, request: Request):
    """Create a new product."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check for duplicate SKU
            cur.execute("SELECT 1 FROM products WHERE sku=%s", (product.sku,))
            if cur.fetchone() is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail=f"Product with SKU '{product.sku}' already exists")
            
            product_id = uuid.uuid4()

            cur.execute("""
                INSERT INTO products (
                    product_id, sku, name, description, price, category, brand,
                    is_active, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                str(product_id),
                product.sku,
                product.name,
                product.description,
                product.price,
                product.category,
                product.brand,
                product.is_active
            ))
            conn.commit()

            cur.execute("""
                SELECT *
                FROM products
                WHERE product_id=%s
            """, (str(product_id),))
            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve created product")
            
            response_data = {
                "message": "New product created",
                "product": {
                    "product_id": row["product_id"],
                    "sku": row["sku"],
                    "name": row["name"],
                    "description": row["description"],
                    "price": row["price"],
                    "category": row["category"] if row["category"] is not None else None,
                    "brand": row["brand"] if row["brand"] is not None else None,
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            }
            
            response.headers["Location"] = str(request.url_for("get_product", product_id=product_id))

            return response_data


@app.get("/products", response_model=PaginateResponse)
def list_products(
        request: Request,
        sku: Optional[str] = Query(None, description="Filter by SKU"),
        name: Optional[str] = Query(
            None, description="Filter by name (contains)"),
        category: Optional[str] = Query(
            None, description="Filter by category"),
        brand: Optional[str] = Query(None, description="Filter by brand"),
        is_active: Optional[bool] = Query(
            None, description="Filter by active status"),
        min_price: Optional[Decimal] = Query(
            None, description="Minimum price"),
        max_price: Optional[Decimal] = Query(
            None, description="Maximum price"),
        limit: int = Query(10, ge=1, le=100),
        offset: int = Query(0, ge=0)
):
    """List all products records with optional filters."""
    sql = """
        SELECT *
        FROM products
    """
    where = []
    params = []

    if sku:
        where.append("sku LIKE %s")
        params.append(f"%{sku}%")
    if name:
        where.append("name LIKE %s")
        params.append(f"%{name}%")
    if category:
        where.append("category LIKE %s")
        params.append(f"%{category}%")
    if brand:
        where.append("brand LIKE %s")
        params.append(f"%{brand}%")
    if is_active is not None:
        where.append("is_active = %s")
        params.append(1 if is_active else 0)
    if min_price is not None:
        where.append("price >= %s")
        params.append(min_price)
    if max_price is not None:
        where.append("price <= %s")
        params.append(max_price)

    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY created_at DESC"
    sql += " LIMIT %s OFFSET %s"

    products = []

    with get_connection() as conn, conn.cursor() as cur:
        total_sql = "SELECT COUNT(*) FROM products"
        if where:
            total_sql += " WHERE " + " AND ".join(where)
        cur.execute(total_sql, params)
        total_row = cur.fetchone()
        total = total_row["COUNT(*)"] if total_row else 0

        cur.execute(sql, params + [limit, offset])
        rows = cur.fetchall()
        products = [row_to_product_read(r) for r in rows]

    params = {key:val for key, val in {
        "sku": sku,
        "name": name,
        "category": category,
        "brand": brand,
        "is_active": is_active,
        "min_price": min_price,
        "max_price": max_price
    }.items() if val is not None} # removing non-explicit None params

    def url(offset_val: int = offset):
        request_params = "&".join([f"{key}={val}" for key, val in params.items()])
        return f"{str(request.url_for("list_products"))}?{request_params}&limit={limit}&offset={offset_val}"

    links = [
        {
            "rel": "current",
            "href": url(offset)
        }
    ]

    if offset > 0:
        links.append({
            "rel": "prev",
            "href": url(max(0, offset - limit))
        })

    if offset + limit < int(total):
        links.append({
            "rel": "next",
            "href": url(offset + limit)
        })

    return PaginateResponse(data=products,links=links)


@app.get("/products/{product_id}", response_model=ProductRead)
def get_product(product_id: UUID = Path(..., description="Product ID")):
    """Get a specific product by ID."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT *
            FROM products
            WHERE product_id=%s
        """, (str(product_id),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=404, detail="Product not found")
        return row_to_product_read(row)


@app.patch("/products/{product_id}", response_model=ProductRead)
def update_product(
        product_id: UUID = Path(..., description="Product ID"),
        product_update: ProductUpdate = None,
):
    """Update a product record (partial update)."""
    update_data = product_update.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    # Check if SKU is being changed and if new SKU already exist
    if "sku" in update_data:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM products WHERE sku = %s AND product_id != %s",
                    (update_data["sku"], str(product_id))
                )
                if cur.fetchone():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Product with SKU '{update_data['sku']}' already exists"
                    )


    fields = []
    params = []
    for k, v in update_data.items():
        fields.append(f"{k}=%s")
        params.append(v)

    params.append(str(product_id))

    with get_connection() as conn:
        with conn.cursor() as cur:
            sql = f"UPDATE products SET {', '.join(fields)}, updated_at=NOW() WHERE product_id=%s"
            print(sql)
            cur.execute(sql, params)
            conn.commit()

            cur.execute("""
                SELECT *
                FROM products
                WHERE product_id=%s
            """, (str(product_id),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404, detail="Product not found")
            return row_to_product_read(row)


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: UUID = Path(..., description="Product ID")):
    """Delete a product record."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM products WHERE product_id=%s",
                        (str(product_id),))
            if cur.rowcount == 0:
                raise HTTPException(
                    status_code=404, detail="Product not found")
            conn.commit()
    return None


# -----------------------------------------------------------------------------
# Inventory endpoints
# -----------------------------------------------------------------------------


@app.post("/inventory", response_model=InventoryResponse, status_code=status.HTTP_201_CREATED)
def create_inventory(inventory: InventoryCreate, response: Response, request: Request):
    """Create a new inventory record for a product."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM products WHERE product_id=%s",
                        (str(inventory.product_id),))
            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=404, detail="Product not found")

            cur.execute("SELECT 1 FROM inventory WHERE product_id=%s",
                        (str(inventory.product_id),))
            if cur.fetchone():
                raise HTTPException(
                    status_code=400, detail="Inventory record already exists for this product")
            
            inventory_id = uuid.uuid4()

            cur.execute("""
                INSERT INTO inventory (
                    inventory_id, product_id, quantity, warehouse_location,
                    reorder_level, reorder_quantity, reserved_quantity,
                    last_restocked_at, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), NOW())
            """, (
                str(inventory_id),
                str(inventory.product_id),
                inventory.quantity,
                inventory.warehouse_location,
                inventory.reorder_level,
                inventory.reorder_quantity,
                inventory.reserved_quantity
            ))
            conn.commit()

            cur.execute("""
                SELECT *,
                       (quantity - reserved_quantity) AS available_quantity,
                       (reorder_level IS NOT NULL AND quantity <= reorder_level) AS needs_reorder
                FROM inventory
                WHERE product_id=%s
            """, (str(inventory.product_id),))
            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve created inventory")
            
            response_data = {
                "message": "New inventory created",
                "product": {
                    "inventory_id": row["inventory_id"],
                    "product_id": row["product_id"],
                    "quantity": row["quantity"],
                    "warehouse_location": row["warehouse_location"] if row["warehouse_location"] is not None else None,
                    "reorder_level": row["reorder_level"] if row["reorder_level"] is not None else None,
                    "reorder_quantity": row["reorder_quantity"] if row["reorder_quantity"] is not None else None,
                    "reserved_quantity": row["reserved_quantity"],
                    "available_quantity": row["available_quantity"],
                    "needs_reorder": bool(row["needs_reorder"]),
                    "last_restocked_at": row["last_restocked_at"] if row["last_restocked_at"] is not None else None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            }
            
            response.headers["Location"] = str(request.url_for("get_inventory", inventory_id=inventory_id))

            return response_data


@app.get("/inventory", response_model=PaginateResponse)
def list_inventory(
    request: Request,
    product_id: Optional[UUID] = Query(None),
    warehouse_location: Optional[str] = Query(None),
    needs_reorder: Optional[bool] = Query(None),
    low_stock: Optional[bool] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all inventory records with optional filters."""
    sql = """
        SELECT *,
               (quantity - reserved_quantity) AS available_quantity,
               (reorder_level IS NOT NULL AND quantity <= reorder_level) AS needs_reorder
        FROM inventory
    """
    where = []
    params = []

    if product_id:
        where.append("product_id = %s")
        params.append(str(product_id))
    if warehouse_location:
        where.append("warehouse_location LIKE %s")
        params.append(f"%{warehouse_location}%")
    if needs_reorder is not None:
        where.append(
            "(reorder_level IS NOT NULL AND quantity <= reorder_level) = %s")
        params.append(1 if needs_reorder else 0)
    if low_stock:
        where.append("quantity < 10")

    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    sql += " LIMIT %s OFFSET %s"

    inventory = []

    with get_connection() as conn, conn.cursor() as cur:
        total_sql = "SELECT COUNT(*) FROM inventory"
        if where:
            total_sql += " WHERE " + " AND ".join(where)
        cur.execute(total_sql, params)
        total_row = cur.fetchone()
        total = total_row["COUNT(*)"] if total_row else 0

        cur.execute(sql, params + [limit, offset])
        rows = cur.fetchall()
        inventory = [row_to_inventory_read(r) for r in rows]

    params = {key:val for key, val in {
        "product_id": product_id,
        "warehouse_location": warehouse_location,
        "needs_reorder": needs_reorder,
        "low_stock": low_stock
    }.items() if val is not None} # removing non-explicit None params

    def url(offset_val: int = offset):
        request_params = "&".join([f"{key}={val}" for key, val in params.items()])
        return f"{str(request.url_for("list_inventory"))}?{request_params}&limit={limit}&offset={offset_val}"

    links = [
        {
            "rel": "current",
            "href": url(offset)
        }
    ]

    if offset > 0:
        links.append({
            "rel": "prev",
            "href": url(max(0, offset - limit))
        })

    if offset + limit < int(total):
        links.append({
            "rel": "next",
            "href": url(offset + limit)
        })

    return PaginateResponse(data=inventory,links=links)


@app.get("/inventory/{inventory_id}", response_model=InventoryRead)
def get_inventory(inventory_id: UUID = Path(...)):
    """Get a specific inventory record by ID."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT *,
                   (quantity - reserved_quantity) AS available_quantity,
                   (reorder_level IS NOT NULL AND quantity <= reorder_level) AS needs_reorder
            FROM inventory
            WHERE inventory_id=%s
        """, (str(inventory_id),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=404, detail="Inventory record not found")
        return row_to_inventory_read(row)


@app.get("/inventory/product/{product_id}", response_model=InventoryRead)
def get_inventory_by_product(product_id: UUID = Path(...)):
    """Get inventory record for a specific product."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT *,
                   (quantity - reserved_quantity) AS available_quantity,
                   (reorder_level IS NOT NULL AND quantity <= reorder_level) AS needs_reorder
            FROM inventory
            WHERE product_id=%s
        """, (str(product_id),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=404, detail="Inventory record not found for this product")
        return row_to_inventory_read(row)


@app.patch("/inventory/{inventory_id}", response_model=InventoryRead)
def update_inventory(inventory_id: UUID, inventory_update: InventoryUpdate):
    """Update an inventory record (partial update)."""
    update_data = inventory_update.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    fields = []
    params = []
    for k, v in update_data.items():
        fields.append(f"{k}=%s")
        params.append(v)

    params.append(str(inventory_id))

    with get_connection() as conn:
        with conn.cursor() as cur:
            sql = f"UPDATE inventory SET {', '.join(fields)}, updated_at=NOW() WHERE inventory_id=%s"
            print(sql)
            cur.execute(sql, params)
            conn.commit()

            cur.execute("""
                SELECT *,
                       (quantity - reserved_quantity) AS available_quantity,
                       (reorder_level IS NOT NULL AND quantity <= reorder_level) AS needs_reorder
                FROM inventory
                WHERE inventory_id=%s
            """, (str(inventory_id),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404, detail="Inventory record not found")
            return row_to_inventory_read(row)


@app.patch("/inventory/{inventory_id}/adjust", response_model=InventoryRead)
def adjust_inventory(inventory_id: UUID, adjustment: InventoryAdjustment):
    """Adjust inventory quantity (add or remove stock)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT quantity, reserved_quantity, reorder_level FROM inventory WHERE inventory_id=%s FOR UPDATE", (str(inventory_id),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404, detail="Inventory record not found")

            new_qty = row["quantity"] + adjustment.adjustment
            if new_qty < 0:
                raise HTTPException(
                    status_code=400, detail="Insufficient stock")

            cur.execute("""
                UPDATE inventory
                SET quantity=%s,
                    last_restocked_at = CASE WHEN %s > 0 THEN NOW() ELSE last_restocked_at END,
                    updated_at=NOW()
                WHERE inventory_id=%s
            """, (new_qty, adjustment.adjustment, str(inventory_id)))
            conn.commit()

            cur.execute("""
                SELECT *,
                       (quantity - reserved_quantity) AS available_quantity,
                       (reorder_level IS NOT NULL AND quantity <= reorder_level) AS needs_reorder
                FROM inventory
                WHERE inventory_id=%s
            """, (str(inventory_id),))
            row = cur.fetchone()
            return row_to_inventory_read(row)


@app.delete("/inventory/{inventory_id}", status_code=204)
def delete_inventory(inventory_id: UUID):
    """Delete an inventory record."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM inventory WHERE inventory_id=%s",
                        (str(inventory_id),))
            if cur.rowcount == 0:
                raise HTTPException(
                    status_code=404, detail="Inventory record not found")
            conn.commit()
    return None


# -----------------------------------------------------------------------------
# Inventory statistics endpoints
# -----------------------------------------------------------------------------


@app.get("/inventory/stats/summary")
def get_inventory_summary():
    """Get inventory summary statistics."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total_products,
                COALESCE(SUM(quantity),0) AS total_quantity,
                COALESCE(SUM(reserved_quantity),0) AS total_reserved,
                COALESCE(SUM(quantity - reserved_quantity),0) AS total_available,
                SUM(CASE WHEN reorder_level IS NOT NULL AND quantity <= reorder_level THEN 1 ELSE 0 END) AS items_needing_reorder,
                SUM(CASE WHEN quantity < 10 THEN 1 ELSE 0 END) AS low_stock_items
            FROM inventory
        """)
        return cur.fetchone()


# -----------------------------------------------------------------------------
# Run the app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
