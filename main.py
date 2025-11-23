from __future__ import annotations

import os
import socket
from datetime import datetime
from decimal import Decimal

from typing import Dict, List, Optional
import uuid
from uuid import UUID
import hashlib
import json
from email.utils import parsedate_to_datetime
from datetime import timezone

from fastapi import FastAPI, HTTPException, Response, status, Request, Query, Path
from fastapi_pagination import add_pagination
from fastapi.middleware.cors import CORSMiddleware

from models.product import ProductCreate, ProductRead, ProductReplace, ProductUpdate, ProductResponse
from models.inventory import (
    InventoryCreate,
    InventoryRead,
    InventoryReplace,
    InventoryUpdate,
    InventoryAdjustment,
    InventoryResponse,
)
from models.async_response import AsyncOperationResponse, OperationStatusResponse, OperationStatus
from models.gen_response import PaginateResponse
from models.health import Health
from db import get_connection
from async_manager import async_operation_manager
import pymysql


def _to_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return v
    
def get_product_links(
        product_id: UUID, 
        request: Request, 
        inventory_id: Optional[UUID] = None
):
    links = [
        {
            "rel": "self",
            "href": str(request.url_for("get_product", product_id=product_id).path)
        }
    ]

    if inventory_id:
        links.append({
            "rel": "inventory-check",
            "href": str(request.url_for("get_inventory", inventory_id=inventory_id).path)
        })
    else:
        links.append({
            "rel": "inventory-check",
            "href": str(request.url_for("create_inventory").path)
        })

    return links

def get_inventory_links(
        inventory_id: UUID,
        request: Request,
        product_id: UUID,
        query_by_product: bool = False
):
    links = []

    if query_by_product:
        links.append(
            {
                "rel": "self",
                "href": str(request.url_for("get_inventory_by_product", product_id=product_id).path)
            }
        )
    else:
        links.append(
            {
                "rel": "self",
                "href": str(request.url_for("get_inventory", inventory_id=inventory_id).path)
            }
        )
    
    links.append(
        {
            "rel": "product",
            "href": str(request.url_for("get_product", product_id=product_id).path)
        }
    )

    return links

def set_product_links(
        product_id: UUID,
        request: Request,
        cur: pymysql.cursors.Cursor
):
    cur.execute("""
        SELECT inventory_id
        FROM inventory
        WHERE product_id=%s
    """, (str(product_id),))
    inv_row = cur.fetchone()

    links = get_product_links(product_id=product_id, request=request)

    if inv_row:
        links = get_product_links(product_id=product_id, request=request, inventory_id=UUID(str(inv_row["inventory_id"])))

    return links

def set_inventory_links(
        inventory_id: UUID,
        request: Request,
        cur: pymysql.cursors.Cursor,
        query_by_product: bool = False
):
    cur.execute("""
        SELECT product_id
        FROM inventory
        WHERE inventory_id=%s
    """, (str(inventory_id),))
    prod_row = cur.fetchone()

    links = get_inventory_links(inventory_id=inventory_id, request=request, product_id=UUID(str(prod_row["product_id"])), query_by_product=query_by_product)

    return links
    
    
def row_to_product_read(row: dict, links: List = []) -> ProductRead:
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
        links=links
    )

def row_to_inventory_read(row: dict, links: List = []) -> InventoryRead:
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
        links=links
    )


def create_etag(row: dict):
    """
    Generates an ETag based on all row data
    """
    if not row:
        return None

    items = []
    for k in sorted(row.keys()):
        v = row[k]

        if isinstance(v, datetime):
            v = v.isoformat()

        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8")

        v = str(v)

        items.append((k, None if v is None else str(v)))

    canonical = json.dumps(items, separators=(",",":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f'"{digest}"'


def _parse_http_date(value: str):
    """Parse an HTTP-date string to a timezone-aware UTC datetime. Returns None on parse error."""
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            # assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def _etag_matches(header_value: str, etag: str) -> bool:
    """Return True if any ETag in header_value matches etag. Support '*' wildcard."""
    if not header_value:
        return False
    header_value = header_value.strip()
    if header_value == '*':
        return True
    parts = [p.strip() for p in header_value.split(',') if p.strip()]
    return any(p == etag for p in parts)


def _set_last_modified_header(response: Response, updated_at):
    try:
        if hasattr(updated_at, 'strftime'):
            # Format as RFC1123 (HTTP-date) in GMT
            lm = updated_at.strftime("%a, %d %b %Y %H:%M:%S GMT")
            response.headers['Last-Modified'] = lm
    except Exception:
        pass


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

# ============================================================================
# CORS Middleware Configuration
# ============================================================================
# Enable CORS to accept requests from any origin (client URL)
# Useful for frontend applications, webhooks, and external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=False,  # Allow credentials (cookies, auth headers)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, PATCH, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (Content-Type, Authorization, etc.)
)

# ============================================================================
# Health endpoints
# ============================================================================


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
# Async Operations endpoints
# -----------------------------------------------------------------------------


@app.get("/operations/{operation_id}", response_model=OperationStatusResponse)
async def get_operation_status(operation_id: str = Path(..., description="Operation ID"), response: Response = None):
    """Poll the status of an async operation."""
    operation = await async_operation_manager.get_operation(operation_id)
    if not operation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Operation {operation_id} not found"
        )
    
    if operation.status == OperationStatus.COMPLETED and operation.result:
        etag = operation.result.get("etag")
        if etag and response:
            response.headers["ETag"] = etag
    
    return OperationStatusResponse(
        operation_id=operation.operation_id,
        status=operation.status,
        message=operation.message,
        progress_percent=operation.progress_percent,
        created_at=operation.created_at,
        updated_at=operation.updated_at,
        result=operation.result,
        error=operation.error
    )



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
            
            links = set_product_links(product_id=product_id, request=request, cur=cur)

            etag = create_etag(row)
            if etag:
                response.headers["ETag"] = etag
            
            response_data = {
                "message": "New product created",
                "product": {
                    "product_id": str(product_id),
                    "sku": row["sku"],
                    "name": row["name"],
                    "description": row["description"],
                    "price": row["price"],
                    "category": row["category"] if row["category"] is not None else None,
                    "brand": row["brand"] if row["brand"] is not None else None,
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "links": links
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

        products = []

        for row in rows:
            links = set_product_links(product_id=UUID(str(row["product_id"])), request=request, cur=cur)
            products.append(row_to_product_read(row=row, links=links))

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
        if request_params:
            request_params += "&"
        return f"{str(request.url_for('list_products').path)}?{request_params}limit={limit}&offset={offset_val}"

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
def get_product(request: Request, response: Response, product_id: UUID = Path(..., description="Product ID")):
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
        
        links = set_product_links(product_id=product_id, request=request, cur=cur)

        etag = create_etag(row)

        inm = request.headers.get("if-none-match")
        if inm and etag and _etag_matches(inm, etag):
            return Response(status_code=status.HTTP_304_NOT_MODIFIED)

        ims = request.headers.get("if-modified-since")
        if ims and row.get("updated_at") is not None:
            parsed = _parse_http_date(ims)
            if parsed is not None:
                updated_at = row.get("updated_at")
                if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                    updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                if updated_at <= parsed:
                    return Response(status_code=status.HTTP_304_NOT_MODIFIED)

        if etag:
            response.headers["ETag"] = etag

        _set_last_modified_header(response, row.get("updated_at"))

        return row_to_product_read(row=row, links=links)


@app.put("/products/{product_id}", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def replace_product(
    request: Request,
    response: Response,
    product_id: UUID = Path(..., description="Product ID"),
    product_replace: ProductReplace = None,
):
    """Replace a product entirely (all fields required). The product ID will remain the same. Returns 202 Accepted with async operation."""
    if product_replace is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body is required")
    
    if_match = request.headers.get("if-match")
    if_unmodified_since = request.headers.get("if-unmodified-since")
    base_url = str(request.base_url).rstrip('/')
    
    async def _async_replace_product():
        """Coroutine to execute product replacement asynchronously."""
        replace_data = product_replace.model_dump()
        
        # Validate that all required fields are present
        required_fields = {"sku", "name", "price", "is_active"}
        missing_fields = required_fields - set(replace_data.keys())
        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Missing required fields for full replacement: {', '.join(missing_fields)}"
            )
        
        # Check if SKU is being changed and if new SKU already exist
        if "sku" in replace_data:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM products WHERE sku = %s AND product_id != %s",
                        (replace_data["sku"], str(product_id))
                    )
                    if cur.fetchone():
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Product with SKU '{replace_data['sku']}' already exists"
                        )

        fields = []
        params = []
        for k, v in replace_data.items():
            fields.append(f"{k}=%s")
            params.append(v)

        params.append(str(product_id))

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if product exists
                cur.execute("SELECT 1 FROM products WHERE product_id=%s", (str(product_id),))
                if not cur.fetchone():
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
                
                # Concurrency checks: prefer If-Match; fallback to If-Unmodified-Since
                cur.execute("SELECT * FROM products WHERE product_id=%s", (str(product_id),))
                current_row = cur.fetchone()
                if not current_row:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
                
                current_etag = create_etag(current_row)

                if if_match:
                    if not _etag_matches(if_match, current_etag):
                        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")
                else:
                    # check If-Unmodified-Since
                    if if_unmodified_since and current_row.get("updated_at") is not None:
                        parsed = _parse_http_date(if_unmodified_since)
                        if parsed is not None:
                            updated_at = current_row.get("updated_at")
                            if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                            if updated_at > parsed:
                                raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Resource has been modified")
                
                sql = f"UPDATE products SET {', '.join(fields)}, updated_at=NOW() WHERE product_id=%s"
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

                # Create links manually without request object
                cur.execute("""
                    SELECT inventory_id
                    FROM inventory
                    WHERE product_id=%s
                """, (str(product_id),))
                inv_row = cur.fetchone()

                links = [
                    {
                        "rel": "self",
                        "href": f"{base_url}/products/{product_id}"
                    }
                ]

                if inv_row:
                    links.append({
                        "rel": "inventory-check",
                        "href": f"{base_url}/inventory/{inv_row['inventory_id']}"
                    })
                else:
                    links.append({
                        "rel": "inventory-check",
                        "href": f"{base_url}/inventory"
                    })

                etag = create_etag(row)

                product_data = row_to_product_read(row=row, links=links)
                return {
                    "product": product_data.model_dump(),
                    "etag": etag
                }
    
    # Create async operation
    operation_id = await async_operation_manager.create_operation(
        message="Product replacement initiated",
        coroutine=_async_replace_product()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Product replacement initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )
        

@app.patch("/products/{product_id}", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def update_product(
    request: Request,
    response: Response,
    product_id: UUID = Path(..., description="Product ID"),
    product_update: ProductUpdate = None,
):
    """Update a product record (partial update). Returns 202 Accepted with async operation."""
    
    # Capture request data that will be needed in the async operation
    if_match = request.headers.get("if-match")
    if_unmodified_since = request.headers.get("if-unmodified-since")
    base_url = str(request.base_url).rstrip('/')
    
    async def _async_update_product():
        """Coroutine to execute product update asynchronously."""
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
                            status_code=status.HTTP_409_CONFLICT,
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
                # Concurrency checks: prefer If-Match; fallback to If-Unmodified-Since
                cur.execute("SELECT * FROM products WHERE product_id=%s", (str(product_id),))
                current_row = cur.fetchone()
                if not current_row:
                    raise HTTPException(status_code=404, detail="Product not found")
                
                current_etag = create_etag(current_row)

                if if_match:
                    if not _etag_matches(if_match, current_etag):
                        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")
                else:
                    if if_unmodified_since and current_row.get("updated_at") is not None:
                        parsed = _parse_http_date(if_unmodified_since)
                        if parsed is not None:
                            updated_at = current_row.get("updated_at")
                            if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                            if updated_at > parsed:
                                raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Resource has been modified")

                sql = f"UPDATE products SET {', '.join(fields)}, updated_at=NOW() WHERE product_id=%s"
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

                # Create links manually without request object
                cur.execute("""
                    SELECT inventory_id
                    FROM inventory
                    WHERE product_id=%s
                """, (str(product_id),))
                inv_row = cur.fetchone()

                links = [
                    {
                        "rel": "self",
                        "href": f"{base_url}/products/{product_id}"
                    }
                ]

                if inv_row:
                    links.append({
                        "rel": "inventory-check",
                        "href": f"{base_url}/inventory/{inv_row['inventory_id']}"
                    })
                else:
                    links.append({
                        "rel": "inventory-check",
                        "href": f"{base_url}/inventory"
                    })

                etag = create_etag(row)

                product_data = row_to_product_read(row=row, links=links)
                return {
                    "product": product_data.model_dump(),
                    "etag": etag
                }
    
    operation_id = await async_operation_manager.create_operation(
        message="Product update initiated",
        coroutine=_async_update_product()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Product update initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )


@app.delete("/products/{product_id}", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def delete_product(
    request: Request,
    response: Response,
    product_id: UUID = Path(..., description="Product ID")
):
    """Delete a product record. Returns 202 Accepted with async operation."""
    
    # Capture request data that will be needed in the async operation
    if_match = request.headers.get("if-match")
    if_unmodified_since = request.headers.get("if-unmodified-since")
    
    async def _async_delete_product():
        """Coroutine to execute product deletion asynchronously."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check current ETag before deleting
                cur.execute("SELECT * FROM products WHERE product_id=%s", (str(product_id),))
                current_row = cur.fetchone()
                if not current_row:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
                
                current_etag = create_etag(current_row)

                if if_match:
                    if not _etag_matches(if_match, current_etag):
                        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")
                else:
                    if if_unmodified_since and current_row.get("updated_at") is not None:
                        parsed = _parse_http_date(if_unmodified_since)
                        if parsed is not None:
                            updated_at = current_row.get("updated_at")
                            if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                            if updated_at > parsed:
                                raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Resource has been modified")

                cur.execute("DELETE FROM products WHERE product_id=%s",
                            (str(product_id),))
                if cur.rowcount == 0:
                    raise HTTPException(
                        status_code=404, detail="Product not found")
                conn.commit()
        
        return {"product_id": str(product_id), "message": "Product deleted successfully"}
    
    operation_id = await async_operation_manager.create_operation(
        message="Product deletion initiated",
        coroutine=_async_delete_product()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Product deletion initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )


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
            
            links = set_inventory_links(inventory_id=inventory_id, request=request, cur=cur)
            
            etag = create_etag(row)
            if etag:
                response.headers["ETag"] = etag
                
            response_data = {
                "message": "New inventory created",
                "product": {
                    "inventory_id": str(inventory_id),
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
                    "updated_at": row["updated_at"],
                    "links": links
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

        inventory = []

        for row in rows:
            links = set_inventory_links(inventory_id=UUID(str(row["inventory_id"])), request=request, cur=cur)
            inventory.append(row_to_inventory_read(row=row, links=links))

    params = {key:val for key, val in {
        "product_id": product_id,
        "warehouse_location": warehouse_location,
        "needs_reorder": needs_reorder,
        "low_stock": low_stock
    }.items() if val is not None} # removing non-explicit None params

    def url(offset_val: int = offset):
        request_params = "&".join([f"{key}={val}" for key, val in params.items()])
        if request_params:
            request_params += "&"
        return f"{str(request.url_for('list_inventory').path)}?{request_params}limit={limit}&offset={offset_val}"

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
def get_inventory(request: Request, response: Response, inventory_id: UUID = Path(...)):
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
                status_code=status.HTTP_404_NOT_FOUND, detail="Inventory record not found")
        
        links = set_inventory_links(inventory_id=inventory_id, request=request, cur=cur)

        etag = create_etag(row)

        inm = request.headers.get("if-none-match")
        if inm and etag and _etag_matches(inm, etag):
            return Response(status_code=status.HTTP_304_NOT_MODIFIED)
        
        ims = request.headers.get("if-modified-since")
        if ims and row.get("updated_at") is not None:
            parsed = _parse_http_date(ims)
            if parsed is not None:
                updated_at = row.get("updated_at")
                if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                    updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                if updated_at <= parsed:
                    return Response(status_code=status.HTTP_304_NOT_MODIFIED)
                
        if etag:
            response.headers["ETag"] = etag

        _set_last_modified_header(response, row.get("updated_at"))

        return row_to_inventory_read(row=row, links=links)


@app.get("/inventory/product/{product_id}", response_model=InventoryRead)
def get_inventory_by_product(request: Request, response: Response, product_id: UUID = Path(...)):
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
        
        links = set_inventory_links(inventory_id=UUID(str(row["inventory_id"])), request=request, cur=cur, query_by_product=True)
        
        etag = create_etag(row)

        if etag:
            inm = request.headers.get("if-none-match")
            if inm and inm == etag:
                return Response(status_code=status.HTTP_304_NOT_MODIFIED)
            response.headers["ETag"] = etag

        return row_to_inventory_read(row=row, links=links)


@app.put("/inventory/{inventory_id}", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def replace_inventory(
    request: Request,
    response: Response,
    inventory_id: UUID,
    inventory_replace: InventoryReplace
):
    """Replace an inventory record entirely (all fields required). Returns 202 Accepted with async operation."""
    
    # Capture request data that will be needed in the async operation
    if_match = request.headers.get("if-match")
    if_unmodified_since = request.headers.get("if-unmodified-since")
    base_url = str(request.base_url).rstrip('/')
    
    async def _async_replace_inventory():
        """Coroutine to execute inventory replacement asynchronously."""
        if inventory_replace is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body is required")
        
        replace_data = inventory_replace.model_dump(exclude_unset=False)
        
        # Validate that the product exists
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM products WHERE product_id=%s", (str(replace_data["product_id"]),))
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, 
                        detail=f"Product with ID {replace_data['product_id']} not found"
                    )
                
        # Validate that all required fields are present
        required_fields = {"product_id", "quantity", "reserved_quantity"}
        missing_fields = required_fields - set(replace_data.keys())
        if missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Missing required fields for full replacement: {', '.join(missing_fields)}"
            )

        fields = []
        params = []
        for k, v in replace_data.items():
            fields.append(f"{k}=%s")
            params.append(v)

        params.append(str(inventory_id))

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Concurrency checks: prefer If-Match; fallback to If-Unmodified-Since
                cur.execute("SELECT * FROM inventory WHERE inventory_id=%s", (str(inventory_id),))
                existing = cur.fetchone()
                if not existing:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inventory record not found")
                
                current_etag = create_etag(existing)
                
                if if_match:
                    if not _etag_matches(if_match, current_etag):
                        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")
                else:
                    if if_unmodified_since and existing.get("updated_at") is not None:
                        parsed = _parse_http_date(if_unmodified_since)
                        if parsed is not None:
                            updated_at = existing.get("updated_at")
                            if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                            if updated_at > parsed:
                                raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Resource has been modified")

                sql = f"UPDATE inventory SET {', '.join(fields)}, updated_at=NOW() WHERE inventory_id=%s"
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
                
                # Create links manually without request object
                cur.execute("""
                    SELECT product_id
                    FROM inventory
                    WHERE inventory_id=%s
                """, (str(inventory_id),))
                prod_row = cur.fetchone()

                links = [
                    {
                        "rel": "self",
                        "href": f"{base_url}/inventory/{inventory_id}"
                    },
                    {
                        "rel": "product",
                        "href": f"{base_url}/products/{prod_row['product_id']}"
                    }
                ]

                etag = create_etag(row)

                inventory_data = row_to_inventory_read(row=row, links=links)
                return {
                    "inventory": inventory_data.model_dump(),
                    "etag": etag
                }
    
    operation_id = await async_operation_manager.create_operation(
        message="Inventory replacement initiated",
        coroutine=_async_replace_inventory()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Inventory replacement initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )


@app.patch("/inventory/{inventory_id}", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def update_inventory(
    request: Request,
    response: Response,
    inventory_id: UUID,
    inventory_update: InventoryUpdate
):
    """Update an inventory record (partial update). Returns 202 Accepted with async operation."""
    
    # Capture request data that will be needed in the async operation
    if_match = request.headers.get("if-match")
    if_unmodified_since = request.headers.get("if-unmodified-since")
    base_url = str(request.base_url).rstrip('/')
    
    async def _async_update_inventory():
        """Coroutine to execute inventory update asynchronously."""
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
                # Concurrency checks: prefer If-Match; fallback to If-Unmodified-Since
                cur.execute("SELECT * FROM inventory WHERE inventory_id=%s", (str(inventory_id),))
                existing = cur.fetchone()
                if not existing:
                    raise HTTPException(status_code=404, detail="Inventory record not found")
                
                current_etag = create_etag(existing)

                if if_match:
                    if not _etag_matches(if_match, current_etag):
                        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")
                else:
                    if if_unmodified_since and existing.get("updated_at") is not None:
                        parsed = _parse_http_date(if_unmodified_since)
                        if parsed is not None:
                            updated_at = existing.get("updated_at")
                            if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                            if updated_at > parsed:
                                raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Resource has been modified")

                sql = f"UPDATE inventory SET {', '.join(fields)}, updated_at=NOW() WHERE inventory_id=%s"
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
                
                # Create links manually without request object
                cur.execute("""
                    SELECT product_id
                    FROM inventory
                    WHERE inventory_id=%s
                """, (str(inventory_id),))
                prod_row = cur.fetchone()

                links = [
                    {
                        "rel": "self",
                        "href": f"{base_url}/inventory/{inventory_id}"
                    },
                    {
                        "rel": "product",
                        "href": f"{base_url}/products/{prod_row['product_id']}"
                    }
                ]

                etag = create_etag(row)

                inventory_data = row_to_inventory_read(row=row, links=links)
                return {
                    "inventory": inventory_data.model_dump(),
                    "etag": etag
                }
    
    operation_id = await async_operation_manager.create_operation(
        message="Inventory update initiated",
        coroutine=_async_update_inventory()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Inventory update initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )


@app.patch("/inventory/{inventory_id}/adjust", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def adjust_inventory(
    request: Request,
    response: Response,
    inventory_id: UUID,
    adjustment: InventoryAdjustment
):
    """Adjust inventory quantity (add or remove stock). Returns 202 Accepted with async operation."""
    
    # Capture request data that will be needed in the async operation
    if_match = request.headers.get("if-match")
    base_url = str(request.base_url).rstrip('/')
    
    async def _async_adjust_inventory():
        """Coroutine to execute inventory adjustment asynchronously."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check If-Match for concurrency
                cur.execute("SELECT * FROM inventory WHERE inventory_id=%s", (str(inventory_id),))
                existing = cur.fetchone()

                current_etag = create_etag(existing) if existing else None

                if if_match and current_etag and if_match != current_etag:
                    raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")

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

                # Create links manually without request object
                cur.execute("""
                    SELECT product_id
                    FROM inventory
                    WHERE inventory_id=%s
                """, (str(inventory_id),))
                prod_row = cur.fetchone()

                links = [
                    {
                        "rel": "self",
                        "href": f"{base_url}/inventory/{inventory_id}"
                    },
                    {
                        "rel": "product",
                        "href": f"{base_url}/products/{prod_row['product_id']}"
                    }
                ]

                etag = create_etag(row)

                inventory_data = row_to_inventory_read(row=row, links=links)
                return {
                    "inventory": inventory_data.model_dump(),
                    "etag": etag
                }
    
    operation_id = await async_operation_manager.create_operation(
        message="Inventory adjustment initiated",
        coroutine=_async_adjust_inventory()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Inventory adjustment initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )


@app.delete("/inventory/{inventory_id}", response_model=AsyncOperationResponse, status_code=status.HTTP_202_ACCEPTED)
async def delete_inventory(
    request: Request,
    inventory_id: UUID,
    response: Response
):
    """Delete an inventory record. Returns 202 Accepted with async operation."""
    
    # Capture request data that will be needed in the async operation
    if_match = request.headers.get("if-match")
    if_unmodified_since = request.headers.get("if-unmodified-since")
    
    async def _async_delete_inventory():
        """Coroutine to execute inventory deletion asynchronously."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM inventory WHERE inventory_id=%s", (str(inventory_id),))
                current_row = cur.fetchone()
                if not current_row:
                    raise HTTPException(status_code=404, detail="Inventory record not found")
                
                current_etag = create_etag(current_row)

                if if_match:
                    if not _etag_matches(if_match, current_etag):
                        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="ETag does not match")
                else:
                    if if_unmodified_since and current_row.get("updated_at") is not None:
                        parsed = _parse_http_date(if_unmodified_since)
                        if parsed is not None:
                            updated_at = current_row.get("updated_at")
                            if hasattr(updated_at, "tzinfo") and updated_at.tzinfo is not None:
                                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
                            if updated_at > parsed:
                                raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Resource has been modified")

                cur.execute("DELETE FROM inventory WHERE inventory_id=%s",
                            (str(inventory_id),))
                if cur.rowcount == 0:
                    raise HTTPException(
                        status_code=404, detail="Inventory record not found")
                conn.commit()
        
        return {"inventory_id": str(inventory_id), "message": "Inventory deleted successfully"}
    
    operation_id = await async_operation_manager.create_operation(
        message="Inventory deletion initiated",
        coroutine=_async_delete_inventory()
    )
    
    response.headers["Location"] = str(request.url_for("get_operation_status", operation_id=operation_id))
    
    return AsyncOperationResponse(
        operation_id=operation_id,
        status=OperationStatus.PENDING,
        message="Inventory deletion initiated",
        status_url=str(request.url_for("get_operation_status", operation_id=operation_id))
    )


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