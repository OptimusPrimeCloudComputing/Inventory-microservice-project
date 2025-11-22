# E-commerce Inventory Management Microservice


## Features

- **Product Management**: Create, read, update, and delete products with comprehensive metadata
- **Inventory Tracking**: Real-time inventory management with unlimited capacity
- **Stock Reservations**: Reserve inventory for pending orders and shopping carts
- **Reorder Management**: Automatic reorder alerts based on configurable thresholds
- **Inventory Adjustments**: Track all inventory changes with full audit history
- **Flexible Fulfillment**: Support for warehouse locations and drop-shipped items
- **Advanced Filtering**: Search and filter products and inventory by multiple criteria

## Tech Stack

- **Framework**: FastAPI 
- **Data Validation**: Pydantic v2
- **Database**: MySQL 8.0+ (Proposed and schema provided in db_schema.txt)
- **Python**: 3.10+

## Project Structure

```
inventory-management/
├── main.py                 # FastAPI application and endpoints
├── models/
│   ├── product.py         # Product Pydantic models
│   ├── inventory.py       # Inventory Pydantic models
│   └── health.py          # Health check model
├── schema.sql             # MySQL database schema
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip install - r requirements.txt

## Run with
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check with optional echo |
| GET | `/health/{path_echo}` | Health check with path parameter |

### Products

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/products` | Create a new product |
| GET | `/products` | List all products (with filters) |
| GET | `/products/{product_id}` | Get a specific product |
| PUT | `/products/{product_id}` | Replace a specific product |
| PATCH | `/products/{product_id}` | Update a product |
| DELETE | `/products/{product_id}` | Delete a product |

**Product Filters:**
- `sku`: Filter by SKU
- `name`: Filter by name (contains)
- `category`: Filter by category
- `brand`: Filter by brand
- `is_active`: Filter by active status
- `min_price`: Minimum price
- `max_price`: Maximum price

### Inventory

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/inventory` | Create inventory record for a product |
| GET | `/inventory` | List all inventory (with filters) |
| GET | `/inventory/{inventory_id}` | Get specific inventory record |
| GET | `/inventory/product/{product_id}` | Get inventory by product ID |
| PUT | `/inventory/{inventory_id}` | Replace specific inventory record |
| PATCH | `/inventory/{inventory_id}` | Update inventory record |
| POST | `/inventory/{inventory_id}/adjust` | Adjust inventory quantity |
| DELETE | `/inventory/{inventory_id}` | Delete inventory record |

**Inventory Filters:**
- `product_id`: Filter by product ID
- `warehouse_location`: Filter by location
- `needs_reorder`: Filter items needing reorder
- `low_stock`: Show items with quantity < 10

### Inventory Statistics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/inventory/stats/summary` | Get inventory summary statistics |


## Key Design Decisions

### Unlimited Inventory Capacity

This system is designed for **e-commerce** where there are no physical warehouse capacity constraints:

- Inventory quantities use `BIGINT` 
- No maximum capacity checks or limits
- Supports drop-shipping (null warehouse locations)
- Scalable for high-volume online retail

### Stock Reservation System

The `reserved_quantity` field enables:
- Shopping cart reservations
- Pending order holds
- Accurate available inventory calculation
- Prevention of overselling

### Reorder Management

Automatic reorder alerts help maintain optimal stock levels:
- Configurable `reorder_level` triggers alerts
- Configurable `reorder_quantity` suggests order amounts
- `needs_reorder` flag for easy filtering

### Audit Trail

The `inventory_adjustments` table (in database schema) tracks:
- All quantity changes
- Reasons for adjustments
- Who made the changes
- Timestamps for compliance

## Database Schema

The MySQL schema includes:

- **products**: Core product information
- **inventory**: Stock levels and locations
- **inventory_adjustments**: Full audit history
- **Views**: Convenient queries for common operations
  - `v_products_inventory`: Products with inventory status
  - `v_low_stock_items`: Items needing reorder
  - `v_out_of_stock`: Items with no available stock
  - `v_available_products`: In-stock, active products

See `schema.sql` for complete details.

## Business Rules

1. **SKU Uniqueness**: Each product must have a unique SKU
2. **Inventory One-to-One**: Each product can have only one inventory record
3. **Non-Negative Stock**: Inventory quantities cannot be negative
4. **Reserved <= Total**: Reserved quantity cannot exceed total quantity
5. **Cascade Delete**: Deleting a product requires removing inventory first
6. **Price Validation**: Product prices must be >= 0

