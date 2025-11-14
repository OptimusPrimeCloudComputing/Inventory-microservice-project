from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, StringConstraints

class Link(BaseModel):
    rel: str = Field(
        ...,
        description="The relation type of the link.",
        json_schema_extra={"example": "self"},
    )
    href: str = Field(
        ...,
        description="The URL of the link.",
        json_schema_extra={"example": "https://api.example.com/products/123"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rel": "self",
                    "href": "https://api.example.com/products/123",
                }
            ]
        }
    }

class PaginateResponse(BaseModel):
    data: List # no specification because it could be for more than one type of collection
    links: List[Link] = Field(
        ...,
        description="Pagination links.",
        json_schema_extra={
            "examples": [
                {
                    "rel": "current",
                    "href": "https://api.example.com/products?limit=2&offset=2"
                },
                {
                    "rel": "prev",
                    "href": "https://api.example.com/products?limit=2&offset=0"
                },
                {
                    "rel": "next",
                    "href": "https://api.example.com/products?limit=2&offset=4"
                }
            ]
        }
    )

    model_config = {
        "json_extrema_schema": {
            "examples": [
                {
                    "data": [],
                    "links": [
                        {
                            "rel": "current",
                            "href": "https://api.example.com/products?limit=2&offset=2"
                        },
                        {
                            "rel": "prev",
                            "href": "https://api.example.com/products?limit=2&offset=0"
                        },
                        {
                            "rel": "next",
                            "href": "https://api.example.com/products?limit=2&offset=4"
                        }
                    ]
                }
            ]
        }
    }