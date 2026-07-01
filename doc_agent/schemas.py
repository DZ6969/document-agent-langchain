"""Pydantic schemas that define the structured output of the extractor."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A single line of an invoice."""

    description: str = Field(description="Product or service description")
    quantity: float | None = Field(default=None, description="Units billed")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total")


class Invoice(BaseModel):
    """Structured representation of an invoice / receipt.

    Every field is optional so the model can leave gaps blank instead of
    hallucinating values that are not present in the document.
    """

    issuer: str | None = Field(default=None, description="Who issued the invoice")
    receiver: str | None = Field(default=None, description="Who the invoice is billed to")
    invoice_id: str | None = Field(default=None, description="Invoice number / folio")
    date: str | None = Field(default=None, description="Issue date, ISO 8601 if possible")
    currency: str | None = Field(default=None, description="Currency code, e.g. MXN, USD")
    subtotal: float | None = Field(default=None)
    tax: float | None = Field(default=None, description="Total tax (e.g. VAT / IVA)")
    total: float | None = Field(default=None)
    line_items: list[LineItem] = Field(
        default_factory=list, description="Individual billed lines"
    )
