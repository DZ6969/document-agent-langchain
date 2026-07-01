from doc_agent.schemas import Invoice, LineItem


def test_invoice_defaults_are_empty():
    inv = Invoice()
    assert inv.total is None
    assert inv.line_items == []


def test_invoice_with_line_items_round_trips():
    inv = Invoice(
        issuer="Acme",
        currency="MXN",
        total=100.0,
        line_items=[LineItem(description="Widget", quantity=2, unit_price=50.0, amount=100.0)],
    )
    data = inv.model_dump()
    assert data["total"] == 100.0
    assert data["line_items"][0]["description"] == "Widget"
