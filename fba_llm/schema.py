# fba_llm/schema.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass(frozen=True)
class CanonicalSchema:
    asin: Optional[str] = None
    title: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    review_count: Optional[str] = None
    weight_lbs: Optional[str] = None

    # demand + funnel
    sessions: Optional[str] = None
    units_ordered: Optional[str] = None
    ordered_revenue: Optional[str] = None
    conversion_rate: Optional[str] = None

    # costs
    cogs: Optional[str] = None
    shipping_per_unit: Optional[str] = None
    fba_fees: Optional[str] = None
    ad_spend: Optional[str] = None

    # quality ops
    refunds: Optional[str] = None
    refund_rate: Optional[str] = None


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Common CSV headers you’ll see from Amazon reports + common tools
CANDIDATES: Dict[str, List[str]] = {
    "asin": ["asin", "asins", "parent_asin", "child_asin", "seller_sku_asin"],
    "title": ["title", "product_title", "product_name", "name"],
    "price": ["price", "avg_price", "buy_box_price", "price_usd", "listing_price"],
    "rating": ["rating", "stars", "avg_rating", "review_rating"],
    "review_count": ["review_count", "reviews", "reviews_count", "rating_count", "ratings_total"],
    "weight_lbs": ["weight_lbs", "weight", "shipping_weight", "weight_lb", "ship_weight_lbs"],

    "sessions": ["sessions", "session", "glance_views", "page_views", "traffic"],
    "units_ordered": ["units_ordered", "ordered_units", "units_sold", "quantity_sold"],
    "ordered_revenue": ["ordered_product_sales", "revenue", "sales", "gross_sales", "ordered_sales"],
    "conversion_rate": ["unit_session_percentage", "unit_session_pct", "conversion_rate", "conv_rate"],

    "cogs": ["cogs", "unit_cost", "product_cost", "cost_of_goods"],
    "shipping_per_unit": ["shipping_per_unit", "shipping_cost", "freight_per_unit", "inbound_shipping_unit"],
    "fba_fees": ["fba_fees", "amazon_fees", "fees", "fulfillment_fee", "total_fees"],
    "ad_spend": ["ad_spend", "ppc_spend", "sponsored_spend", "ads_spend"],

    "refunds": ["refunds", "returns", "returned_units"],
    "refund_rate": ["refund_rate", "return_rate", "returns_pct"],
}


def detect_schema(fieldnames: List[str]) -> CanonicalSchema:
    if not fieldnames:
        return CanonicalSchema()

    original = list(fieldnames)
    normalized = [_norm(c) for c in original]
    norm_to_orig = {normalized[i]: original[i] for i in range(len(original))}

    def pick(key: str) -> Optional[str]:
        for cand in CANDIDATES.get(key, []):
            nc = _norm(cand)
            if nc in norm_to_orig:
                return norm_to_orig[nc]
        return None

    return CanonicalSchema(
        asin=pick("asin"),
        title=pick("title"),
        price=pick("price"),
        rating=pick("rating"),
        review_count=pick("review_count"),
        weight_lbs=pick("weight_lbs"),
        sessions=pick("sessions"),
        units_ordered=pick("units_ordered"),
        ordered_revenue=pick("ordered_revenue"),
        conversion_rate=pick("conversion_rate"),
        cogs=pick("cogs"),
        shipping_per_unit=pick("shipping_per_unit"),
        fba_fees=pick("fba_fees"),
        ad_spend=pick("ad_spend"),
        refunds=pick("refunds"),
        refund_rate=pick("refund_rate"),
    )


def missing_fields(schema: CanonicalSchema) -> List[str]:
    out = []
    for k, v in schema.__dict__.items():
        if v is None:
            out.append(k)
    return out