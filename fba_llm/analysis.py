# fba_llm/analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class Assumptions:
    lead_time_days: int = 60
    target_margin_pct: float = 30.0
    ad_spend_per_unit: float = 0.0  # optional if you don’t know
    startup_budget: float = 0.0     # optional


@dataclass(frozen=True)
class UnitEconomics:
    price: Optional[float]
    landed_cost: Optional[float]
    fba_fees: Optional[float]
    ad_spend_per_unit: float
    profit_per_unit: Optional[float]
    margin_pct: Optional[float]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_0_100(x: float) -> int:
    return int(round(clamp(x, 0.0, 100.0)))


def compute_unit_econ(
    *,
    price: Optional[float],
    cogs: Optional[float],
    shipping_per_unit: Optional[float],
    fba_fees: Optional[float],
    assumptions: Assumptions,
) -> UnitEconomics:
    if price is None:
        return UnitEconomics(price=None, landed_cost=None, fba_fees=fba_fees, ad_spend_per_unit=assumptions.ad_spend_per_unit,
                             profit_per_unit=None, margin_pct=None)

    landed_cost = None
    if cogs is not None or shipping_per_unit is not None:
        landed_cost = (cogs or 0.0) + (shipping_per_unit or 0.0)

    profit = None
    margin = None
    if landed_cost is not None and fba_fees is not None:
        profit = price - landed_cost - fba_fees - assumptions.ad_spend_per_unit
        if price != 0:
            margin = (profit / price) * 100.0

    return UnitEconomics(
        price=price,
        landed_cost=landed_cost,
        fba_fees=fba_fees,
        ad_spend_per_unit=assumptions.ad_spend_per_unit,
        profit_per_unit=profit,
        margin_pct=margin,
    )


def compute_scores(
    *,
    price_spread_pct: Optional[float],
    rating_spread: Optional[float],
    review_dominance_ratio: Optional[float],
    shipping_complexity_label: str,
    quality_consistency_label: str,
    price_competition_label: str,
    review_concentration_label: str,
    sample_confidence_label: str,
    unit_econ: UnitEconomics,
) -> Dict[str, int]:
    # price war risk: high spread often means price segmentation/war; label already helps
    pwr = 50.0
    if price_spread_pct is not None:
        # 0% -> 20, 60% -> ~85, 100% -> 95
        pwr = 20.0 + (clamp(price_spread_pct, 0.0, 100.0) / 100.0) * 75.0
    if "HIGH" in price_competition_label.upper():
        pwr += 10
    if "LOW" in price_competition_label.upper():
        pwr -= 10

    # entry barrier: dominant reviews + concentration raises barrier
    eb = 40.0
    if review_dominance_ratio is not None:
        # 1.0 -> low barrier, 3.0 -> high barrier
        eb = 25.0 + clamp((review_dominance_ratio - 1.0) / 2.5, 0.0, 1.0) * 70.0
    if "DOMINANT" in review_concentration_label.upper():
        eb += 10

    # quality risk: rating spread + volatility label
    qr = 35.0
    if rating_spread is not None:
        # 0.0 -> 15, 1.0 -> 85
        qr = 15.0 + clamp(rating_spread / 1.2, 0.0, 1.0) * 75.0
    if "VOLATILE" in quality_consistency_label.upper():
        qr += 10

    # ops/shipping: you already label light/medium/heavy; light lowers risk
    ops = 50.0
    lab = shipping_complexity_label.upper()
    if "LIGHT" in lab:
        ops = 20.0
    elif "MEDIUM" in lab:
        ops = 45.0
    elif "HEAVY" in lab:
        ops = 75.0

    # unit economics viability if available
    ue = 50.0
    if unit_econ.margin_pct is not None:
        # margin 0% -> 10, 30% -> 70, 50% -> 90
        ue = 10.0 + clamp(unit_econ.margin_pct / 50.0, 0.0, 1.0) * 80.0

    # sample confidence penalizes reliability
    conf_penalty = 0.0
    if "LOW" in sample_confidence_label.upper():
        conf_penalty = 12.0

    # overall viability: high risks reduce, good unit econ improves
    overall = (
        0.40 * ue
        + 0.10 * (100.0 - ops)
        + 0.15 * (100.0 - pwr)
        + 0.20 * (100.0 - eb)
        + 0.15 * (100.0 - qr)
        - conf_penalty
    )

    return {
        "PRICE_WAR_RISK_SCORE": score_0_100(pwr),
        "ENTRY_BARRIER_SCORE": score_0_100(eb),
        "QUALITY_RISK_SCORE": score_0_100(qr),
        "OPS_RISK_SCORE": score_0_100(ops),
        "UNIT_ECON_SCORE": score_0_100(ue),
        "OVERALL_VIABILITY_SCORE": score_0_100(overall),
    }


def format_structured_analysis(
    *,
    assumptions: Assumptions,
    unit_econ: UnitEconomics,
    scores: Dict[str, int],
    missing: List[str],
) -> str:
    # Everything numeric is computed here
    lines = []
    lines.append("STRUCTURED_ANALYSIS (COMPUTED; DO NOT CHANGE):")
    for k in ["OVERALL_VIABILITY_SCORE", "UNIT_ECON_SCORE", "PRICE_WAR_RISK_SCORE", "ENTRY_BARRIER_SCORE", "QUALITY_RISK_SCORE", "OPS_RISK_SCORE"]:
        if k in scores:
            lines.append(f"{k}: {scores[k]}/100")

    lines.append("")
    lines.append("ASSUMPTIONS (USED IN COMPUTATIONS):")
    lines.append(f"LEAD_TIME_DAYS: {assumptions.lead_time_days}")
    lines.append(f"TARGET_MARGIN_PCT: {assumptions.target_margin_pct:.2f}")
    lines.append(f"AD_SPEND_PER_UNIT: {assumptions.ad_spend_per_unit:.2f}")
    lines.append(f"STARTUP_BUDGET: {assumptions.startup_budget:.2f}")

    lines.append("")
    lines.append("UNIT_ECONOMICS (COMPUTED; MAY BE INSUFFICIENT):")
    lines.append(f"PRICE: {unit_econ.price if unit_econ.price is not None else 'INSUFFICIENT DATA'}")
    lines.append(f"LANDED_COST: {unit_econ.landed_cost if unit_econ.landed_cost is not None else 'INSUFFICIENT DATA'}")
    lines.append(f"FBA_FEES: {unit_econ.fba_fees if unit_econ.fba_fees is not None else 'INSUFFICIENT DATA'}")
    lines.append(f"PROFIT_PER_UNIT: {unit_econ.profit_per_unit if unit_econ.profit_per_unit is not None else 'INSUFFICIENT DATA'}")
    lines.append(f"MARGIN_PCT: {unit_econ.margin_pct if unit_econ.margin_pct is not None else 'INSUFFICIENT DATA'}")

    lines.append("")
    lines.append("MISSING_FIELDS (CANONICAL):")
    if missing:
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append("- (none)")

    return "\n".join(lines)