from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fba_llm.analysis import Assumptions
from fba_llm.ingest import build_combined_facts_block
from fba_llm.image_ingest import build_png_facts_block
from fba_llm.llm_backend import generate_text
from fba_llm.qa_prompts import (
    build_followup_question_prompt,
    build_screening_summary_prompt,
)


@dataclass
class AnalysisSession:
    question: str
    assumptions: Assumptions
    metrics_csv: Optional[Path] = None
    reviews_txt: Optional[Path] = None
    png_path: Optional[Path] = None
    deep_review_analysis: bool = False

    facts_block: str = ""
    screening_summary: str = ""
    image_facts_block: str = ""
    built: bool = field(default=False, init=False)
    history: List[Dict[str, str]] = field(default_factory=list, init=False)

    def build(self) -> None:
        base_facts = build_combined_facts_block(
            metrics_csv=self.metrics_csv,
            reviews_txt=self.reviews_txt,
            deep_review_analysis=self.deep_review_analysis,
            lead_time_days=self.assumptions.lead_time_days,
            target_margin_pct=self.assumptions.target_margin_pct,
            ad_spend_per_unit_usd=self.assumptions.ad_spend_per_unit,
        )

        extra_sections: list[str] = []

        if self.png_path is not None:
            self.image_facts_block = build_png_facts_block(self.png_path)
            extra_sections.append(f"=== IMAGES ===\n{self.image_facts_block}")

        self.facts_block = base_facts
        if extra_sections:
            self.facts_block = self.facts_block + "\n\n" + "\n\n".join(extra_sections)

        self.built = True

    def ensure_built(self) -> None:
        if not self.built or not self.facts_block.strip():
            self.build()

    def generate_screening_summary(self) -> str:
        self.ensure_built()

        # Reset history on a fresh screening run
        self.history = []

        prompt = build_screening_summary_prompt(self.question, self.facts_block)
        out = generate_text(prompt, max_tokens=600, temperature=0.1, timeout_s=90)
        self.screening_summary = (out or "").strip()

        self.history.append({"role": "user", "content": self.question})
        self.history.append({"role": "assistant", "content": self.screening_summary})

        return self.screening_summary

    def ask(self, user_question: str) -> str:
        self.ensure_built()

        prompt = build_followup_question_prompt(user_question, self.facts_block, self.history)
        out = generate_text(prompt, max_tokens=600, temperature=0.1, timeout_s=90)
        answer = (out or "").strip()

        self.history.append({"role": "user", "content": user_question})
        self.history.append({"role": "assistant", "content": answer})

        return answer