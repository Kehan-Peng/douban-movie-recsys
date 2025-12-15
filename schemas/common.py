from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RecommendationItemSchema:
    id: int
    title: str
    rate: float
    cover_url: Optional[str] = None
    reason: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentMetricSchema:
    algorithm: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    coverage: float
    diversity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRunSchema:
    run_name: str
    sample_users: int
    top_k: int
    metrics: List[ExperimentMetricSchema] = field(default_factory=list)
    note: str = ""
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = [metric.to_dict() for metric in self.metrics]
        return payload
