from __future__ import annotations

from typing import Dict, List

from repositories.experiment_repository import ExperimentRepository
from schemas.common import ExperimentMetricSchema, ExperimentRunSchema
from myutils.evaluation import evaluate_recommenders


class ExperimentService:
    def __init__(self) -> None:
        self.repository = ExperimentRepository()

    def build_snapshot(self, top_k: int = 5, note: str = "") -> Dict:
        payload = evaluate_recommenders(top_k=top_k)
        metrics = [ExperimentMetricSchema(**metric) for metric in payload.get("metrics", [])]
        run = ExperimentRunSchema(
            run_name=f"offline_eval_top{top_k}",
            sample_users=int(payload.get("sample_users", 0)),
            top_k=int(payload.get("top_k", top_k)),
            metrics=metrics,
            note=note,
        )
        return self.repository.save_run(run.run_name, run.to_dict(), note=note)

    def list_snapshots(self, limit: int = 20) -> List[Dict]:
        return self.repository.list_runs(limit=limit)

    def build_trend_payload(self, limit: int = 20) -> Dict:
        runs = self.list_snapshots(limit=limit)
        trends = {}
        for run in reversed(runs):
            for metric in run["metrics"].get("metrics", []):
                algorithm = metric["algorithm"]
                trends.setdefault(
                    algorithm,
                    {
                        "algorithm": algorithm,
                        "precision": [],
                        "recall": [],
                        "ndcg": [],
                        "coverage": [],
                        "diversity": [],
                        "labels": [],
                    },
                )
                trends[algorithm]["precision"].append(metric["precision_at_k"])
                trends[algorithm]["recall"].append(metric["recall_at_k"])
                trends[algorithm]["ndcg"].append(metric["ndcg_at_k"])
                trends[algorithm]["coverage"].append(metric["coverage"])
                trends[algorithm]["diversity"].append(metric["diversity"])
                trends[algorithm]["labels"].append(run["created_at"])
        latest_run = runs[0] if runs else None
        latest_metrics = latest_run["metrics"].get("metrics", []) if latest_run else []
        return {
            "runs": runs,
            "trends": list(trends.values()),
            "timeline_labels": [run["created_at"] for run in reversed(runs)],
            "latest_metrics": latest_metrics,
        }
