from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict

from flask import Blueprint, abort, current_app, jsonify, render_template, request, session

from .crawler.core import STATUS_FILE
from .crawler.jobs import build_behavior_dataset, crawl_movie_comments, crawl_top_movies
from .evaluation import evaluate_recommenders
from .query import DB_PATH
from .rl.features import get_movie_feature_map
from .rl.local_ppo import (
    ensure_bootstrap_model,
    rollback_model_version,
    status_payload,
    train_pending_batch_if_ready,
)
from services.experiment_service import ExperimentService
from services.system_service import SystemService
from services.ui_audit_service import UIAuditService


admin_bp = Blueprint("admin_bp", __name__)
_JOB_STATE: Dict[str, Dict] = {}
_experiment_service = ExperimentService()
_system_service = SystemService()
_ui_audit_service = UIAuditService(Path(__file__).resolve().parents[1])


def is_admin_user() -> bool:
    email = (session.get("email") or "").lower()
    return bool(email) and email in current_app.config["ADMIN_EMAILS"]


def admin_required():
    if not session.get("email"):
        abort(401)
    if not is_admin_user():
        abort(403)


def _read_crawler_status() -> Dict:
    if not STATUS_FILE.exists():
        return {}
    return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


def _system_counts() -> Dict:
    return _system_service.overview_counts(
        db_path=str(DB_PATH),
        movie_feature_cache_size=len(get_movie_feature_map()),
    )


def _overview_payload() -> Dict:
    return {
        "system": _system_counts(),
        "rl": status_payload(),
        "crawler": _read_crawler_status(),
        "jobs": _JOB_STATE,
        "evaluation": evaluate_recommenders(),
        "experiments": _experiment_service.build_trend_payload(limit=10),
        "ui_audit": _ui_audit_service.audit(current_app),
    }


def _run_async_job(job_name: str, target, **kwargs) -> Dict:
    if _JOB_STATE.get(job_name, {}).get("status") == "running":
        return {"accepted": False, "message": "任务正在执行中"}

    _JOB_STATE[job_name] = {"status": "running", "params": kwargs}
    app_obj = current_app._get_current_object()

    def runner():
        try:
            result = target(**kwargs)
            _JOB_STATE[job_name] = {"status": "completed", "result": result}
        except Exception as exc:  # pragma: no cover - background task path
            app_obj.logger.exception("admin job failed: %s", job_name)
            _JOB_STATE[job_name] = {"status": "failed", "message": str(exc)}

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return {"accepted": True, "message": "任务已启动", "job": job_name}


@admin_bp.route("/admin/dashboard")
def admin_dashboard():
    admin_required()
    return render_template("admin_dashboard.html")


@admin_bp.route("/admin/models")
def admin_models():
    admin_required()
    return render_template("admin_models.html")


@admin_bp.route("/admin/crawler")
def admin_crawler():
    admin_required()
    return render_template("admin_crawler.html")


@admin_bp.route("/admin/experiments")
def admin_experiments():
    admin_required()
    return render_template("admin_experiments.html")


@admin_bp.route("/api/v1/admin/overview")
def api_admin_overview():
    admin_required()
    return jsonify({"code": 200, "data": _overview_payload()})


@admin_bp.route("/api/v1/admin/evaluation")
def api_admin_evaluation():
    admin_required()
    return jsonify({"code": 200, "data": evaluate_recommenders()})


@admin_bp.route("/api/v1/admin/experiments")
def api_admin_experiments():
    admin_required()
    limit = request.args.get("limit", default=20, type=int)
    return jsonify({"code": 200, "data": _experiment_service.build_trend_payload(limit=limit)})


@admin_bp.route("/api/v1/admin/experiments/run", methods=["POST"])
def api_admin_experiments_run():
    admin_required()
    payload = request.get_json(silent=True) or {}
    top_k = int(payload.get("top_k", 5))
    note = str(payload.get("note", "")).strip()
    result = _experiment_service.build_snapshot(top_k=top_k, note=note)
    return jsonify({"code": 200, "data": result, "msg": "实验快照已生成"})


@admin_bp.route("/api/v1/admin/models")
def api_admin_models():
    admin_required()
    return jsonify({"code": 200, "data": status_payload()})


@admin_bp.route("/api/v1/admin/models/bootstrap", methods=["POST"])
def api_admin_model_bootstrap():
    admin_required()
    result = ensure_bootstrap_model()
    return jsonify({"code": 200, "data": result, "msg": "Bootstrap 完成"})


@admin_bp.route("/api/v1/admin/models/train", methods=["POST"])
def api_admin_model_train():
    admin_required()
    force = bool((request.get_json(silent=True) or {}).get("force", False))
    result = train_pending_batch_if_ready(force=force)
    return jsonify({"code": 200, "data": result, "msg": "训练已执行"})


@admin_bp.route("/api/v1/admin/models/<version_tag>/rollback", methods=["POST"])
def api_admin_model_rollback(version_tag: str):
    admin_required()
    result = rollback_model_version(version_tag)
    return jsonify({"code": 200 if result else 404, "ok": result, "version_tag": version_tag}), (200 if result else 404)


@admin_bp.route("/api/v1/admin/crawler/status")
def api_admin_crawler_status():
    admin_required()
    return jsonify({"code": 200, "data": {"crawler": _read_crawler_status(), "jobs": _JOB_STATE}})


@admin_bp.route("/api/v1/admin/crawler/run", methods=["POST"])
def api_admin_crawler_run():
    admin_required()
    payload = request.get_json(silent=True) or {}
    job = payload.get("job")
    if job == "movies":
        result = _run_async_job("movies", crawl_top_movies, pages=int(payload.get("pages", 8)))
    elif job == "comments":
        result = _run_async_job(
            "comments",
            crawl_movie_comments,
            pages_per_movie=int(payload.get("pages_per_movie", 3)),
            limit_movies=int(payload["limit_movies"]) if payload.get("limit_movies") else None,
        )
    elif job == "behaviors":
        result = _run_async_job(
            "behaviors",
            build_behavior_dataset,
            user_count=int(payload.get("user_count", 60)),
            min_behaviors=int(payload.get("min_behaviors", 8)),
            max_behaviors=int(payload.get("max_behaviors", 16)),
        )
    else:
        return jsonify({"code": 400, "msg": "不支持的任务类型"}), 400
    return jsonify({"code": 200, "data": result})
