from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


class UIAuditService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.templates_dir = self.project_root / "templates"
        self.static_dir = self.project_root / "static"

    def audit(self, flask_app) -> Dict:
        return {
            "template_routes": self._audit_template_routes(flask_app),
            "admin_actions": self._audit_admin_actions(),
        }

    def _audit_template_routes(self, flask_app) -> Dict:
        endpoint_pattern = re.compile(r"url_for\('([^']+)'")
        valid_endpoints = set(flask_app.view_functions.keys()) | {"static"}
        issues: List[Dict] = []
        checked = 0
        for template_path in sorted(self.templates_dir.rglob("*.html")):
            checked += 1
            content = template_path.read_text(encoding="utf-8", errors="ignore")
            for endpoint in endpoint_pattern.findall(content):
                if endpoint not in valid_endpoints:
                    issues.append({"file": str(template_path), "missing_endpoint": endpoint})
        return {"checked_files": checked, "issues": issues, "healthy": not issues}

    def _audit_admin_actions(self) -> Dict:
        template_paths = [
            self.templates_dir / "admin_dashboard.html",
            self.templates_dir / "admin_models.html",
            self.templates_dir / "admin_crawler.html",
            self.templates_dir / "admin_experiments.html",
        ]
        method_pattern = re.compile(r'@click="([a-zA-Z0-9_]+)')
        js_content = (self.static_dir / "js" / "admin-dashboard.js").read_text(encoding="utf-8", errors="ignore")
        available_methods = set(re.findall(r"async\s+([a-zA-Z0-9_]+)\(", js_content))
        available_methods.update(re.findall(r"([a-zA-Z0-9_]+)\(\)\s*\{", js_content))
        issues: List[Dict] = []
        checked = 0
        for template_path in template_paths:
            if not template_path.exists():
                continue
            checked += 1
            content = template_path.read_text(encoding="utf-8", errors="ignore")
            for method in method_pattern.findall(content):
                if method not in available_methods:
                    issues.append({"file": str(template_path), "missing_method": method})
        return {"checked_files": checked, "issues": issues, "healthy": not issues}
