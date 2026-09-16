from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import httpx
from langsmith import traceable

T = TypeVar("T")

TRACE_METADATA_FIELDS = (
    "run_id",
    "git_revision",
    "code_dirty",
    "data_version",
    "tmdb_cache_version",
    "prompt_version",
    "agent_version",
    "dependency_lock_hash",
    "model_name",
    "model_digest",
    "schema_version",
)

EVENT_FIELDS = {
    "status",
    "duration_seconds",
    "model_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "taste_cache_hit",
    "recommendation_count",
    "error_type",
}


@dataclass(frozen=True, slots=True)
class LangSmithSettings:
    tracing_enabled: bool
    payloads_visible: bool
    project: str
    environment: str


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def configure_langsmith(
    environment: MutableMapping[str, str] | None = None,
) -> LangSmithSettings:
    values = environment if environment is not None else os.environ
    app_environment = values.get("APP_ENV", "local").strip().lower() or "local"
    tracing_enabled = _enabled(
        values.get("LANGSMITH_TRACING", values.get("LANGCHAIN_TRACING_V2", "false"))
    )
    payloads_visible = tracing_enabled and app_environment == "demo"
    if tracing_enabled:
        visibility = "false" if payloads_visible else "true"
        values["LANGSMITH_HIDE_INPUTS"] = visibility
        values["LANGSMITH_HIDE_OUTPUTS"] = visibility
    values.setdefault("LANGSMITH_PROJECT", "letterboxd-recommender")
    return LangSmithSettings(
        tracing_enabled=tracing_enabled,
        payloads_visible=payloads_visible,
        project=values["LANGSMITH_PROJECT"],
        environment=app_environment,
    )


def artifact_fingerprint(paths: Sequence[Path]) -> str:
    """Build a stable content fingerprint for a collection of artifacts."""
    digest = hashlib.sha256()
    files = [Path(item).resolve() for item in paths]
    if not files:
        return f"sha256:{digest.hexdigest()}"
    common = Path(os.path.commonpath([str(path) for path in files]))
    if len(files) == 1 or common.is_file():
        common = common.parent
    for path in sorted(files, key=lambda item: item.relative_to(common).as_posix()):
        content = path.read_bytes()
        name = path.relative_to(common).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def build_manifest(
    *,
    run_id: str,
    git_revision: str,
    code_dirty: bool,
    data_paths: Sequence[Path],
    tmdb_cache_path: Path,
    prompt_paths: Sequence[Path],
    tool_paths: Sequence[Path],
    dependency_lock_path: Path,
    model_name: str,
    model_digest: str,
    model_config: Mapping[str, object],
    python_version: str,
    ollama_version: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "git_revision": git_revision,
        "code_dirty": code_dirty,
        "data_version": artifact_fingerprint(data_paths),
        "tmdb_cache_version": artifact_fingerprint([tmdb_cache_path]),
        "prompt_version": artifact_fingerprint(prompt_paths),
        "agent_version": artifact_fingerprint([*prompt_paths, *tool_paths]),
        "dependency_lock_hash": artifact_fingerprint([dependency_lock_path]),
        "model_name": model_name,
        "model_digest": model_digest,
        "model_config": dict(model_config),
        "python_version": python_version,
        "ollama_version": ollama_version,
        "schema_version": 1,
    }


def safe_trace_metadata(
    source: Mapping[str, object], *, environment: str
) -> dict[str, object]:
    metadata = {field: source[field] for field in TRACE_METADATA_FIELDS if field in source}
    metadata["environment"] = environment
    return metadata


class RunRecorder:
    def __init__(self, runs_root: Path, manifest: Mapping[str, object]) -> None:
        self.manifest = dict(manifest)
        self.run_dir = Path(runs_root) / str(self.manifest["run_id"])
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(self, event: str, **values: object) -> None:
        payload: dict[str, object] = {
            "event": event,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        payload.update({key: value for key, value in values.items() if key in EVENT_FIELDS})
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def trace_call(
    name: str,
    operation: Callable[[], T],
    *,
    inputs: Mapping[str, object],
    metadata: Mapping[str, object],
    environment: MutableMapping[str, str] | None = None,
) -> T:
    settings = configure_langsmith(environment)
    if not settings.tracing_enabled:
        return operation()

    @traceable(
        name=name,
        run_type="chain",
        metadata=dict(metadata),
        project_name=settings.project,
        enabled=True,
    )
    def traced(payload: Mapping[str, object]) -> T:
        return operation()

    return traced(dict(inputs))


def repository_version(root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return revision, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


def ollama_version_info(base_url: str, model_name: str) -> tuple[str, str]:
    try:
        with httpx.Client(base_url=base_url, timeout=5) as client:
            version = str(client.get("/api/version").raise_for_status().json()["version"])
            models = client.get("/api/tags").raise_for_status().json().get("models", [])
        match = next(
            (
                model
                for model in models
                if model.get("name") == model_name or model.get("model") == model_name
            ),
            None,
        )
        digest = str(match.get("digest", "unknown")) if match else "unknown"
        return version, digest
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return "unknown", "unknown"


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


class RecommendationObserver:
    def __init__(
        self,
        *,
        runs_root: Path,
        manifest_factory: Callable[[str], Mapping[str, object]],
        environment: MutableMapping[str, str] | None = None,
    ) -> None:
        self.runs_root = runs_root
        self.manifest_factory = manifest_factory
        self.environment = environment if environment is not None else os.environ

    def experiment_metadata(self) -> dict[str, object]:
        manifest = dict(self.manifest_factory("experiment"))
        settings = configure_langsmith(self.environment)
        metadata = safe_trace_metadata(manifest, environment=settings.environment)
        metadata["models"] = [manifest["model_name"]]
        metadata["prompts"] = [manifest["prompt_version"]]
        metadata["tools"] = [manifest["agent_version"]]
        return metadata

    def observe(self, request: str, operation: Callable[[], T]) -> T:
        """Trace one recommendation and persist its local operational metrics."""
        manifest = dict(self.manifest_factory(new_run_id()))
        recorder = RunRecorder(self.runs_root, manifest)
        settings = configure_langsmith(self.environment)
        metadata = safe_trace_metadata(manifest, environment=settings.environment)
        started = time.perf_counter()
        recorder.record("recommendation_started", status="running")
        try:
            result = trace_call(
                "recommendation_request",
                operation,
                inputs={"request": request},
                metadata=metadata,
                environment=self.environment,
            )
        except Exception as error:
            recorder.record(
                "recommendation_failed",
                status="error",
                duration_seconds=time.perf_counter() - started,
                error_type=type(error).__name__,
            )
            raise
        metrics = getattr(result, "metrics", {})
        tokens = metrics.get("tokens", {}) if isinstance(metrics, Mapping) else {}
        recommendations = getattr(result, "recommendations", ())
        recorder.record(
            "recommendation_finished",
            status="ok",
            duration_seconds=time.perf_counter() - started,
            model_calls=metrics.get("model_calls", 0),
            tool_calls=metrics.get("tool_calls", 0),
            input_tokens=tokens.get("input_tokens", 0),
            output_tokens=tokens.get("output_tokens", 0),
            total_tokens=tokens.get("total_tokens", 0),
            taste_cache_hit=metrics.get("taste_cache_hit", False),
            recommendation_count=len(recommendations),
        )
        return result


def runtime_manifest_factory(
    *,
    root: Path,
    data_dir: Path,
    cache_path: Path,
    model_name: str,
    model_config: Mapping[str, object],
    ollama_version: str,
    model_digest: str,
) -> Callable[[str], Mapping[str, object]]:
    git_revision, code_dirty = repository_version(root)
    data_paths = sorted(data_dir.rglob("*.csv"))
    prompt_paths = [
        root / "src" / "agents" / "taste.py",
        root / "src" / "agents" / "history.py",
        root / "src" / "agents" / "recommendation.py",
    ]
    tool_paths = [
        root / "src" / "models.py",
        *sorted((root / "src" / "tools").glob("*.py")),
    ]
    dependency_path = root / "requirements.lock.txt"
    if not dependency_path.exists():
        dependency_path = root / "requirements.txt"

    def factory(run_id: str) -> Mapping[str, object]:
        return build_manifest(
            run_id=run_id,
            git_revision=git_revision,
            code_dirty=code_dirty,
            data_paths=data_paths,
            tmdb_cache_path=cache_path,
            prompt_paths=prompt_paths,
            tool_paths=tool_paths,
            dependency_lock_path=dependency_path,
            model_name=model_name,
            model_digest=model_digest,
            model_config=model_config,
            python_version=platform.python_version(),
            ollama_version=ollama_version,
        )

    return factory
