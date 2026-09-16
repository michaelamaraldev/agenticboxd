import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.observability import (
    RunRecorder,
    artifact_fingerprint,
    build_manifest,
    configure_langsmith,
    safe_trace_metadata,
    trace_call,
)


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_fingerprint_is_stable_and_changes_with_content(self):
        first = self.root / "a.txt"
        second = self.root / "b.txt"
        first.write_text("alpha", encoding="utf-8")
        second.write_text("beta", encoding="utf-8")
        original = artifact_fingerprint([first, second])
        self.assertEqual(original, artifact_fingerprint([second, first]))
        self.assertRegex(original, r"^sha256:[0-9a-f]{64}$")
        second.write_text("changed", encoding="utf-8")
        self.assertNotEqual(original, artifact_fingerprint([first, second]))

    def test_manifest_contains_version_identifiers_without_payloads(self):
        data_file = self.root / "watchlist.csv"
        cache_file = self.root / "tmdb_cache.json"
        prompt_file = self.root / "recommendation.py"
        tool_file = self.root / "get_tmdb_details.py"
        lock_file = self.root / "requirements.lock.txt"
        data_file.write_text("private-watchlist-title", encoding="utf-8")
        cache_file.write_text('{"overview":"private-overview"}', encoding="utf-8")
        prompt_file.write_text("private prompt text", encoding="utf-8")
        tool_file.write_text("tool schema", encoding="utf-8")
        lock_file.write_text("package==1.0", encoding="utf-8")
        manifest = build_manifest(
            run_id="run-123",
            git_revision="abc123",
            code_dirty=True,
            data_paths=[data_file],
            tmdb_cache_path=cache_file,
            prompt_paths=[prompt_file],
            tool_paths=[tool_file],
            dependency_lock_path=lock_file,
            model_name="qwen3.5:4b",
            model_digest="sha256:model",
            model_config={"temperature": 0},
            python_version="3.12.14",
            ollama_version="0.34.0",
        )
        expected = {
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
            "model_config",
            "python_version",
            "ollama_version",
            "schema_version",
        }
        self.assertEqual(expected, set(manifest))
        serialized = json.dumps(manifest)
        for private_value in (
            "private-watchlist-title",
            "private-overview",
            "private prompt text",
            "tool schema",
        ):
            self.assertNotIn(private_value, serialized)


class LocalObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {
            "run_id": "run-123",
            "git_revision": "abc123",
            "code_dirty": False,
            "data_version": "sha256:data",
            "tmdb_cache_version": "sha256:tmdb",
            "prompt_version": "sha256:prompt",
            "agent_version": "sha256:agent",
            "dependency_lock_hash": "sha256:lock",
            "model_name": "qwen3.5:4b",
            "model_digest": "sha256:model",
            "model_config": {"temperature": 0},
            "python_version": "3.12.14",
            "ollama_version": "0.34.0",
            "schema_version": 1,
        }

    def test_recorder_writes_manifest_and_sanitized_jsonl_events(self):
        recorder = RunRecorder(self.root, self.manifest)
        recorder.record(
            "recommendation_finished",
            status="ok",
            duration_seconds=1.25,
            model_calls=3,
            tool_calls=4,
            input_tokens=120,
            output_tokens=40,
            taste_cache_hit=True,
            recommendation_count=1,
            request="private request",
            response="private response",
            api_key="secret-key",
        )
        run_dir = self.root / "run-123"
        self.assertEqual(self.manifest, json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")))
        event = json.loads((run_dir / "events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual("recommendation_finished", event["event"])
        self.assertEqual("ok", event["status"])
        self.assertEqual(3, event["model_calls"])
        serialized = json.dumps(event)
        for forbidden in ("private request", "private response", "secret-key", "api_key"):
            self.assertNotIn(forbidden, serialized)

    def test_trace_metadata_uses_only_version_allowlist(self):
        source = {
            **self.manifest,
            "request": "private request",
            "response": "private response",
            "api_key": "secret-key",
        }
        metadata = safe_trace_metadata(source, environment="demo")
        self.assertEqual(
            {
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
                "environment",
                "schema_version",
            },
            set(metadata),
        )
        self.assertEqual("demo", metadata["environment"])


class LangSmithConfigurationTests(unittest.TestCase):
    def test_demo_mode_can_show_inputs_and_outputs(self):
        environment = {
            "APP_ENV": "demo",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_HIDE_INPUTS": "false",
            "LANGSMITH_HIDE_OUTPUTS": "false",
        }
        settings = configure_langsmith(environment)
        self.assertTrue(settings.tracing_enabled)
        self.assertTrue(settings.payloads_visible)
        self.assertEqual("false", environment["LANGSMITH_HIDE_INPUTS"])
        self.assertEqual("false", environment["LANGSMITH_HIDE_OUTPUTS"])

    def test_non_demo_mode_forces_payloads_hidden(self):
        environment = {
            "APP_ENV": "production",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_HIDE_INPUTS": "false",
            "LANGSMITH_HIDE_OUTPUTS": "false",
        }
        settings = configure_langsmith(environment)
        self.assertTrue(settings.tracing_enabled)
        self.assertFalse(settings.payloads_visible)
        self.assertEqual("true", environment["LANGSMITH_HIDE_INPUTS"])
        self.assertEqual("true", environment["LANGSMITH_HIDE_OUTPUTS"])

    def test_disabled_tracing_returns_operation_result_unchanged(self):
        operation = Mock(return_value={"recommendation": "result"})
        result = trace_call(
            "recommendation_request",
            operation,
            inputs={"request": "private request"},
            metadata={"run_id": "run-123"},
            environment={"LANGSMITH_TRACING": "false"},
        )
        self.assertEqual({"recommendation": "result"}, result)
        operation.assert_called_once_with()
