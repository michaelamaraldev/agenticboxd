from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parents[1]


def runtime_source() -> str:
    root = project_root()
    paths = [root / "app.py"]
    paths.extend((root / "src").rglob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_project_uses_src_runtime_layout() -> None:
    root = project_root()

    assert {path.name for path in (root / "src" / "agents").glob("*.py")} == {
        "__init__.py",
        "recommendation.py",
        "taste.py",
    }
    assert {path.name for path in (root / "src" / "tools").glob("*.py")} == {
        "__init__.py",
        "analyze_taste.py",
        "get_tmdb_details.py",
    }
    assert all((root / "src" / name).is_file() for name in ("__init__.py", "models.py", "tmdb.py"))
    assert all(not (root / name).exists() for name in ("models.py", "tmdb.py", "agents", "tools"))


def test_project_has_two_agents_and_two_tools_without_langgraph_api() -> None:
    source = runtime_source()
    forbidden = (
        "from langgraph",
        "import langgraph",
        "StateGraph",
        "CompiledStateGraph",
        "ToolRuntime",
        "InvocationContext",
    )

    assert all(token not in source for token in forbidden)
    assert source.count("create_agent(") == 2
    assert source.count("@tool") == 2
    assert "taste_agent.invoke" in source
    assert "recommendation_agent.invoke" in source


def test_project_has_pinned_runtime_requirements() -> None:
    requirements = (project_root() / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert requirements == [
        "httpx==0.28.1",
        "langchain==1.3.18",
        "langchain-ollama==1.1.0",
        "pydantic==2.13.4",
        "python-dotenv==1.2.3",
    ]
