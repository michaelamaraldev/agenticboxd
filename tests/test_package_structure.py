from __future__ import annotations

import tomllib
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parents[1]


def test_project_uses_flat_agents_and_tools_layout() -> None:
    root = project_root()

    assert not (root / "src" / "cine").exists()
    assert {path.name for path in (root / "agents").glob("*.py")} == {
        "__init__.py",
        "recommendation.py",
        "taste.py",
    }
    assert {path.name for path in (root / "tools").glob("*.py")} == {
        "__init__.py",
        "analyze_taste.py",
        "get_tmdb_details.py",
    }
    assert all((root / name).is_file() for name in ("app.py", "models.py", "tmdb.py"))


def test_project_has_two_agents_and_two_tools_without_langgraph_api() -> None:
    root = project_root()
    files = [root / "app.py", root / "models.py", root / "tmdb.py"]
    files.extend((root / "agents").glob("*.py"))
    files.extend((root / "tools").glob("*.py"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

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


def test_project_is_an_unpacked_uv_application() -> None:
    configuration = tomllib.loads((project_root() / "pyproject.toml").read_text(encoding="utf-8"))

    assert configuration["tool"]["uv"]["package"] is False
    assert "build-system" not in configuration
    assert "scripts" not in configuration["project"]
    dependencies = configuration["project"]["dependencies"]
    assert {item.split(">=")[0] for item in dependencies} == {
        "httpx",
        "langchain",
        "langchain-ollama",
        "pydantic",
        "python-dotenv",
    }
