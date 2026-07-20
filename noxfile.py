"""Nox sessions for geoembed — lint, type-check, test."""

import nox

nox.options.sessions = ["ruff", "pyrefly", "tests"]
nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.12", "3.13"]
LATEST_PYTHON = "3.13"


def _install(session: nox.Session, *extras: str) -> None:
    """Install the project with given extras."""
    deps = ",".join(extras)
    if deps:
        session.install(f".[{deps}]")
    else:
        session.install(".")


def _run(session: nox.Session, *args: str) -> None:
    """Run a command in the session."""
    session.run(*args)


# ── Linting ───────────────────────────────────────────────────────────────────


@nox.session(python=LATEST_PYTHON, tags=["lint"])
def ruff(session: nox.Session) -> None:
    """Lint and format check with ruff."""
    session.install("ruff==0.11.5")
    _run(session, "ruff", "check", "src/", "tests/")
    _run(session, "ruff", "format", "--check", "src/", "tests/")


# ── Type Checking ─────────────────────────────────────────────────────────────


@nox.session(python=LATEST_PYTHON, tags=["typecheck"])
def pyrefly(session: nox.Session) -> None:
    """Verify types using pyrefly."""
    args = session.posargs or ("src/geoembed",)
    _install(session, "spatial", "cog", "stac")
    session.install("pyrefly>=0.61.0")
    _run(session, "pyrefly", "check", *args)


# ── Testing ───────────────────────────────────────────────────────────────────


@nox.session(python=PYTHON_VERSIONS, tags=["test"])
def tests(session: nox.Session) -> None:
    """Run the test suite with pytest."""
    _install(session, "spatial", "cog", "stac", "models")
    session.install("pytest", "pytest-cov")
    args = session.posargs or [
        "tests/",
        "-v",
        "--tb=short",
        "-m",
        "not slow and not gpu",
    ]
    _run(session, "pytest", *args)


@nox.session(python=LATEST_PYTHON, tags=["test"])
def tests_slow(session: nox.Session) -> None:
    """Run slow tests (model loading, full inference)."""
    _install(session, "spatial", "cog", "stac", "models")
    session.install("pytest", "pytest-cov")
    _run(session, "pytest", "tests/", "-v", "-m", "slow")


# ── Coverage ──────────────────────────────────────────────────────────────────


@nox.session(python=LATEST_PYTHON, tags=["test"])
def coverage(session: nox.Session) -> None:
    """Run tests with coverage report."""
    _install(session, "spatial", "cog", "stac", "models")
    session.install("pytest", "pytest-cov")
    _run(
        session,
        "pytest",
        "tests/",
        "--cov=src/geoembed",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-m",
        "not slow and not gpu",
    )
