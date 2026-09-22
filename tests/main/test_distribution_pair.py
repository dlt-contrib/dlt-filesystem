"""The two distributions this repository builds have to stay a matched pair.

`omniload` depends on `dlt-filesystem` without a version specifier, and that is a
measured decision rather than an omission. Two things make a pin unenforceable here:

- The versions are computed by versioningit from the git tag at build time, so an
  exact one cannot be written into a static `dependencies` list at all.
- A specifier on a workspace member is not checked locally. `uv pip compile` resolves
  `dlt-filesystem==99.0.0` against the local tree and exits 0, so a wrong pin would
  reach PyPI with every local check green.

What "lockstep" claims is that one tag produces one version for both projects. That is
asserted here at its cause, the two `[tool.versioningit]` declarations, rather than by
comparing installed versions: an installed version records the commit each project was
*installed* at, so two editable installs made at different commits differ without
anything being wrong.
"""

import importlib.metadata as metadata
import pathlib
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 only
    tomllib = pytest.importorskip("tomli")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECTS = {
    "omniload": REPO_ROOT / "pyproject.toml",
    "dlt-filesystem": REPO_ROOT / "packages" / "dlt-filesystem" / "pyproject.toml",
}


def _pyproject(name: str) -> dict:
    path = PROJECTS[name]
    assert path.is_file(), f"no project file at {path}"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_both_projects_derive_their_version_the_same_way():
    """One git tag, one version, whichever project is asked.

    Equality of the whole table rather than of the tag pattern alone, because any
    difference in method or default reaches the version string.
    """
    tables = {name: _pyproject(name)["tool"]["versioningit"] for name in PROJECTS}
    assert tables["omniload"], "omniload declares no versioningit configuration"
    assert tables["omniload"] == tables["dlt-filesystem"]


def test_both_projects_let_versioningit_supply_the_version():
    """An equal `[tool.versioningit]` table is inert unless versioningit is wired in.

    Without this, either project could move to a different dynamic-version provider,
    keep the matching table nobody reads any more, and diverge with the assertion above
    still green.
    """
    backends = {}
    for name in PROJECTS:
        build_system = _pyproject(name)["build-system"]
        requires = " ".join(build_system["requires"]).lower()
        assert "versioningit" in requires, f"{name} does not build with versioningit"
        backends[name] = build_system["build-backend"]
    assert backends["omniload"] == backends["dlt-filesystem"]


def test_both_projects_take_their_version_from_the_build():
    """A static `version` in either project would break the pairing silently."""
    for name in PROJECTS:
        project = _pyproject(name)["project"]
        assert "version" not in project, f"{name} pins a static version"
        assert "version" in project.get("dynamic", []), f"{name} version is not dynamic"


def test_omniload_requires_the_package_without_a_specifier():
    """A specifier here would be unchecked locally and stale on PyPI.

    Asserted against the built metadata rather than against `pyproject.toml`, because
    the requirement a user resolves is the one in `Requires-Dist`.
    """
    try:
        requires = metadata.requires("omniload") or []
    except metadata.PackageNotFoundError:  # pragma: no cover - environment-dependent
        pytest.skip("omniload is not installed in this environment")
    # The base dependency only. `dlt-filesystem; extra == "filesystem"` is the
    # forwarding extra and is bare for its own reasons, so matching on the name
    # alone would read the wrong line and pass whatever the base entry says.
    base = [
        requirement
        for requirement in requires
        if ";" not in requirement and requirement.startswith("dlt-filesystem")
    ]
    assert base == ["dlt-filesystem"], (
        f"expected an unspecified base requirement on dlt-filesystem, found {base}"
    )
