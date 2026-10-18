#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///
"""
Probe whether `dlt_filesystem` imports on the dependency set it actually
declares: the project's own `[project.dependencies]` (with their transitive
dependencies), and nothing else.

The test suite cannot answer this. It runs on an install with every extra, and
on whatever else the developer's environment carries, so a module reaching for
something only an extra (or nothing at all) supplies goes green there. This is
the only check in the tree that would notice a new one before a user installing
`dlt-filesystem` without extras does.

Two assertions, deliberately of different strength:

  1. HARD  -- `import dlt_filesystem.source.core`, the source entry point, must
     succeed. Anything it reaches is a real dependency of the package.
  2. SWEEP -- every module in the package is force-imported and the failures
     reported. A module may import something heavy at its own top level
     *because it is itself imported lazily*; that costs nothing at startup and
     is not a defect, so those are exempted in ALLOWED_MISSING with the reason.
     Anything else fails. Only a `ModuleNotFoundError` can be exempted: an
     installed module whose symbol is missing raises a plain `ImportError`
     carrying the same `name`, and exempting on the name alone would swallow a
     genuinely broken import inside the exempted module.

Each assertion runs in its own ephemeral environment built by `uv run
--isolated --no-config`, holding that dependency set and nothing else. Neither
flag is belt and braces. A `dependency-metadata` entry in an ambient `uv.toml`
can declare that some installed package requires the very name the set omits,
and `--isolated` does not stop uv reading configuration; `--no-config` stops a
discovered file, and an explicit `UV_CONFIG_FILE` outranks it (measured on uv
0.9.26). So the environment is refused rather than cleaned: any `UV_*` variable
the probe cannot vouch for and it declines to run. Dropping them would be its
own answer rather than a neutral one, because an index variable names where a
distribution comes from, and removing it can resolve the same name off PyPI with
different transitive dependencies and no error at all. The refusal is
deliberately wider than the set that can actually change an install, since
sorting the inert ones from the rest is the judgement it is declining to make.
The one exemption is `UV_RUN_RECURSION_DEPTH`, which `uv run` sets on everything
it starts and which would otherwise refuse this program's own `uv run --script`
form.

Only `dlt_filesystem` is put on the path, through a staging directory holding a
link to it alone, so neither the repository root nor anything else under `src/`
can supply a module the package imports. The interpreter runs isolated (`-I`) so an
inherited PYTHONPATH cannot supply a dependency the set omits, and under its own
`pycache_prefix` so the real tree's `__pycache__` can neither be written to nor
read from. That last one is not tidiness: a same-size edit with an unchanged
mtime is served from a stale `.pyc`, so the broken source is never read and the
probe passes over it.

Every declared dependency is validated as a requirement and then written out
verbatim. Validation is needed because a requirements file is a script rather
than a list: a member reading `-r /elsewhere.txt` is an instruction uv honours,
a bare `x.tar.gz` is a local archive it installs from the working directory, and
a TOML table degrades to its keys under `list()`. Writing the original back out
is needed because re-serialising a parse re-quotes an environment marker, and a
marker whose literal contains the other quote character then means something
else (measured on packaging 26.0).

Every assertion has to both exit zero and say it finished, and it says so on a
dedicated file descriptor rather than in its own output. Neither half is
redundant. An exit status alone cannot carry completion: the entry assertion
imports without a handler, so a module raising `SystemExit(0)` there ends the
interpreter with status 0 having printed nothing, and `os._exit(0)` does it from
anywhere, the sweep included, without raising at all. (The sweep does catch
`SystemExit`, per module, which is what lets it keep going and report the rest.)
A token in
ordinary output cannot carry provenance either: a module that prints the token
and then calls `os._exit(0)` forges it, and stdout and stderr are both reachable
from any module being imported. The descriptor is marked non-inheritable before the
package is touched, so a subprocess started at import time cannot write stray
bytes into the evidence either. What remains is code that writes to that
descriptor in this process, or forks rather than execs; nothing in-process rules
that out, and it is not what an accident looks like.

What it does not check: this imports from the source tree, so it says nothing
about a built wheel's metadata or packaged contents. Nor can an import sweep see
a lazy runtime import, a `try: import x except ImportError` that swallows a
missing dependency at module top, or a direct dependency that some other
requirement happens to supply transitively. There is no timeout either, so a
module that blocks at import hangs the probe rather than failing it. And it
takes `pyproject.toml` at its word once the file parses: a declaration that is
well formed and wrong is the answer it gives.

POSIX only. The completion evidence rides an inherited file descriptor, and
`subprocess`'s `pass_fds` has no Windows equivalent.

Usage:
    python release/util/standalone_import_probe.py [--keep] [--python 3.13]
    uv run --script release/util/standalone_import_probe.py

The second form needs no environment of its own: the script metadata above
declares what reading `pyproject.toml` costs, and uv supplies it. This file is
also its own child, re-entered in place with `--child`; `-I` keeps its directory
off `sys.path` at every version the project supports.

Exit status:
    0  the package imports on its own dependency set
    1  it does not, and the output says which module reached for what
    2  this program could not answer: it could not read `pyproject.toml`, or
       could not stage or walk the tree, or the environment is one it declines
       to guess about
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

#: Repository root, from this file's location: `<root>/release/util/<this>`.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The package under probe, and the source entry point assertion 1 imports.
PACKAGE = "dlt_filesystem"
ENTRY_MODULE = "dlt_filesystem.source.core"

#: The project directory. Its `[project.dependencies]` *is* the standalone
#: dependency set, so this reads the declaration a user installs.
PACKAGE_PROJECT = REPO_ROOT

#: `module: dependency` a module may import at its own top level without that
#: dependency belonging to the package's import surface. Keyed on the pair, not
#: the dependency alone: exempting a name outright would let any *other* module
#: start importing it unnoticed.
ALLOWED_MISSING = {
    # `source/format/bson_codec.py` imports bson at module top precisely because
    # that module is only ever imported lazily, from `_read_bson`. The cost is
    # paid when a BSON file is read and never at startup, and the `[bson]` extra
    # supplies it.
    "dlt_filesystem.source.format.bson_codec": "bson",
}

#: Floors that separate "the package is broken" from "this program is broken".
#: A requirement set that came back nearly empty, or a module walk that found
#: nearly nothing, would otherwise pass while proving nothing.
MIN_REQUIREMENTS = 5
MIN_MODULES = 10

#: Written to the evidence descriptor by a child that reached its verdict.
ENTRY_TOKEN = b"ENTRY-IMPORT-COMPLETED\n"
SWEEP_TOKEN = b"SWEEP-COMPLETED\n"

#: The one `UV_*` variable the refusal below tolerates. `uv run` sets it on
#: every program it starts, as a recursion guard, so refusing it would refuse
#: this program's own documented `uv run --script` invocation. It carries a
#: depth counter and nothing about resolution; measured as the only variable uv
#: injects, on uv 0.9.26.
UV_ENV_ALLOWED = {"UV_RUN_RECURSION_DEPTH"}

#: Every suffix the import system will load a module from. Anything here that
#: is not `.py` cannot be swept by walking source, so the sweep refuses it
#: rather than passing over it.
IMPORTABLE_SUFFIXES = tuple(
    dict.fromkeys(
        importlib.machinery.SOURCE_SUFFIXES
        + importlib.machinery.BYTECODE_SUFFIXES
        + importlib.machinery.EXTENSION_SUFFIXES
    )
)

#: Suffixes uv reads as a local archive rather than as a distribution name.
#: `Requirement("compat.tar.gz")` parses happily, name and all, and uv then
#: installs the file of that name from the working directory, running its build
#: and supplying whatever it likes. Path-shaped members (`./x`, `/x`, `a/b`) are
#: already refused by the parse; a bare archive name is the form that is not.
ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tar.zst",
    ".tar",
    ".tgz",
    ".tbz",
    ".txz",
    ".whl",
    ".zip",
)

#: Written to the evidence descriptor by a child whose own sanity check
#: failed, so the driver can tell "this program is broken" from "the package
#: is broken". An exit status cannot carry that distinction: a module raising
#: `SystemExit(2)` at import time produces the same status a floor would.
BROKEN_TOKEN = b"PROBE-BROKEN\n"


# --------------------------------------------------------------------------
# The dependency set
# --------------------------------------------------------------------------


def read_requirement_set(pyproject: pathlib.Path) -> list[str]:
    """
    The package's own `[project.dependencies]`, each validated and returned as
    written.
    """
    # Imported here rather than at module top so this file also runs as the
    # child below, under an interpreter that carries neither.
    try:
        import tomllib
        from packaging.requirements import Requirement
    except ImportError as exc:
        # Run as `python <this>` under the project interpreter rather than
        # `uv run --script`, neither name is guaranteed to be there.
        raise Failure(
            f"reading pyproject.toml needs Python 3.11+ and `packaging` "
            f"({type(exc).__name__}: {exc}). `uv run --script {__file__}` "
            "supplies both.",
            status=2,
        ) from exc

    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        dependencies = project["dependencies"]
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        # Not the package's failure: this program could not read the file it
        # derives the dependency set from, so it has no verdict to give.
        raise Failure(
            f"could not read `[project.dependencies]` from {pyproject}: "
            f"{type(exc).__name__}: {exc}",
            status=2,
        ) from exc

    def parse(value: object, where: str) -> "Requirement":
        """
        Validate a member as a requirement. The caller writes the *original*
        string out, never `str(parsed)`.

        Validation is what a requirements file needs, because it is a script
        rather than a list: a member reading `-r /elsewhere.txt` is an
        instruction uv honours, and a TOML table degrades to its keys under
        `list()`. Both would build an environment wider than the project declares.

        Round-tripping through the parse is not, and was the wrong instinct.
        `str(Requirement(...))` re-quotes an environment marker, so a marker
        whose string literal contains the other quote character comes back as
        several tokens. Measured on packaging 26.0:
        `demo; os_name == 'x" or os_name != "x'` is always false and its
        re-serialisation is always true, and the two compare equal, so no
        assertion over `Requirement` catches it. 26.3 keeps the original
        quoting, which makes it a property of whichever version is installed
        rather than one this program can rely on.
        """
        if not isinstance(value, str) or "\n" in value or "\r" in value:
            raise Failure(f"{where} is not a requirement string: {value!r}", status=2)
        try:
            parsed = Requirement(value)
        except Exception as exc:
            raise Failure(
                f"{where} does not parse: {value!r} ({exc})", status=2
            ) from exc
        if parsed.marker is not None:
            # uv evaluates a root requirement with no extras selected, so a
            # marker reading `extra` means one thing where it is declared and
            # another once resolved as a root requirement, installing a
            # distribution nothing asked for. Detected by evaluating the marker
            # under two different `extra` values rather than by looking for the
            # word, which would also fire on `os_name == 'extra'`.
            try:
                varies = parsed.marker.evaluate(
                    {"extra": "\x00one"}
                ) != parsed.marker.evaluate({"extra": "\x00not-an-extra"})
            except Exception as exc:
                raise Failure(
                    f"{where} has a marker that will not evaluate: {value!r} ({exc})",
                    status=2,
                ) from exc
            if varies:
                raise Failure(
                    f"{where} has a marker whose value depends on the selected "
                    f"extra: {value!r}. Copied out of the extra it would mean "
                    "something else, and this does not model that.",
                    status=2,
                )
        if parsed.url is None:
            if parsed.name.lower().endswith(ARCHIVE_SUFFIXES):
                raise Failure(
                    f"{where} names an archive rather than a distribution: "
                    f"{value!r}. uv would install the file of that name from "
                    "the working directory.",
                    status=2,
                )
        elif "://" not in parsed.url or parsed.url.startswith("file:"):
            # A direct reference to a path, in any of its spellings
            # (`foo @ x.whl`, `foo @ ./x.whl`, `foo @ /tmp/x.whl`,
            # `foo @ file:///tmp/x.whl`). uv installs the file from disk, so
            # what lands in the environment is decided outside pyproject.toml
            # altogether, which is the same hole the bare archive name opened.
            # A remote URL is left alone: it is still the file pyproject names.
            raise Failure(
                f"{where} is a direct reference to a local path: {value!r}. "
                "uv would install it from disk into the probe environment.",
                status=2,
            )
        return parsed

    if not isinstance(dependencies, list):
        raise Failure(
            f"`[project.dependencies]` in {pyproject} is a "
            f"{type(dependencies).__name__}, not a list",
            status=2,
        )
    requirements = []
    for index, dep in enumerate(dependencies):
        parse(dep, f"`[project.dependencies]` entry {index}")
        requirements.append(dep)

    if len(requirements) < MIN_REQUIREMENTS:
        raise Failure(
            f"read only {len(requirements)} requirement(s) from {pyproject}.\n"
            "      The extraction is broken, not the dependency set.",
            status=2,
        )
    return requirements


class _ProbeBroken(Exception):
    """The child's own sanity check failed, so it has no verdict to give."""


class Failure(Exception):
    """A failure to report as `FAIL: ...` and exit on."""

    def __init__(self, message: str, status: int = 1) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Running an assertion
# --------------------------------------------------------------------------


def uv_prefix(python: str, requirements: pathlib.Path) -> list[str]:
    """
    The uv invocation that builds the standalone environment, shared so the
    `--keep` reproduction line cannot drift from what actually ran.

    `--isolated` gives an environment built for this invocation and discarded
    after it, rather than a venv this program creates and has to keep honest.
    It does not stop uv reading configuration, and a `dependency-metadata` entry
    there can declare that some installed package requires the very name the
    set omits, which is exactly the false green this program exists to
    prevent. Measured on uv 0.9.26: a discovered `uv.toml` reaches an
    `--isolated --no-project` resolution and `--no-config` stops it. An explicit
    `UV_CONFIG_FILE` outranks `--no-config`, which is why the environment is
    refused rather than cleaned, up in `probe`.
    """
    return [
        "uv",
        "run",
        "--quiet",
        "--isolated",
        "--no-config",
        "--no-project",
        "--python",
        python,
        "--with-requirements",
        str(requirements),
    ]


def run_assertion(
    *,
    mode: str,
    child: pathlib.Path,
    stage: pathlib.Path,
    requirements: pathlib.Path,
    python: str,
    pycache: pathlib.Path,
    tmp: pathlib.Path,
) -> tuple[int, str, bytes]:
    """
    Run one assertion in an ephemeral environment holding the standalone
    dependency set. Returns its exit status, its combined output, and whatever
    it wrote to the evidence descriptor.
    """
    evidence_path = tmp / f"{mode}.evidence"
    with open(evidence_path, "w+b") as evidence:
        fd = evidence.fileno()
        os.set_inheritable(fd, True)
        command = [
            *uv_prefix(python, requirements),
            "--",
            "python",
            # -I: no PYTHONPATH, no user site, no cwd on the path. A dependency
            # the set omits cannot arrive from the caller's environment.
            "-I",
            # Its own bytecode cache, so a stale `.pyc` in the real tree can
            # neither serve this run nor be written by it.
            "-X",
            f"pycache_prefix={pycache}",
            str(child),
            "--child",
            mode,
            "--stage",
            str(stage),
            "--evidence-fd",
            str(fd),
        ]
        completed = subprocess.run(  # noqa: S603 - a command this file built
            command,
            pass_fds=(fd,),
            capture_output=True,
            text=True,
            check=False,
        )
        # No flush: nothing was written through this handle. The child writes
        # to the inherited descriptor with `os.write`, which is unbuffered, so
        # seeking back over the shared file offset is all that is needed.
        evidence.seek(0)
        written = evidence.read()

    output = completed.stdout + completed.stderr
    return completed.returncode, output, written


# --------------------------------------------------------------------------
# The child: what runs inside the standalone environment
# --------------------------------------------------------------------------


def child_main(mode: str, stage: str, evidence_fd: int) -> int:
    """
    Runs under the probe interpreter, inside the ephemeral environment, with
    only `stage` added to the path. Uses the standard library alone: anything
    else it reached for would be a dependency the standalone set does not carry.
    """
    # Before anything in the package is imported: a module that starts a child
    # with `close_fds=False`, or through `os.system`, would otherwise hand it
    # the evidence descriptor, and a stray byte from that child corrupts the
    # token.
    os.set_inheritable(evidence_fd, False)
    # Before the package is on the path, let alone imported: the descriptor's
    # number arrives on the command line, and leaving it there would hand every
    # module being imported the exact `os.write` needed to forge a completion
    # token. The fd stays open and guessable, which the module docstring says;
    # what this removes is having published it.
    sys.argv = sys.argv[:1]
    sys.path.insert(0, stage)

    try:
        if mode == "entry":
            return _child_entry(evidence_fd, stage)
        return _child_sweep(evidence_fd, stage)
    except _ProbeBroken as broken:
        # Said on the descriptor, not through an exit status: any module can
        # raise `SystemExit(2)` at import time, so a status cannot distinguish
        # this program failing from the package failing.
        print(f"FAIL: {broken}")
        os.write(evidence_fd, BROKEN_TOKEN)
        return 2


def _child_entry(evidence_fd: int, stage: str) -> int:
    import importlib.metadata

    print(
        f"    distributions installed: {len(list(importlib.metadata.distributions()))}"
    )
    module = importlib.import_module(ENTRY_MODULE)
    # The sweep checks this through `__path__`; the entry assertion needs its
    # own. A regular `dlt_filesystem` package in the environment would win over
    # the staged namespace package even with the stage first on `sys.path`, and
    # this assertion would report IMPORT OK for a tree nobody asked about.
    source = pathlib.Path(getattr(module, "__file__", "") or "")
    if not source.is_relative_to(pathlib.Path(stage).resolve()):
        raise _ProbeBroken(f"{ENTRY_MODULE} came from {source}, not the staged tree")
    print("    IMPORT OK")
    os.write(evidence_fd, ENTRY_TOKEN)
    return 0


def _is_cache_artifact(path: pathlib.Path) -> bool:
    """
    A `.pyc` inside a `__pycache__` directory is an ordinary compiled cache for
    a source file beside it, not a module in its own right. One anywhere else is
    a sourceless module and is importable by name.
    """
    return path.suffix == ".pyc" and path.parent.name == "__pycache__"


def _iter_py_files(root: pathlib.Path) -> "Iterator[pathlib.Path]":
    """
    Every `.py` file under `root`, with each failure to classify an entry raised
    rather than absorbed.

    `os.walk(onerror=...)` is not enough. CPython catches an `OSError` from
    `DirEntry.is_dir()` and files that entry under `filenames` *without* calling
    `onerror`, so a directory it cannot stat is silently demoted to a file and
    then dropped for not ending in `.py`. A symlink whose target sits in an
    unsearchable directory does exactly that; measured on 3.13, where the walk
    reports one error, misses the subtree, and returns the rest as though the
    tree were complete.

    An explicit stack rather than recursion, so a deep tree raises nothing that
    would escape the caller's `_ProbeBroken` handling and be read as a package
    defect. `__pycache__` is walked like any other directory: it can hold
    ordinary importable source, and a `.pyc` fails the suffix test anyway.
    """
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise _ProbeBroken(f"could not read {directory}: {exc}") from exc

        for entry in entries:
            path = pathlib.Path(entry.path)
            try:
                # Before `is_dir`, so a symlink is never classified by following
                # it. The walk does not descend into one, so its modules would
                # go unswept, and a package here has no reason to hold one.
                if entry.is_symlink():
                    raise _ProbeBroken(f"{path} is a symlink; the walk follows none")
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise _ProbeBroken(f"could not classify {path}: {exc}") from exc

            if is_directory:
                pending.append(path)
            elif entry.name.endswith(".py"):
                yield path
            elif entry.name.endswith(IMPORTABLE_SUFFIXES) and not _is_cache_artifact(
                path
            ):
                # A sourceless `.pyc` beside the source, or a compiled extension,
                # imports perfectly well and this sweep cannot enumerate it. Say
                # so rather than walk past it: an undeclared import inside one
                # would never be reached.
                raise _ProbeBroken(
                    f"{path} is importable and is not source; this sweep only "
                    "understands `.py` modules"
                )


def _walk_modules(stage: str) -> list[str]:
    """
    Every module in the package, by walking the tree.

    `dlt_filesystem` carries no `__init__.py` at any level, so it and every
    subpackage are namespace packages. `pkgutil.walk_packages` yields only
    directories that have an `__init__`, which here means it finds the two
    top-level modules and stops -- a sweep that reports OK having looked at
    almost nothing.
    """
    package = importlib.import_module(PACKAGE)
    roots = [pathlib.Path(entry) for entry in package.__path__]
    # A namespace package spans every directory of its name on the path, so more
    # than one root means the sweep is walking something besides what was
    # staged, and a module resolved from elsewhere would be swept as though it
    # came from this tree.
    if len(roots) != 1 or roots[0].parent != pathlib.Path(stage):
        raise _ProbeBroken(f"{PACKAGE}.__path__ is {roots}, not the staged tree")

    modules = set()
    for root in roots:
        for path in _iter_py_files(root):
            suffix = ".".join(path.relative_to(root).with_suffix("").parts)
            modules.add(f"{PACKAGE}.{suffix}".removesuffix(".__init__"))
    return sorted(modules)


def _child_sweep(evidence_fd: int, stage: str) -> int:
    modules = _walk_modules(stage)
    if len(modules) < MIN_MODULES:
        raise _ProbeBroken(
            f"the sweep found only {len(modules)} module(s). "
            "The walk is broken, not the package."
        )

    # Every target starts from the same module table. Without this, a circular
    # import can leave a sibling half-built in `sys.modules` -- module A imports
    # Z, Z imports A back and finishes, then A fails -- and the sweep later
    # "imports" Z straight from cache, so Z's own missing dependency is never
    # reached. Measured: importing Z first raises, importing it after A does
    # not. What this does not reset is the state of the *dependencies*, which
    # stay loaded across targets.
    preloaded = {
        key for key in sys.modules if key == PACKAGE or key.startswith(f"{PACKAGE}.")
    }

    failures: list[tuple[str, str, bool]] = []
    for name in modules:
        for key in [
            key
            for key in sys.modules
            if (key == PACKAGE or key.startswith(f"{PACKAGE}."))
            and key not in preloaded
        ]:
            del sys.modules[key]
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # The only failure an exemption may cover: the dependency is simply
            # not installed. Checked before the ImportError arm below, which it
            # is a subclass of.
            dep = exc.name or f"{type(exc).__name__}: {exc}"
            failures.append((name, dep, ALLOWED_MISSING.get(name) == dep))
        except ImportError as exc:
            # Every other ImportError subclass, dlt's own
            # MissingDependencyException included, and a plain ImportError from
            # a module that *is* installed. That last one carries `exc.name`
            # as well, so an exemption keyed on the name alone would swallow a
            # genuinely broken import inside the exempted module.
            failures.append((name, f"{type(exc).__name__}: {exc}", False))
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            # BaseException, not Exception: a module raising `SystemExit(0)` at
            # import time would otherwise end this interpreter with status 0,
            # having printed nothing and discarded every failure recorded so far.
            failures.append((name, f"{type(exc).__name__}: {exc}", False))

    print(f"    modules walked: {len(modules)}")
    for module, dep, is_allowed in sorted(failures):
        print(f"    {'allowed' if is_allowed else 'UNDECLARED':<10} {dep}  <- {module}")

    if any(not is_allowed for _, _, is_allowed in failures):
        print()
        print("FAIL: modules in the package failed to import on the standalone")
        print("      dependency set. Declare the requirement in the package's")
        print("      own `[project.dependencies]`, make the import lazy, or")
        print("      exempt the exact module/dependency pair in ALLOWED_MISSING")
        print("      with the reason it")
        print("      costs nothing at startup.")
        return 1

    print("    SWEEP OK")
    # On the evidence descriptor, not stdout: a module that printed this line
    # and then called `os._exit(0)` would otherwise pass the completion check
    # having stopped the sweep before it reached a verdict.
    os.write(evidence_fd, SWEEP_TOKEN)
    return 0


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def _report(mode: str, status: int, output: str, evidence: bytes, token: bytes) -> None:
    """Raise the right failure for an assertion that did not pass cleanly."""
    if status == 0 and evidence == token:
        return
    if BROKEN_TOKEN in evidence:
        # The child's own sanity check failed, so it never had a verdict about
        # the package to give. Carried out as status 2 rather than folded in
        # with a package defect.
        raise Failure(
            f"the {mode} assertion could not run:\n" + _indent(output), status=2
        )
    if status != 0:
        raise Failure(
            f"the {mode} assertion failed (interpreter exited {status}):\n"
            + _indent(output),
            status=1,
        )
    # Exit 0 without the token: something ended the interpreter before it could
    # reach a verdict, and whatever it had found went with it. `os._exit(0)`
    # does it from anywhere and raises nothing at all; an uncaught
    # `SystemExit(0)` does it on the entry path, which has no handler.
    raise Failure(
        f"the {mode} assertion exited 0 without reaching its verdict, so\n"
        "      whatever it had found is gone. A module ending the interpreter at\n"
        "      import time does this.\n" + _indent(output),
        status=1,
    )


def _indent(text: str) -> str:
    return "".join(f"    {line}\n" for line in text.splitlines())


def probe(python: str, keep: bool) -> int:
    ambient = sorted(
        k for k in os.environ if k.startswith("UV_") and k not in UV_ENV_ALLOWED
    )
    if ambient:
        # Not filtered out and carried on: an index variable names *where* a
        # distribution comes from, and dropping it can resolve the same name off
        # PyPI instead, with different transitive dependencies and no error.
        # Silently answering a different question is the one outcome this
        # program must not have, so it declines instead of guessing which of
        # these are inert.
        raise Failure(
            "these uv environment variables may affect what gets installed, "
            "and dropping them may affect it too, so this declines to guess: "
            + ", ".join(ambient)
            + ".\n      Unset them and re-run.",
            status=2,
        )
    if os.name != "posix":
        # The completion evidence rides an inherited file descriptor, and
        # `subprocess`'s `pass_fds` is POSIX-only.
        raise Failure(
            f"this probe needs a POSIX host; os.name is {os.name!r}", status=2
        )
    if shutil.which("uv") is None:
        raise Failure("uv is required", status=2)

    # Resolved once, here, so the path the child puts on `sys.path` and the one it
    # checks `__file__` against agree. macOS keeps `$TMPDIR` under `/var`, a link to
    # `/private/var`, and the entry assertion would refuse every run there.
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="standalone-import-probe-")).resolve()
    try:
        print("==> reading the standalone dependency set from pyproject.toml")
        requirements = read_requirement_set(PACKAGE_PROJECT / "pyproject.toml")
        for requirement in requirements:
            print(f"    {requirement}")
        requirements_file = tmp / "requirements.txt"
        requirements_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")

        # Exposes `dlt_filesystem` and nothing else. A link to `src/` would put
        # anything else that lands there on the path too, and a module the
        # package imports from it would resolve here and nowhere else.
        stage = tmp / "stage"
        stage.mkdir()
        source = PACKAGE_PROJECT / "src" / PACKAGE
        if not source.is_dir():
            # Staged blind, this would be a dangling symlink and every
            # assertion would blame the package for a moved directory.
            raise Failure(f"no package source at {source}", status=2)
        try:
            (stage / PACKAGE).symlink_to(source)
        except OSError as exc:
            raise Failure(f"could not stage {source}: {exc}", status=2) from exc

        # The child is this same file, re-entered with `--child`. `-I` keeps
        # the script's own directory off `sys.path` at every version the project
        # supports (measured at 3.10 and 3.13), so running it in place exposes
        # nothing that the staged directory does not.
        child = pathlib.Path(__file__).resolve()

        for mode, header, token in (
            ("entry", f"assertion 1 (hard): import {ENTRY_MODULE}", ENTRY_TOKEN),
            ("sweep", "assertion 2 (sweep): force-import every module", SWEEP_TOKEN),
        ):
            print()
            print(f"==> {header}")
            status, output, evidence = run_assertion(
                mode=mode,
                child=child,
                stage=stage,
                requirements=requirements_file,
                python=python,
                pycache=tmp / f"pycache-{mode}",
                tmp=tmp,
            )
            print(output, end="")
            _report(mode, status, output, evidence, token)

        print()
        print("PROBE OK")
        return 0
    finally:
        if keep:
            print(f"probe working directory kept at: {tmp}")
            print("    reproduce that environment with:")
            print(
                "      "
                + shlex.join(
                    [*uv_prefix(python, tmp / "requirements.txt"), "--", "python", "-I"]
                )
            )
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=os.environ.get("PROBE_PYTHON", "3.13"),
        help="interpreter the package is imported under (default: 3.13)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the working directory, including the requirement set",
    )
    # The child entry point. Not part of the interface: the driver re-enters
    # this file with these, inside the environment it built.
    parser.add_argument("--child", choices=("entry", "sweep"), help=argparse.SUPPRESS)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--evidence-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        return child_main(args.child, args.stage, args.evidence_fd)

    try:
        return probe(args.python, args.keep)
    except Failure as failure:
        print(f"FAIL: {failure}", file=sys.stderr)
        return failure.status


if __name__ == "__main__":
    sys.exit(main())
