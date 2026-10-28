#!/usr/bin/env python3
#
# ==-- compare_solver_stat - compare solver statistics across compilers --==#
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2026 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ==------------------------------------------------------------------------==#
#
# Compiles Swift files with several compilers using -stats-output-dir and
# prints a table with one row per run, comparing the chosen statistics:
#
#   main     the commit main@origin points to (built from source)
#   wip      the working directory as it was when the script started
#   others   prebuilt swift.org snapshots, given with --snapshot NAME=DATE
#
# Every compiler gets two columns, NAME-off and NAME-on, for example main-off
# and main-on. The on column adds the experimental solver flags (--on-flags).
# A compiler that does not know one of those flags has those runs skipped, and
# a column in which every run is skipped is left out of the table.
#
# By default it compiles every .swift file under
# validation-test/Sema/type_checker_perf. Rows where a column differs from the
# baseline are marked. Below the table, a summary counts for each column how
# many runs got smaller or larger than the baseline, and a second summary
# compares chosen pairs, measured column against reference column
# (--compare, default wip-on against main-on). --by-dir repeats the summaries
# for each subdirectory. The report is printed, and it is also saved as
# <out>/report.md unless --no-report is given.
#
# Each RUN line at the top of a test file is one run, with the flags of that
# line. A file with several RUN lines is compiled once per line, and its rows
# are named file#1, file#2, and so on. The lit substitutions are translated,
# -verify is dropped, and -solver-scope-threshold=N is not passed. N is
# remembered instead, and every compiler's scope count is reported as over or
# under it. Lines that run something other than the frontend are skipped.
#
# The script builds swift-frontend twice: first the working directory, then
# main@origin, which means it temporarily moves the repository to another
# commit. It refuses to do so unless the repository is in a clean state, and it
# puts the repository back afterwards, also when a build fails.
#
# The saved binaries use the stdlib in the build directory, which this script
# does not rebuild. Each saved compiler sits in <out>/compilers/<name>/bin, and
# its lib directory next to it is a link to the build directory's lib
# directory, or a copy of its host libraries with --copy-lib.
#
# ==------------------------------------------------------------------------==#

import argparse
import datetime
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


_SWIFT_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _SWIFT_DIR.parent

_DEFAULT_BUILD_DIR = (
    _ROOT_DIR / "build" / "Ninja-RelWithDebInfoAssert" / "swift-macosx-arm64"
)
_DEFAULT_TEST_DIR = _SWIFT_DIR / "validation-test" / "Sema" / "type_checker_perf"
_DEFAULT_WORKSPACE = "/tmp/swift_snapshot_tool_workspace_v1"
_DEFAULT_ON_FLAGS = (
    "-solver-enable-promote-supertypes "
    "-solver-enable-type-var-joins "
    "-solver-enable-short-conversions"
)
_DEFAULT_EXTRA_FLAGS = "-debug-time-function-bodies"

# The statistic that -solver-scope-threshold limits.
_SCOPE_STAT = "NumConstraintScopes"

# What the frontend prints when it does not have an option.
_UNKNOWN_ARGUMENT = "unknown argument"

# Flags from a RUN line that are not passed on.
_DROPPED_FLAGS = {"-verify", "-verify-ignore-unrelated"}

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


class Failure(Exception):
    """An error that has already been explained to the user."""


class Unsupported(Exception):
    """A RUN line the script cannot translate."""


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _command_text(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def _run(command, cwd=None, stream=False):
    """Runs a command, returning its stdout. Raises Failure with the command
    and its output if it fails."""
    try:
        result = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            stdout=None if stream else subprocess.PIPE,
            stderr=None if stream else subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise Failure(f"Could not run: {_command_text(command)}\n{error}")
    if result.returncode != 0:
        output = ""
        if not stream:
            output = (result.stdout + result.stderr).strip()
        raise Failure(
            f"Command failed with exit code {result.returncode}: "
            f"{_command_text(command)}\n{output}"
        )
    return "" if stream else result.stdout


# -----------------------------------------------------------------------------
# Version control


class Vcs:
    """A repository that can be checked, moved to main@origin and restored."""

    def __init__(self, path):
        self.path = path
        self.stashed = False

    def run(self, *command, stream=False):
        return _run(command, cwd=self.path, stream=stream)

    def current(self):
        raise NotImplementedError

    def problems(self, allow_dirty):
        """Returns reasons the repository is not safe to switch away from."""
        raise NotImplementedError

    def fetch(self):
        raise NotImplementedError

    def main_commit(self):
        raise NotImplementedError

    def switch_to(self, commit):
        raise NotImplementedError

    def restore(self, token):
        raise NotImplementedError

    def label(self, token):
        """Returns a short description of a position for the report."""
        return token

    def restore_hint(self, token):
        """Returns the command that moves back to a position by hand."""
        return token


class Jj(Vcs):
    name = "jj"

    def _log_template(self, revset, template):
        return self.run("jj", "log", "-r", revset, "--no-graph", "-T", template).strip()

    def _parents(self):
        output = self.run(
            "jj", "log", "-r", "@-", "--no-graph", "-T", 'commit_id ++ "\\n"'
        )
        return output.split()

    def current(self):
        # Any jj command snapshots the working copy first.
        empty = self._log_template("@", "empty") == "true"
        described = bool(self._log_template("@", "description"))
        if empty and not described:
            # jj abandons an empty, undescribed change when it is left, so it
            # cannot be edited again. A new empty change on the same parents
            # is the same position.
            return "new " + " ".join(self._parents())
        return "edit " + self._log_template("@", "change_id")

    def label(self, token):
        action, _, target = token.partition(" ")
        if action == "edit":
            return target
        return "an empty change on " + ", ".join(p[:12] for p in target.split())

    def restore_hint(self, token):
        return "jj " + token

    def problems(self, allow_dirty):
        found = []
        if self._log_template("@ & conflicts()", "change_id"):
            found.append("the working-copy change has conflicts")
        # Edits live in the working-copy change, which jj keeps, so there is
        # no dirty state to check for here.
        return found

    def fetch(self):
        self.run("jj", "git", "fetch", stream=True)

    def main_commit(self):
        return self._log_template("main@origin", "commit_id")

    def switch_to(self, commit):
        # An empty child of the commit has the same tree. Unlike `jj edit`
        # this also works when the commit is immutable, and jj abandons the
        # child when the script moves back.
        self.run("jj", "new", commit)

    def restore(self, token):
        action, _, target = token.partition(" ")
        self.run("jj", action, *target.split())
        after_action, _, after_target = self.current().partition(" ")
        if after_action != action or set(after_target.split()) != set(target.split()):
            raise Failure(f"Could not verify the restore of {self.label(token)}.")


class Git(Vcs):
    name = "git"

    def _git_path(self, name):
        path = self.run("git", "rev-parse", "--git-path", name).strip()
        return Path(path) if os.path.isabs(path) else self.path / path

    def current(self):
        branch = subprocess.run(
            ["git", "symbolic-ref", "-q", "--short", "HEAD"],
            cwd=self.path,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        return branch or self.run("git", "rev-parse", "HEAD").strip()

    def problems(self, allow_dirty):
        found = []
        states = [
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "BISECT_LOG",
            "rebase-merge",
            "rebase-apply",
        ]
        for state in states:
            if self._git_path(state).exists():
                found.append(f"a git operation is in progress ({state})")
        if self.run("git", "diff", "--name-only", "--diff-filter=U").strip():
            found.append("there are unmerged paths")
        if self.run("git", "status", "--porcelain").strip() and not allow_dirty:
            found.append(
                "there are uncommitted or untracked files (use --allow-dirty "
                "to stash them while main is built)"
            )
        return found

    def fetch(self):
        self.run("git", "fetch", "origin", stream=True)

    def main_commit(self):
        return self.run("git", "rev-parse", "origin/main").strip()

    def switch_to(self, commit):
        if self.run("git", "status", "--porcelain").strip():
            self.run("git", "stash", "push", "--include-untracked")
            self.stashed = True
        self.run("git", "checkout", "--detach", commit)

    def restore(self, token):
        self.run("git", "checkout", token)
        if self.stashed:
            self.run("git", "stash", "pop")
            self.stashed = False
        if self.current() != token:
            raise Failure(f"Could not verify the restore of {token}.")

    def restore_hint(self, token):
        return f"git checkout {token}"


def _pick_vcs(path, requested, force):
    has_jj = (path / ".jj").exists()
    has_git = (path / ".git").exists()
    if requested == "auto":
        requested = "jj" if has_jj else "git"
    if requested == "git" and has_jj and not force:
        raise Failure(
            f"{path} is a colocated jj repository. Mutating git commands "
            "would fight with jj. Use --vcs jj, or pass --force-vcs."
        )
    if requested == "jj" and not has_jj:
        raise Failure(f"{path} is not a jj repository.")
    if requested == "git" and not has_git:
        raise Failure(f"{path} is not a git repository.")
    return Jj(path) if requested == "jj" else Git(path)


# -----------------------------------------------------------------------------
# Compilers


class Compiler:
    def __init__(self, name, frontend, resource_dir, flags):
        self.name = name
        self.frontend = frontend
        self.resource_dir = resource_dir
        self.flags = flags


def _sdk_path(override):
    if override:
        return override
    try:
        return _run(["xcrun", "--show-sdk-path"]).strip()
    except Failure:
        return None


def _install_libs(args, compiler_dir):
    """Makes the host libraries available to a saved swift-frontend. The
    binary loads them through its rpath, ../lib/swift/host/compiler, so they
    have to sit next to the bin directory it was copied into."""
    source = args.build_dir / "lib"
    target = compiler_dir / "lib"
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    if args.copy_lib:
        (target / "swift").mkdir(parents=True)
        shutil.copytree(
            source / "swift" / "host", target / "swift" / "host", symlinks=True
        )
    else:
        target.symlink_to(source)


def _build_frontend(args, name, destination):
    """Builds swift-frontend and saves it. The output of ninja goes to
    <out>/logs/build-<name>.log, and the terminal only gets the result."""
    ninja = args.build_dir.parent / "build-utils" / "ninja"
    command = [ninja, "-C", args.build_dir, "bin/swift-frontend"]
    log_path = destination.parent.parent / "logs" / f"build-{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(log_path, "w") as log_file:
        log_file.write(f"{_command_text(command)}\n\n")
        log_file.flush()
        try:
            result = subprocess.run(
                [str(part) for part in command],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as error:
            raise Failure(f"Could not run: {_command_text(command)}\n{error}")
    seconds = time.monotonic() - start
    if result.returncode != 0:
        raise Failure(
            f"The {name} build FAILED after {seconds:.0f} s "
            f"(ninja exit code {result.returncode}). See {log_path}"
        )
    (destination / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.build_dir / "bin" / "swift-frontend", destination / "bin")
    _install_libs(args, destination)
    _log(f"The {name} build succeeded in {seconds:.0f} s. Log: {log_path}")


def _check_ready(args, swift_vcs, syntax_vcs):
    problems = []
    ninja = args.build_dir.parent / "build-utils" / "ninja"
    if not ninja.exists():
        problems.append(f"the cached-build ninja wrapper {ninja} does not exist")
    if not args.build_dir.is_dir():
        problems.append(f"the build directory {args.build_dir} does not exist")
    for vcs in (swift_vcs, syntax_vcs):
        if vcs is not None:
            for problem in vcs.problems(args.allow_dirty):
                problems.append(f"{vcs.path.name}: {problem}")
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise Failure(
            "The repository is not in a state the script can safely switch "
            f"away from:\n{details}\n"
            "Put the repository into a good state and run the script again."
        )


def _build_wip_and_main(args, compilers_dir):
    """Builds the working directory, then main@origin, and puts the
    repository back."""
    swift_vcs = _pick_vcs(_SWIFT_DIR, args.vcs, args.force_vcs)
    syntax_path = _ROOT_DIR / "swift-syntax"
    syntax_vcs = None
    if syntax_path.is_dir():
        syntax_vcs = _pick_vcs(syntax_path, "auto", args.force_vcs)
        _log(
            "Note: swift-syntax is checked but not moved, so main is built "
            "against the swift-syntax checkout you have."
        )
    _check_ready(args, swift_vcs, syntax_vcs)

    original = swift_vcs.current()
    _log(f"Recorded the current position: {swift_vcs.label(original)}")
    if args.fetch:
        swift_vcs.fetch()
    main_commit = swift_vcs.main_commit()
    if not main_commit:
        raise Failure("Could not resolve main@origin.")

    _log("Building wip ...")
    _build_frontend(args, "wip", compilers_dir / "wip")
    label = swift_vcs.label(original)
    (compilers_dir / "wip" / "commit").write_text(label + "\n")

    try:
        _log(f"Building main ({main_commit}) ...")
        swift_vcs.switch_to(main_commit)
        _build_frontend(args, "main", compilers_dir / "main")
        (compilers_dir / "main" / "commit").write_text(main_commit + "\n")
    finally:
        _log("Restoring the original position ...")
        try:
            swift_vcs.restore(original)
        except Failure as error:
            hint = swift_vcs.restore_hint(original)
            _log(f"{error}\nRestore the repository by hand: {hint}")
            raise


def _download_snapshot(args, date):
    if args.snapshot_tool:
        tool = shlex.split(args.snapshot_tool)
    elif os.environ.get("SWIFT_SNAPSHOT_TOOL"):
        tool = shlex.split(os.environ["SWIFT_SNAPSHOT_TOOL"])
    else:
        package = _SWIFT_DIR / "utils" / "swift_snapshot_tool"
        tool = [
            "swift",
            "run",
            "--package-path",
            package,
            "swift_snapshot_tool",
        ]
    command = tool + ["download", "--date", date, "--workspace", args.workspace]
    if args.snapshot_branch:
        command.append(f"--{args.snapshot_branch}")
    lines = _run(command).strip().splitlines()
    root = Path(lines[-1]) if lines else None
    if root is None or not root.is_dir():
        raise Failure(
            "swift_snapshot_tool did not print a toolchain directory:\n"
            + "\n".join(lines)
        )
    return root


def _collect_compilers(args, out_dir):
    compilers_dir = out_dir / "compilers"
    if not args.skip_build:
        compilers_dir.mkdir(parents=True, exist_ok=True)
        _build_wip_and_main(args, compilers_dir)

    resource_dir = args.build_dir / "lib" / "swift"
    on_flags = shlex.split(args.on_flags)
    compilers = []
    for label in ("main", "wip"):
        frontend = compilers_dir / label / "bin" / "swift-frontend"
        if not frontend.exists():
            raise Failure(f"There is no saved compiler at {frontend}.")
        for state, flags in (("off", []), ("on", on_flags)):
            compilers.append(
                Compiler(f"{label}-{state}", frontend, resource_dir, flags)
            )

    for spec in args.snapshot:
        name, separator, date = spec.partition("=")
        if not separator or not name or not date:
            raise Failure(f"--snapshot expects NAME=DATE, got '{spec}'.")
        root = _download_snapshot(args, date)
        for state, flags in (("off", []), ("on", on_flags)):
            compilers.append(
                Compiler(
                    f"{name}-{state}",
                    root / "usr" / "bin" / "swift-frontend",
                    root / "usr" / "lib" / "swift",
                    flags,
                )
            )
    return compilers


# -----------------------------------------------------------------------------
# Runs


class Run:
    """One compilation of a test file, with the flags of one RUN line."""

    def __init__(self, name, path, flags, threshold):
        self.name = name
        self.path = path
        self.flags = flags
        self.threshold = threshold


def _find_files(args):
    """Returns (path, name) pairs, where name identifies the file in the
    report."""
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise Failure(f"{path} does not exist.")
        return [(path.resolve(), path.name)]
    directory = args.dir
    if not directory.is_dir():
        raise Failure(f"The test directory {directory} does not exist.")
    paths = sorted(directory.rglob("*.swift"))
    if not paths:
        raise Failure(f"There are no .swift files under {directory}.")
    return [(path, path.relative_to(directory).as_posix()) for path in paths]


def _read_run_lines(path):
    """Returns the text after RUN: for each RUN line in the comment block at
    the top of a file. A line that ends in a backslash continues on the next
    RUN line."""
    lines = []
    pending = ""
    with open(path, encoding="utf-8", errors="replace") as source:
        for line in source:
            line = line.strip()
            if line and not line.startswith("//"):
                break
            match = re.match(r"//\s*RUN:\s?(.*)$", line)
            if not match:
                continue
            text = match.group(1).strip()
            if text.endswith("\\"):
                pending += text[:-1] + " "
                continue
            lines.append(pending + text)
            pending = ""
    return lines


def _translate_run_line(text):
    """Turns a RUN line into (frontend flags, scope threshold or None).
    Raises Unsupported if the line runs something else than the frontend or
    uses a substitution the script does not know."""
    try:
        tokens = shlex.split(text.split("|", 1)[0])
    except ValueError as error:
        raise Unsupported(str(error))
    if tokens and tokens[0] == "not":
        tokens = tokens[1:]
    tokens = [token for token in tokens if token != "2>&1"]
    if not tokens:
        raise Unsupported("the line is empty")

    flags = []
    if tokens[0] == "%target-typecheck-verify-swift":
        flags = ["-typecheck", "-disable-objc-attr-requires-foundation-module"]
    elif tokens[0] != "%target-swift-frontend":
        raise Unsupported(f"it runs {tokens[0]}")

    threshold = None
    for token in tokens[1:]:
        match = re.fullmatch(r"-solver-scope-threshold=(\d+)", token)
        if match:
            threshold = int(match.group(1))
            continue
        if token == "%s" or token in _DROPPED_FLAGS:
            continue
        token = token.replace("%target-cpu", platform.machine())
        if "%" in token:
            raise Unsupported(f"it uses the substitution in {token}")
        flags.append(token)
    if "-typecheck" not in flags:
        flags.insert(0, "-typecheck")
    return flags, threshold


def _plan_runs(files):
    """Returns the runs for the files and a note for each skipped RUN line."""
    runs = []
    skipped = []
    for path, name in files:
        lines = _read_run_lines(path)
        if not lines:
            runs.append(Run(name, path, ["-typecheck"], None))
            continue
        for index, line in enumerate(lines, 1):
            run_name = name if len(lines) == 1 else f"{name}#{index}"
            try:
                flags, threshold = _translate_run_line(line)
            except Unsupported as error:
                skipped.append(f"{run_name}: skipped, {error}: {line}")
                continue
            runs.append(Run(run_name, path, flags, threshold))
    return runs, skipped


def _run_compiler(compiler, runs, args, sdk, out_dir):
    """Compiles every run. Returns the names of the runs that wrote
    statistics and the names of the runs that were skipped. A non-zero exit is
    normal here, because the tests expect errors, and does not stop the run. A
    run is skipped when the compiler rejects an argument as unknown, which
    means it does not have one of the options. A run that writes no
    statistics for another reason, for example a crash, is recorded as
    failed."""
    have_stats = set()
    skipped = set()
    for run in runs:
        stats_dir = out_dir / "stats" / compiler.name / run.name
        stats_dir.mkdir(parents=True)
        # The RUN line flags come last, so they win over the flags of the
        # compiler column.
        command = [
            compiler.frontend,
            run.path,
            "-stats-output-dir",
            stats_dir,
            "-resource-dir",
            compiler.resource_dir,
        ]
        if sdk:
            command += ["-sdk", sdk]
        command += shlex.split(args.extra) + compiler.flags + run.flags

        log_path = out_dir / "logs" / compiler.name / f"{run.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_file:
            log_file.write(f"{_command_text(command)}\n\n")
            log_file.flush()
            subprocess.run(
                [str(part) for part in command],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if any(stats_dir.glob("*.json")):
            have_stats.add(run.name)
        elif _UNKNOWN_ARGUMENT in log_path.read_text(errors="replace"):
            skipped.add(run.name)
    return have_stats, skipped


# -----------------------------------------------------------------------------
# Reporting


def _read_stat(stats_dir, stat):
    """Sums the statistic over every JSON file in a stats directory. Counters
    that are zero are left out of the file, so a file without the statistic
    counts as 0. A key matches when it is the statistic or ends with
    '.<statistic>'. Returns None when there are no stats files."""
    total = None
    for path in sorted(stats_dir.glob("*.json")):
        with open(path) as stats_file:
            data = json.load(stats_file)
        total = total or 0
        for key, value in data.items():
            if key == stat or key.endswith("." + stat):
                total += value
    return total


def _marker(directions):
    if directions == {"smaller"}:
        return "▼"
    if directions == {"larger"}:
        return "▲"
    if directions:
        return "±"
    return " "


def _cell(value, base, is_baseline):
    """Returns the text of a cell and the direction of its change, if any."""
    if value is None:
        return "fail", None
    text = f"{value:,}"
    if is_baseline or base is None or value == base:
        return text, None
    direction = "smaller" if value < base else "larger"
    if base:
        text += f" ({(value - base) / base * 100:+.1f}%)"
    return text, direction


def _compare_columns(runs, stat, measured, reference, value_of, skipped_runs):
    """Counts how the measured column compares with the reference column over
    the runs. A run is skipped when either column skipped it, and failed when
    either column wrote no statistics. over counts the runs in which the
    measured column is over the scope threshold of its RUN line, and newly
    over and newly under count the runs in which that changed against the
    reference."""
    counts = {
        "runs": 0,
        "smaller": 0,
        "larger": 0,
        "same": 0,
        "failed": 0,
        "skipped": 0,
        "over": 0,
        "newly_over": 0,
        "newly_under": 0,
    }
    for run in runs:
        counts["runs"] += 1
        if run.name in skipped_runs[measured] or run.name in skipped_runs[reference]:
            counts["skipped"] += 1
            continue
        value = value_of(measured, run, stat)
        base = value_of(reference, run, stat)
        if value is None or base is None:
            counts["failed"] += 1
            continue
        if value < base:
            counts["smaller"] += 1
        elif value > base:
            counts["larger"] += 1
        else:
            counts["same"] += 1
        if stat == _SCOPE_STAT and run.threshold is not None:
            over = value > run.threshold
            base_over = base > run.threshold
            counts["over"] += over
            counts["newly_over"] += over and not base_over
            counts["newly_under"] += base_over and not over
    return counts


def _format_matrix(title, rows, with_thresholds):
    """Returns the lines of a summary table. Each row is (label, counts,
    is_baseline). The baseline row has nothing to compare, so it shows a dash
    instead of the comparison counts."""
    header = ["", "smaller", "larger", "same", "fail", "skip"]
    if with_thresholds:
        header += ["|", "over", "newly over", "newly under"]
    table = [header]
    for label, counts, is_baseline in rows:

        def compared(key):
            return "-" if is_baseline else str(counts[key])

        cells = [
            label,
            compared("smaller"),
            compared("larger"),
            compared("same"),
            str(counts["failed"]),
            str(counts["skipped"]),
        ]
        if with_thresholds:
            cells += [
                "|",
                str(counts["over"]),
                compared("newly_over"),
                compared("newly_under"),
            ]
        table.append(cells)
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = [title]
    for row in table:
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(widths[i]) for i, cell in enumerate(row) if i > 0]
        lines.append("  ".join(cells))
    return lines


def _valid_columns(args):
    """Returns the names of all the columns the run can have, before any of it
    has run."""
    compilers = ["main", "wip"] + [spec.partition("=")[0] for spec in args.snapshot]
    return [f"{name}-{state}" for name in compilers for state in ("off", "on")]


def _resolve_column(name, valid):
    """Accepts a column name, or a compiler name for its off column."""
    if name in valid:
        return name
    if f"{name}-off" in valid:
        return f"{name}-off"
    raise Failure(f"'{name}' is not a column. The columns are: {', '.join(valid)}.")


def _parse_pairs(specs, valid):
    """Returns the (measured, reference) pairs for the pair table. The default
    compares wip-on with main-on."""
    if not specs:
        return [("wip-on", "main-on")]
    pairs = []
    for spec in specs:
        measured, separator, reference = spec.partition(":")
        if not separator or not measured or not reference:
            raise Failure(f"--compare expects MEASURED:REFERENCE, got '{spec}'.")
        pairs.append(
            (_resolve_column(measured, valid), _resolve_column(reference, valid))
        )
    return pairs


class Report:
    """Collects the report twice: as the text printed to the terminal, and as
    a markdown document that is saved to a file."""

    def __init__(self):
        self.terminal = []
        self.markdown = []

    def begin(self):
        self.terminal += ["", "========", ""]
        self.markdown += ["# compare_solver_stat report", ""]

    def meta(self, text):
        """A fact that only the markdown document needs."""
        self.markdown.append(f"- {text}")

    def fact(self, text):
        self.terminal.append(text)
        self.markdown.append(f"- {text}")

    def section(self, title, show=True):
        """Starts a section. The terminal gets the title only when show is set,
        the markdown always gets a heading."""
        self.terminal += [""] + ([title] if show else [])
        self.markdown += ["", f"## {title}", ""]

    def block(self, plain, colored=None):
        self.terminal += colored or plain
        self.markdown += ["```text"] + plain + ["```"]

    def note(self, text):
        self.terminal += ["", text]
        self.markdown += ["", text]


def _print_table(compilers, runs, have_stats, skipped_runs, omitted, args, out_dir):
    names = [compiler.name for compiler in compilers]
    baseline = args.baseline or names[0]
    if baseline not in names:
        raise Failure(f"The baseline '{baseline}' is not one of {names}.")
    use_color = sys.stdout.isatty() and not args.no_color
    pairs = [(m, r) for m, r in args.pairs if m in names and r in names]
    for measured, reference in args.pairs:
        if (measured, reference) not in pairs:
            _log(f"Left out {measured} vs {reference}: a column was skipped entirely.")

    cache = {}

    def value_of(compiler_name, run, stat):
        key = (compiler_name, run.name, stat)
        if key not in cache:
            cache[key] = None
            if run.name in have_stats[compiler_name]:
                stats_dir = out_dir / "stats" / compiler_name / run.name
                cache[key] = _read_stat(stats_dir, stat)
        return cache[key]

    report = Report()
    report.begin()
    report.meta(f"date: {datetime.datetime.now():%Y-%m-%d %H:%M}")
    report.meta(f"command: `{_command_text(sys.argv)}`")
    report.meta(f"test directory: {args.file or args.dir}, {len(runs)} runs")
    report.meta(f"columns: {', '.join(names)}")
    for label in ("main", "wip"):
        commit_file = out_dir / "compilers" / label / "commit"
        if commit_file.exists():
            report.fact(f"{label}: {commit_file.read_text().strip()}")
    report.fact(f"percentages are relative to the baseline, {baseline}")
    report.fact("▼ smaller  ▲ larger  ± mixed (compared with the baseline)")
    if omitted:
        report.fact(
            f"left out, every run skipped (unknown option): {', '.join(omitted)}"
        )

    # rows hold (marker, label, threshold, cells), where cells are
    # (text, direction) per compiler.
    rows = []
    totals = {name: {stat: 0 for stat in args.stat} for name in names}
    for run in runs:
        for stat in args.stat:
            values = {name: value_of(name, run, stat) for name in names}
            base = values[baseline]
            has_threshold = stat == _SCOPE_STAT and run.threshold is not None
            cells = []
            directions = set()
            for name in names:
                value = values[name]
                if run.name in skipped_runs[name]:
                    text, direction = "skip", None
                else:
                    text, direction = _cell(value, base, name == baseline)
                if has_threshold and value is not None:
                    text += " over" if value > run.threshold else " under"
                cells.append((text, direction))
                if direction:
                    directions.add(direction)
            if all(value is not None for value in values.values()):
                for name in names:
                    totals[name][stat] += values[name]
            if args.only_changed and not directions:
                continue
            label = run.name if len(args.stat) == 1 else f"{run.name} [{stat}]"
            threshold = f"{run.threshold:,}" if has_threshold else ""
            rows.append((_marker(directions), label, threshold, cells))

    header = ["", "run", "threshold"] + names
    table = [header]
    for marker, label, threshold, cells in rows:
        table.append([marker, label, threshold] + [text for text, _ in cells])
    for stat in args.stat:
        label = "total" if len(args.stat) == 1 else f"total [{stat}]"
        cells = []
        for name in names:
            text, _ = _cell(
                totals[name][stat], totals[baseline][stat], name == baseline
            )
            cells.append(text)
        table.append([" ", label, ""] + cells)

    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    colors = {"smaller": _GREEN, "larger": _RED}

    def render(row, directions, color):
        parts = []
        for index, cell in enumerate(row):
            padded = cell.ljust(widths[index])
            direction = directions[index] if directions else None
            if color and direction:
                padded = colors[direction] + padded + _RESET
            parts.append(padded)
        return "  ".join(parts).rstrip()

    plain_lines = [render(header, None, False)]
    colored_lines = [render(header, None, use_color)]
    for index, row in enumerate(table[1:]):
        directions = None
        if index < len(rows):
            directions = [None] * 3 + [direction for _, direction in rows[index][3]]
        plain_lines.append(render(row, directions, False))
        colored_lines.append(render(row, directions, use_color))
    report.section("Runs", show=False)
    report.block(plain_lines, colored_lines)

    def print_summary(subset, suffix):
        for stat in args.stat:
            with_thresholds = stat == _SCOPE_STAT and any(
                run.threshold is not None for run in subset
            )
            matrix = []
            for name in names:
                counts = _compare_columns(
                    subset, stat, name, baseline, value_of, skipped_runs
                )
                label = f"{name} (baseline)" if name == baseline else name
                matrix.append((label, counts, name == baseline))
            title = f"{stat}: measured vs baseline {baseline}, {len(subset)} runs"
            heading, *lines = _format_matrix(title + suffix, matrix, with_thresholds)
            report.section(heading)
            report.block(lines)
            if pairs:
                matrix = []
                for measured, reference in pairs:
                    counts = _compare_columns(
                        subset, stat, measured, reference, value_of, skipped_runs
                    )
                    matrix.append((f"{measured} vs {reference}", counts, False))
                title = f"{stat}: measured vs reference, {len(subset)} runs"
                heading, *lines = _format_matrix(
                    title + suffix, matrix, with_thresholds
                )
                report.section(heading)
                report.block(lines)

    print_summary(runs, "")
    if args.by_dir:
        groups = {}
        for run in runs:
            group = run.name.split("/")[0] if "/" in run.name else "."
            groups.setdefault(group, []).append(run)
        if len(groups) > 1:
            for group in sorted(groups):
                print_summary(groups[group], f" in {group}")
        else:
            report.note("--by-dir: there are no subdirectories to break down.")
    report.note(
        "A threshold limits one expression, but NumConstraintScopes counts "
        "the whole file, so 'over' can overstate a file with many expressions."
    )

    print("\n".join(report.terminal))
    if args.report:
        report_path = out_dir / "report.md"
        report_path.write_text("\n".join(report.markdown) + "\n")
        _log(f"\nThe report is saved in {report_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compile files with several compilers, using the flags of "
        "their RUN lines, and compare their -stats-output-dir statistics."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="compile only this Swift file instead of the test directory",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_TEST_DIR,
        help="directory whose .swift files are compiled, including "
        "subdirectories (default: validation-test/Sema/type_checker_perf)",
    )
    parser.add_argument(
        "--stat",
        action="append",
        help="statistic to compare (repeatable, default: NumConstraintScopes, "
        "which is always included because the scope thresholds refer to it)",
    )
    parser.add_argument(
        "--baseline",
        help="column the percentages and the summary are relative to, for "
        "example snap1-off. A compiler name stands for its off column "
        "(default: main-off)",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="hide the rows where no column differs from the baseline",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also save the report as a markdown file, report.md, at the top "
        "level of --out (--no-report turns it off)",
    )
    parser.add_argument(
        "--by-dir",
        action="store_true",
        help="also print the summary tables for each subdirectory of --dir "
        "(files directly in --dir are grouped as '.')",
    )
    parser.add_argument(
        "--compare",
        action="append",
        metavar="MEASURED:REFERENCE",
        help="add a row to the pair table that compares the measured column "
        "with the reference column, for example wip-on:wip-off (repeatable, "
        "default: wip-on:main-on). A compiler name stands for its off column.",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="do not colour the changed cells"
    )
    parser.add_argument(
        "--vcs",
        choices=["auto", "jj", "git"],
        default="auto",
        help="version control system of the swift repository",
    )
    parser.add_argument(
        "--force-vcs",
        action="store_true",
        help="allow --vcs git in a colocated jj repository",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="fetch from origin before resolving main@origin",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="with git, stash uncommitted files while main is built",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        metavar="NAME=DATE",
        help="add a swift.org snapshot (yyyy-MM-dd) as a column (repeatable)",
    )
    parser.add_argument(
        "--snapshot-branch",
        help="swift_snapshot_tool branch flag, e.g. development or release_6_2",
    )
    parser.add_argument(
        "--snapshot-tool",
        help="command for swift_snapshot_tool (or set SWIFT_SNAPSHOT_TOOL)",
    )
    parser.add_argument(
        "--workspace",
        default=_DEFAULT_WORKSPACE,
        help="where swift_snapshot_tool downloads toolchains",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=_DEFAULT_BUILD_DIR,
        help="the swift build directory",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("compare_stats"), help="output directory"
    )
    parser.add_argument(
        "--extra",
        default=_DEFAULT_EXTRA_FLAGS,
        help="frontend flags passed to every compiler, before the RUN line flags",
    )
    parser.add_argument(
        "--on-flags",
        default=_DEFAULT_ON_FLAGS,
        help="frontend flags that make up the on column of every compiler; "
        "flags on a RUN line come after them and win",
    )
    parser.add_argument("--sdk", help="SDK path (default: xcrun --show-sdk-path)")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse the compilers saved in the output directory",
    )
    parser.add_argument(
        "--copy-lib",
        action="store_true",
        help="copy the host libraries next to each saved compiler (about 250 MB "
        "each) instead of linking to the build directory's lib directory. A "
        "link is the default, but it follows whatever the build directory "
        "holds when the compilers run, which is the state of the last build.",
    )
    args = parser.parse_args()
    args.stat = args.stat or [_SCOPE_STAT]
    if _SCOPE_STAT not in args.stat:
        args.stat.insert(0, _SCOPE_STAT)
    args.build_dir = args.build_dir.resolve()
    args.dir = args.dir.resolve()
    return args


def main():
    args = _parse_args()
    out_dir = args.out.resolve()
    try:
        files = _find_files(args)
        # Check the column names before anything is built.
        valid = _valid_columns(args)
        if args.baseline:
            args.baseline = _resolve_column(args.baseline, valid)
        args.pairs = _parse_pairs(args.compare, valid)
        runs, skipped = _plan_runs(files)
        for note in skipped:
            _log(note)
        if not runs:
            raise Failure("There is nothing to run.")
        _log(f"{len(runs)} runs from {len(files)} files, {len(skipped)} skipped.")
        for stale in ("stats", "logs"):
            if (out_dir / stale).exists():
                shutil.rmtree(out_dir / stale)
        compilers = _collect_compilers(args, out_dir)
        sdk = _sdk_path(args.sdk)
        have_stats = {}
        skipped_runs = {}
        for compiler in compilers:
            _log(f"\n{compiler.name}: compiling {len(runs)} runs ...")
            have_stats[compiler.name], skipped_runs[compiler.name] = _run_compiler(
                compiler, runs, args, sdk, out_dir
            )
        # A column in which every run was skipped has nothing to show.
        omitted = [
            compiler.name
            for compiler in compilers
            if len(skipped_runs[compiler.name]) == len(runs)
        ]
        for name in omitted:
            _log(f"{name}: every run was skipped, the compiler lacks an option.")
        if args.baseline in omitted:
            raise Failure(f"The baseline '{args.baseline}' was skipped entirely.")
        compilers = [c for c in compilers if c.name not in omitted]
        if not compilers:
            raise Failure("Every column was skipped.")
        _print_table(compilers, runs, have_stats, skipped_runs, omitted, args, out_dir)
    except Failure as error:
        _log(f"error: {error}")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
