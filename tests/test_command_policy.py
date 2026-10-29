"""Accept/reject contract for the shared, pure Bash command check."""

from __future__ import annotations

from pathlib import Path

import pytest

from localmcp.sandbox import CommandResult
from localmcp.sandbox.command_policy import check_command, rejected_command
from localmcp.sandbox.runtime import command_environment

TOOLS = ("cat", "head", "grep", "find", "git", "awk", "xargs", "sed", "sort", "wc", "sleep", "touch")


@pytest.mark.parametrize(
    "command",
    [
        "cat file.txt",
        "cat *.txt | head -5",
        "git log --oneline -3 && git diff HEAD~1 --stat",
        "git ls-files | xargs wc -l",
        "find . -name '*.py' -not -path './.git/*'",
        "awk 'NR==1 {print $0}' data",
        "echo head=$(git rev-parse --short HEAD)",
        'echo "$(cat "$(echo path)")"',
        'f=$(git ls-files | grep -m1 "py$"); git blame -L1,2 "$f"',
        "printf '%s\\n' hello >out 2>&1",
        "cat <<< 'hello'",
        "cat <in >>out",
        "echo a &>out",
        'for f in *.txt; do cat "$f"; done',
        "while [ ! -e ready ]; do sleep 0.05; done",
        "until test -f ready; do sleep 0.05; done",
        "if test -f a; then cat a; elif false; then :; else echo absent; fi",
        'case "$f" in *.py|*.txt) cat "$f";; *) echo ignored;; esac',
        "(sleep 1; touch late) >/dev/null 2>&1 & touch started",
        "{ cat a; echo done; }",
        "! false",
        "command -v git cat",
        "command -V git",
        "read -r line < ready",
        'printf "%s" "$TMPDIR"',
        "printf '100%%n\\n'",
        'echo "${name:-$(cat fallback)}"',
        'echo "${#name}"',
        'test "$x" = y',
        '[ "$x" -eq 3 ]',
        "echo '$(unlisted)' # unlisted is inert text",
        'echo "\\$(unlisted)"',
        "echo $'hello\\nworld'",
        "exit 7",
        "kill -TERM $$",
        "wait",
        "for i in a b; do continue; break; done",
        "",
        "# comment only",
    ],
)
def test_supported_bash(command: str) -> None:
    assert check_command(command, TOOLS) is None


@pytest.mark.parametrize(
    "command",
    [
        "unlisted",
        "cat a; unlisted; echo done",
        "true || unlisted",
        "echo $(unlisted)",
        'echo "${x:-$(unlisted)}"',
        "cat >$(unlisted)",
        "for i in $(unlisted); do echo x; done",
        "if false; then unlisted; fi",
        "/bin/cat a",
        "./cat a",
        "dir/cat a",
        "'cat' a",
        '"cat" a',
        "c\\at a",
        "$tool a",
        "$(echo cat) a",
        "c{a,b}t a",
        "c*t a",
        "eval 'cat a'",
        "source code.sh",
        ". code.sh",
        "exec cat a",
        "builtin cat a",
        "command cat a",
        "command -p cat a",
        "command -v $tool",
        "command -v /bin/cat",
        "enable -f payload.so payload",
        "alias cat=unlisted",
        "trap 'unlisted' EXIT",
        "set -- unlisted; $1",
        "shopt -s extglob",
        "export PATH=.",
        "declare -n ref=PATH",
        "cat() { unlisted; }; cat",
        "function cat { echo hi; }",
        "time unlisted",
        "coproc cat a",
        "echo <(cat a)",
        "echo `cat a`",
        "cat <<EOF\n$(unlisted)\nEOF",
        'echo "$(cat"',
        "if true; then echo hi",
        "echo $((1+2))",
        "((x++))",
        "for ((i=0;i<3;i++)); do echo hi; done",
        '[[ "$x" == y ]]',
        "echo ${x@P}",
        "echo ${!x}",
        "echo ${x:1}",
        "echo ${PATH:=.}",
        "echo ${a[$(unlisted)]}",
        "PATH=. cat a",
        "IFS=x",
        "BASH_ENV=code.sh cat a",
        "BASH_CMDS[cat]=./payload",
        "LC_ALL=x",
        "for PATH in .; do cat a; done",
        "printf -v PATH %s .",
        "printf %n PATH",
        "printf '%10n' PATH",
        "printf %ln PATH",
        "printf '%hn' PATH",
        "printf '%-5zn' PATH",
        "printf '%%%jn' PATH",
        'printf "$format" value',
        "read PATH",
        "read -a BASH_CMDS",
        "read $name",
        "test -v 'a[$(unlisted)]'",
        "test $arguments",
        "[ -v 'a[$(unlisted)]' ]",
        "[ $arguments ]",
        "wait -p PATH",
        "exit $code",
    ],
)
def test_rejected_bash(command: str) -> None:
    assert check_command(command, TOOLS) is not None


def test_tool_grants_are_exact_and_builtin_dispatch_is_never_implicit() -> None:
    assert check_command("python3 -c 'print(1)'", TOOLS)
    assert check_command("python3 -c 'print(1)'", ("python3",)) is None
    assert check_command("eval 'unlisted'", ("eval",))
    assert check_command("echo builtin", ()) is None


def test_granted_tools_may_interpret_arguments_as_programs() -> None:
    # The check makes no claims about code interpreted or launched by a tool.
    assert check_command("awk 'BEGIN {system(\"unlisted\")}'", TOOLS) is None
    assert check_command("find . -exec /bin/unlisted {} +", TOOLS) is None


def test_deep_bash_is_rejected_or_checked_without_python_recursion() -> None:
    assert check_command("[ " + "! " * 1500 + "-e x ]", ()) is None
    assert check_command("echo " + "$(echo " * 1000 + "$(unlisted)" + ")" * 1000, ())


def test_unencodable_command_fails_closed() -> None:
    assert check_command("echo \ud800", ()) == "cannot parse command"


@pytest.mark.parametrize(
    "variable",
    [
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_FUTURE_SETTING",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FUTURE_SETTING",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "DEVELOPER_DIR",
        "xcrun_db",
        "RANDOM",
        "SRANDOM",
        "OPTIND",
        "SECONDS",
        "TMOUT",
    ],
)
@pytest.mark.parametrize(
    "template", ["{name}=payload cat file", "{name}=payload", "for {name} in payload; do :; done", "read {name}"]
)
def test_shell_loader_and_fixed_environment_cannot_be_changed(variable: str, template: str) -> None:
    assert check_command(template.format(name=variable), TOOLS)


@pytest.mark.parametrize("developer_directory", [None, Path("/toolchain")])
def test_every_fixed_environment_variable_is_protected(developer_directory: Path | None) -> None:
    for variable in command_environment(Path("/scratch"), developer_directory=developer_directory):
        for command in (f"{variable}=payload cat file", f"read {variable}", f"for {variable} in value; do :; done"):
            assert check_command(command, TOOLS), command


def test_numeric_shell_variables_cannot_evaluate_unchecked_commands() -> None:
    command = "value='array[$(unlisted)]'; RANDOM=$value"
    assert check_command(command, TOOLS)


def test_rejection_result_is_shared_by_both_backends() -> None:
    assert rejected_command("cat file", TOOLS) is None
    assert rejected_command("ls", TOOLS) == CommandResult(
        exit_code=126, stdout="", stderr="localmcp: command is not allowed: ls"
    )
