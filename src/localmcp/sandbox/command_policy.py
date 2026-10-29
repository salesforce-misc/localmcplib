"""Conservative, host-side Bash syntax and command-name checks.

This checks the submitted shell program, including nested substitutions. It
does not inspect arguments interpreted as programs by a granted tool.
"""

from __future__ import annotations

import re
from pathlib import Path

import tree_sitter_bash
from tree_sitter import Language, Node, Parser

from localmcp.sandbox.runtime import command_environment

_LANGUAGE = Language(tree_sitter_bash.language())
_COMMAND_NAME = re.compile(rb"(?:[A-Za-z0-9_][A-Za-z0-9_.+:-]*|:)")
_VARIABLE_NAME = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_ENVIRONMENT_VARIABLES = frozenset(
    name.encode() for name in command_environment(Path("/"), developer_directory=Path("/"))
)
_BUILTINS = frozenset(
    {":", "true", "false", "echo", "printf", "pwd", "cd", "test", "read", "exit", "break", "continue", "wait", "kill"}
)
_DISPATCH = frozenset(
    {
        "eval",
        "source",
        ".",
        "exec",
        "builtin",
        "enable",
        "alias",
        "unalias",
        "trap",
        "set",
        "shopt",
        "export",
        "readonly",
        "declare",
        "typeset",
        "local",
        "let",
        "unset",
        "time",
        "coproc",
        "getopts",
        "mapfile",
        "readarray",
        "hash",
    }
)
_SHELL_VARIABLES = frozenset(
    {
        b"PATH",
        b"IFS",
        b"ENV",
        b"SHELLOPTS",
        b"CDPATH",
        b"GLOBIGNORE",
        b"FIGNORE",
        b"EXECIGNORE",
        b"PROMPT_COMMAND",
        b"PS0",
        b"PS1",
        b"PS2",
        b"PS3",
        b"PS4",
        b"POSIXLY_CORRECT",
        b"LANG",
        b"LANGUAGE",
        b"RANDOM",
        b"SRANDOM",
        b"SECONDS",
        b"LINENO",
        b"OPTIND",
        b"OPTERR",
        b"UID",
        b"EUID",
        b"PPID",
        b"HISTCMD",
        b"HISTSIZE",
        b"HISTFILESIZE",
        b"IGNOREEOF",
        b"COLUMNS",
        b"LINES",
        b"MAILCHECK",
        b"TMOUT",
    }
)
_NODES = frozenset(
    {
        "program",
        "list",
        "pipeline",
        "command",
        "command_name",
        "word",
        "number",
        "string",
        "string_content",
        "raw_string",
        "ansi_c_string",
        "concatenation",
        "simple_expansion",
        "expansion",
        "variable_name",
        "special_variable_name",
        "command_substitution",
        "redirected_statement",
        "file_redirect",
        "file_descriptor",
        "herestring_redirect",
        "variable_assignment",
        "variable_assignments",
        "for_statement",
        "while_statement",
        "do_group",
        "if_statement",
        "elif_clause",
        "else_clause",
        "case_statement",
        "case_item",
        "extglob_pattern",
        "compound_statement",
        "subshell",
        "negated_command",
        "test_command",
        "unary_expression",
        "binary_expression",
        "test_operator",
        "comment",
    }
)
_UNARY_TESTS = frozenset(
    {
        "-a",
        "-b",
        "-c",
        "-d",
        "-e",
        "-f",
        "-g",
        "-h",
        "-k",
        "-L",
        "-n",
        "-O",
        "-G",
        "-p",
        "-r",
        "-S",
        "-s",
        "-t",
        "-u",
        "-w",
        "-x",
        "-z",
    }
)
_BINARY_TESTS = frozenset({"=", "==", "!=", "-eq", "-ne", "-gt", "-ge", "-lt", "-le", "-ef", "-nt", "-ot"})


def _variable_allowed(name: bytes) -> bool:
    return (
        bool(_VARIABLE_NAME.fullmatch(name))
        and name not in _SHELL_VARIABLES | _ENVIRONMENT_VARIABLES
        and not name.startswith((b"BASH", b"LC_", b"LD_", b"DYLD_", b"GIT_CONFIG_"))
    )


def _literal(node: Node) -> str | None:
    """Read an argument without evaluating expansions or shell escapes."""
    value = node.text or b""
    if node.type in {"word", "number", "test_operator", "!", "=", "==", "!="} and not any(
        c in value for c in b"\\*?[]{}~"
    ):
        return value.decode()
    if node.type == "raw_string":
        return value[1:-1].decode()
    if node.type == "string" and all(child.type == "string_content" for child in node.named_children):
        if b"\\" not in value:
            return value[1:-1].decode()
    return None


def _test_arguments(arguments: list[Node]) -> bool:
    """Accept basic test forms without variable/array introspection or evaluation."""
    while arguments and _literal(arguments[0]) == "!":
        arguments = arguments[1:]
    if any(
        _literal(argument) is None and argument.type not in {"string", "raw_string", "ansi_c_string"}
        for argument in arguments
    ):
        return False
    if not arguments:
        return True
    if len(arguments) == 1:
        return True
    if len(arguments) == 2:
        return _literal(arguments[0]) in _UNARY_TESTS
    if len(arguments) == 3:
        return _literal(arguments[1]) in _BINARY_TESTS
    return False


def _test_tokens(node: Node) -> list[Node]:
    """Flatten grammar expression wrappers, preserving shell operand boundaries."""
    pending = [node]
    tokens = []
    while pending:
        current = pending.pop()
        if current.type in {"unary_expression", "binary_expression", "test_command"}:
            pending.extend(reversed(current.children))
        elif current.type not in {"[", "]"}:
            tokens.append(current)
    return tokens


def _builtin_reason(name: str, arguments: list[Node]) -> str | None:
    literals = [_literal(argument) for argument in arguments]
    if name == "command":
        if (
            len(literals) < 2
            or literals[0] not in {"-v", "-V"}
            or any(value is None or not _COMMAND_NAME.fullmatch(value.encode()) for value in literals[1:])
        ):
            return "command supports only literal -v/-V lookups"
    elif name == "printf":
        formats = literals[1:] if literals[:1] == ["--"] else literals
        if not formats or formats[0] is None or formats[0].startswith("-"):
            return "printf requires a literal format and cannot assign variables"
        if re.search(r"%[^%a-zA-Z]*n", formats[0]):
            return "printf cannot assign variables with %n"
    elif name == "read":
        names = literals[1:] if literals[:1] == ["-r"] else literals
        if any(value is None or not _variable_allowed(value.encode()) for value in names):
            return "read requires ordinary literal variable names, optionally after -r"
    elif name == "test" and not _test_arguments(arguments):
        return "unsupported test expression"
    elif name in {"wait", "break", "continue", "exit"}:
        if any(value is None or not re.fullmatch(r"[0-9]+", value) for value in literals):
            return f"{name} supports only literal numeric arguments"
    return None


def check_command(command: str, tools: tuple[str, ...]) -> str | None:
    """Return a rejection reason, or ``None`` for the supported Bash subset.

    No filesystem or environment reads occur here. Each call has its own parser
    so concurrent commands cannot share mutable parsing state.
    """
    try:
        tree = Parser(_LANGUAGE).parse(command.encode())
    except (UnicodeError, ValueError, RecursionError):
        return "cannot parse command"
    if tree.root_node.has_error:
        return "invalid or incomplete Bash syntax"
    granted = frozenset(tools)
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.is_missing or node.type == "ERROR":
            return "invalid or incomplete Bash syntax"
        if node.is_named and node.type not in _NODES:
            return f"unsupported Bash syntax: {node.type}"
        if node.type == "command":
            name_node = node.child_by_field_name("name")
            value = (name_node.text or b"") if name_node else b""
            if (
                not name_node
                or len(name_node.named_children) != 1
                or name_node.named_children[0].type != "word"
                or not _COMMAND_NAME.fullmatch(value)
            ):
                return "command names must be literal words without paths or expansions"
            name = value.decode()
            if name in _DISPATCH or name not in granted | _BUILTINS | {"command"}:
                return f"command is not allowed: {name}"
            arguments = node.children_by_field_name("argument")
            if reason := _builtin_reason(name, arguments):
                return reason
        elif node.type == "variable_assignment":
            name_node = node.child_by_field_name("name")
            if name_node is None or not _variable_allowed(name_node.text or b""):
                return "assignments cannot change shell, loader, or sandbox environment settings"
        elif node.type == "for_statement":
            variable = node.child_by_field_name("variable")
            if variable is None or not _variable_allowed(variable.text or b""):
                return "loop variables cannot change shell settings"
        elif node.type == "command_substitution" and not (node.text or b"").startswith(b"$("):
            return "only $(...) command substitutions are supported"
        elif node.type == "expansion":
            operators = [child.type for child in node.children if not child.is_named]
            if any(operator not in {"${", "}", "#", ":-", "-", ":+", "+", ":?", "?"} for operator in operators):
                return "unsupported parameter expansion"
        elif node.type == "test_command":
            if (node.text or b"").startswith(b"[[") or not _test_arguments(_test_tokens(node)):
                return "unsupported test expression"
        pending.extend(reversed(node.children))
    return None
