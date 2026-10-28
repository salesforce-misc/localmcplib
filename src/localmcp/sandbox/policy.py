"""Shared filesystem validation for policy-bound command sandboxes."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from localmcp.sandbox.base import RootAccess, SandboxError, SandboxRoot


def _validated_roots(roots: tuple[SandboxRoot, ...]) -> tuple[SandboxRoot, ...]:
    resolved: list[SandboxRoot] = []
    by_path: dict[Path, RootAccess] = {}
    for root in roots:
        try:
            path = root.path.resolve(strict=True)
        except OSError as exc:
            raise SandboxError(f"sandbox root does not exist: {root.path}") from exc
        if not path.is_dir():
            raise SandboxError(f"sandbox root is not a directory: {root.path}")
        previous = by_path.get(path)
        if previous is not None:
            if previous != root.access:
                raise SandboxError(f"sandbox root has conflicting access grants: {path}")
            continue
        by_path[path] = root.access
        resolved.append(SandboxRoot(path, root.access))

    # A read-only root inside a read-write root is write-protected there; a read-write
    # root inside that protected tree could not be granted, and normalizing below would
    # silently fold it into the outer read-write root.
    for protected in _protected_roots(tuple(resolved)):
        for root in resolved:
            if root.access == RootAccess.READ_WRITE and root.path.is_relative_to(protected):
                raise SandboxError(f"read-write root is inside a write-protected read-only root: {root.path}")

    normalized: list[SandboxRoot] = []
    for index, root in enumerate(resolved):
        if index != 0 and any(
            root.access == existing.access and root.path.is_relative_to(existing.path) for existing in normalized
        ):
            continue
        normalized = [
            existing
            for existing_index, existing in enumerate(normalized)
            if existing_index == 0 or existing.access != root.access or not existing.path.is_relative_to(root.path)
        ]
        normalized.append(root)
    return tuple(normalized)


def _denied_paths_in_roots(denied_paths: tuple[Path, ...], roots: tuple[SandboxRoot, ...]) -> tuple[Path, ...]:
    """Make denied paths absolute without following symlinks, dedupe them, and require each inside a root."""
    normalized = tuple(dict.fromkeys(Path(os.path.abspath(path)) for path in denied_paths))
    if any(not any(path.is_relative_to(root.path) for root in roots) for path in normalized):
        raise SandboxError("sandbox denied path is outside its declared roots")
    return normalized


def _check_denied_paths(denied_paths: tuple[Path, ...], kind: str = "denied path") -> None:
    """Reject aliases that would undermine path-based access restrictions.

    A path through a symlink can name a different policy target, and a file with
    another hard link can remain accessible under that other name. ``kind`` names
    the paths in errors: denied paths are hidden, and write-protected read-only
    roots are checked the same way.
    """

    def uncheckable(error: OSError) -> None:
        # A file that is gone has no other name.
        if not isinstance(error, FileNotFoundError):
            raise SandboxError(f"sandbox {kind} cannot be checked for hard links: {error.filename}") from error

    for denied in denied_paths:
        if Path(os.path.realpath(denied)) != denied:
            raise SandboxError(f"sandbox {kind} goes through a symlink and cannot be enforced: {denied}")
        names = [denied]
        if denied.is_dir():
            names.extend(
                Path(directory, name) for directory, _, files in os.walk(denied, onerror=uncheckable) for name in files
            )
        for name in names:
            try:
                status = name.lstat()
            except OSError as exc:
                uncheckable(exc)
                continue
            if not stat.S_ISDIR(status.st_mode) and status.st_nlink > 1:
                raise SandboxError(f"sandbox {kind} has another hard link and cannot be enforced: {name}")


def _protected_roots(roots: tuple[SandboxRoot, ...]) -> tuple[Path, ...]:
    """Return the read-only roots that lie inside a read-write root, which must be write-protected."""
    return tuple(
        root.path
        for root in roots
        if root.access == RootAccess.READ_ONLY
        and any(other.access == RootAccess.READ_WRITE and root.path.is_relative_to(other.path) for other in roots)
    )


def _check_scratch(scratch: Path, roots: tuple[SandboxRoot, ...]) -> None:
    """Fail unless ``scratch`` is a real path outside every root, and so outside every denied path.

    Keeping scratch separate prevents root grants or denials from changing its authority.
    """
    if Path(os.path.realpath(scratch)) != scratch or any(scratch.is_relative_to(root.path) for root in roots):
        raise SandboxError(f"sandbox temporary directory must be a real path outside its roots: {scratch}")


def _path_ancestors(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    ancestors: set[Path] = set()
    for path in paths:
        ancestors.update(parent for parent in path.parents if parent != Path(path.anchor))
    return tuple(sorted(ancestors, key=os.fspath))
