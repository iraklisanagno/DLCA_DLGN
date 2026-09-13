"""Exclusive artifacts and verification; never write to an incumbent path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent
REPO = ROOT.parents[2]
WORKSPACE = REPO.parents[1]
PRESERVATION = WORKSPACE / "preservation/unified_refinement_20260906"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("UR1 writes must stay inside its new directory")
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"refusing to replace artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(content)


def verify_incumbent(full=False):
    from experiments.coverage_dlgn.run_fourth_round import implementation_hash
    manifest = read(PRESERVATION / "manifest.json")
    if manifest["status"] != "verified":
        raise RuntimeError("missing verified recovery archive")
    if implementation_hash() != manifest["old_implementation_sha256"]:
        raise RuntimeError("incumbent implementation changed")
    certificate = read(OLD / "summary/fourth_round_completion_audit.json")
    for section in ("artifact_sha256", "document_sha256"):
        for name, digest in certificate[section].items():
            if sha(OLD / name) != digest:
                raise RuntimeError(f"incumbent evidence changed: {name}")
    checked = 0
    if full:
        for name, record in manifest["files"].items():
            path = WORKSPACE / name
            if "symlink" in record:
                if not path.is_symlink() or str(path.readlink()) != record["symlink"]:
                    raise RuntimeError(f"incumbent symlink changed: {name}")
            elif not path.is_file() or sha(path) != record["sha256"]:
                raise RuntimeError(f"incumbent file changed: {name}")
            checked += 1
    return dict(status="pass", incumbent_implementation_sha256=implementation_hash(),
                preservation_manifest_sha256=sha(PRESERVATION / "manifest.json"),
                original_files_checked=checked, full=full)


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(ROOT.rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    digest.update((ROOT / "PROTOCOL.md").read_bytes())
    return digest.hexdigest()
