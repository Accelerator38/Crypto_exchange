"""Bounded final research archive. Does not import trading or training modules."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATE = "20261027"
DEST = ROOT / "Archives" / ("genetic_primus_final_" + DATE)
DOCS = ROOT / "docs" / ("genetic_primus_final_archive_" + DATE)
REPORTS = ROOT / "Reports/Exia/Genetic_Primus"
BRANCH = "codex/genetic-primus-final-archive-20261027"
TAG = "genetic-primus-final-20261027"
REPO = "Accelerator38/Crypto_exchange"
TEXT = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".ps1", ".sh", ".csv", ".html", ".svg"}
RESULTS = {".md", ".json", ".png", ".svg", ".npz", ".joblib", ".csv", ".txt"}
OMIT_PARTS = {"__pycache__", ".git", "node_modules", ".pytest_cache"}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "aws_access_key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    "openai_key": re.compile(rb"\bsk-(?:proj-|svcacct-)[A-Za-z0-9_-]{40,}"),
}


def rel(p):
    return p.relative_to(ROOT).as_posix()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, env=None, data=None):
    run = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=ROOT,
                         input=data, capture_output=True, env=env)
    if run.returncode:
        raise RuntimeError("git " + args[0] + " failed: " + run.stderr.decode("utf-8", "replace")[:400])
    return run.stdout


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def files_under(path):
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
            and not any(x in OMIT_PARTS for x in p.parts) and p.suffix != ".pyc"]


def research_sources():
    selected = set()
    for top in ("tools", "configs", "docs", "tests"):
        for p in files_under(ROOT / top):
            if "genetic_primus" in p.relative_to(ROOT / top).as_posix().lower():
                selected.add(p)
    for folder in ("src/exia/genetic_primus", "src/exia/genetic_primus_statistics", "src/exia/genetic_primus_v8_clean", "src/simple_research"):
        selected.update(files_under(ROOT / folder))
    return sorted(p for p in selected if p.suffix in TEXT or p.name == ".gitattributes")


def public_result(p):
    relative = p.relative_to(REPORTS)
    parts = [x.lower() for x in relative.parts]
    if any("portable" in x or "wheelhouse" in x or x in {"site-packages", "data", "raw", "external", "vendor"} for x in parts):
        return False
    if any(x.startswith(".env") or "credentials" in x or "secret" in x for x in parts):
        return False
    if p.suffix not in RESULTS:
        return False
    # Keep summary results and decision ledgers; raw market inputs/features stay local.
    if p.suffix == ".npz" and re.search(r"(?:ohlcv|features|market_data|orderbook|raw_book)", p.name, re.I):
        return False
    return True


def inventory():
    source = research_sources()
    all_reports = files_under(REPORTS)
    public = [p for p in all_reports if public_result(p)]
    local = set(source + all_reports)
    for folder in (ROOT / "Retrodate").glob("genetic_primus*"):
        if folder.is_dir():
            local.update(files_under(folder))
    # Include exact research inputs referenced by frozen registries, when available locally.
    for p in source:
        if p.suffix != ".json" or p.stat().st_size > 5_000_000:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError):
            continue
        stack = [obj]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.keys())
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str) and value.startswith(("Retrodate/", "Retrodate\\")):
                q = ROOT / value
                if q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(ROOT):
                    local.add(q)
    # Preserve currently tracked project work locally, including uncommitted changes,
    # while leaving unrelated runtime changes out of the public research commit.
    for name in git("ls-files", "-z").decode("utf-8").split("\0"):
        if not name:
            continue
        q = ROOT / name
        if q.is_file() and not q.is_symlink() and not q.name.startswith(".env"):
            local.add(q)
    # Small read-friendly evidence in git; complete eligible results in a release ZIP.
    git_results = [p for p in public if p.suffix in {".md", ".png", ".svg"}
                   or (p.suffix == ".json" and p.stat().st_size <= 131072)]
    git_files = sorted(set(source + git_results))
    return dict(source=source, public=sorted(public), local=sorted(local), git=git_files)


def scan(paths):
    findings = []
    for p in paths:
        if p.name.startswith(".env") or re.search(r"(?:^|[_.-])(?:credentials|private_key|apikeys)(?:[_.-]|$)", p.name, re.I):
            findings.append(dict(path=rel(p), rule="sensitive_filename"))
        if p.suffix not in TEXT:
            continue
        content = p.read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(dict(path=rel(p), rule=name))
        credential = re.compile(rb'''["'](?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|password)["']\s*[:=]\s*["']([^"'\r\n]{16,})["']''', re.I)
        for match in credential.finditer(content):
            value = match.group(1).lower()
            if not any(word in value for word in (b"your", b"example", b"dummy", b"test", b"changeme", b"placeholder", b"getenv", b"environ", b"${", b"<", b"x" * 8)):
                findings.append(dict(path=rel(p), rule="credential_literal_review_required"))
    return findings


def record(paths):
    return {rel(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths}


def prepare():
    assert not (DEST / "archive_plan.json").exists(), "archive plan already frozen"
    assert not git("ls-files", "-u").strip(), "unmerged files"
    inv = inventory()
    findings = scan(sorted(set(inv["source"] + inv["public"])))
    if findings:
        print(json.dumps(dict(secret_scan_findings=findings[:8], total=len(findings)), ensure_ascii=False))
        return
    versions = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "matplotlib"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    plan = dict(date="2026-10-27", status="CONSERVED_RESEARCH_ONLY", branch=BRANCH, tag=TAG,
                source_head=git("rev-parse", "HEAD").decode().strip(),
                source_branch=git("branch", "--show-current").decode().strip(),
                source_index_sha256=sha(ROOT / ".git/index"),
                source_status_sha256=hashlib.sha256(git("status", "--porcelain=v1", "-z", "-uall")).hexdigest(),
                python=sys.version, packages=versions, secret_scan_findings=0,
                archive_tool_sha256=sha(Path(__file__)), files=record(inv["local"]),
                public_files=[rel(p) for p in sorted(set(inv["source"] + inv["public"]))],
                git_files=[rel(p) for p in inv["git"]],
                exclusions="No API credentials, current trading-runtime edits, external source market datasets, venv or downloaded wheels published. Full research results and available referenced data retained locally.",
                credentials_allowed=False, orders_enabled=False, paper=False, live=False,
                runtime_authority=False, network_allowed=False, promotion_authority=False)
    write(DEST / "archive_plan.json", plan)
    print(json.dumps(dict(frozen_files=len(plan["files"]), public_files=len(plan["public_files"]), git_files=len(plan["git_files"]), secret_findings=0)))


def pack(kind):
    plan = json.loads((DEST / "archive_plan.json").read_text(encoding="utf-8"))
    names = plan["public_files"] if kind == "public" else sorted(plan["files"])
    name = ("genetic-primus-research-artifacts-" if kind == "public" else "genetic-primus-local-preservation-") + DATE + ".zip"
    path = DEST / name
    assert not path.exists(), "archive already exists"
    for file in names:
        assert sha(ROOT / file) == plan["files"][file]["sha256"], file
    # STORE already compressed numerical/model binaries; DEFLATE text and source code.
    with zipfile.ZipFile(path, "x", allowZip64=True) as z:
        for file in names:
            p = ROOT / file
            method = zipfile.ZIP_STORED if p.suffix in {".npz", ".png", ".joblib", ".zip", ".whl", ".parquet"} else zipfile.ZIP_DEFLATED
            z.write(p, file, compress_type=method, compresslevel=4 if method == zipfile.ZIP_DEFLATED else None)
        subset = {file: plan["files"][file] for file in names}
        z.writestr("ARCHIVE_CONTENT_MANIFEST.json", json.dumps(dict(status=plan["status"], files=subset, python=plan["python"], packages=plan["packages"]), ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_DEFLATED)
    info = dict(file=name, bytes=path.stat().st_size, sha256=sha(path), members=len(names) + 1,
                public=(kind == "public"), source_head=plan["source_head"])
    write(DEST / (kind + "_archive.json"), info)
    print(json.dumps(info))


def verify_pack(kind):
    info = json.loads((DEST / (kind + "_archive.json")).read_text(encoding="utf-8"))
    path = DEST / info["file"]
    assert sha(path) == info["sha256"]
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("ARCHIVE_CONTENT_MANIFEST.json"))
        assert len(z.infolist()) == len(manifest["files"]) + 1
        for file, rec in manifest["files"].items():
            h = hashlib.sha256()
            with z.open(file) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            assert h.hexdigest() == rec["sha256"], file
    write(DEST / (kind + "_verification.json"), dict(verified_files=len(manifest["files"]), sha256=info["sha256"], all_member_sha256_match=True))
    print(json.dumps(dict(kind=kind, verified_files=len(manifest["files"]), sha256=info["sha256"])))


def commit():
    plan = json.loads((DEST / "archive_plan.json").read_text(encoding="utf-8"))
    assert sha(ROOT / ".git/index") == plan["source_index_sha256"], "original index changed"
    assert git("rev-parse", "HEAD").decode().strip() == plan["source_head"]
    assert not git("ls-files", "-u").strip()
    if subprocess.run(["git", "show-ref", "--verify", "--quiet", "refs/heads/" + BRANCH], cwd=ROOT).returncode == 0:
        raise RuntimeError("archive branch already exists")
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = str(DEST / "archive.gitindex")
    git("read-tree", plan["source_head"], env=env)
    rows = []
    for file in plan["git_files"]:
        p = ROOT / file
        assert sha(p) == plan["files"][file]["sha256"], file
    input_paths = "\n".join(json.dumps(file, ensure_ascii=False) for file in plan["git_files"]) + "\n"
    blobs = git("hash-object", "-w", "--no-filters", "--stdin-paths", data=input_paths.encode()).decode().splitlines()
    assert len(blobs) == len(plan["git_files"])
    for file, blob in zip(plan["git_files"], blobs):
        assert re.fullmatch(r"[0-9a-f]{40}", blob)
        rows.append("100644 " + blob + "\t" + file + "\0")
    # Archive-only root attributes preserve exact pinned bytes on other checkouts.
    old_attributes = git("show", plan["source_head"] + ":.gitattributes")
    extra = b"\n# Frozen research archive: preserve pinned bytes.\nsrc/exia/genetic_primus/** -text\nsrc/exia/genetic_primus_statistics/** -text\nsrc/exia/genetic_primus_v8_clean/** -text\nsrc/simple_research/** -text\nReports/Exia/Genetic_Primus/** -text\nconfigs/*genetic_primus* -text\ntools/*genetic_primus* -text\ntests/*genetic_primus* -text\ndocs/*GENETIC_PRIMUS* -text\ndocs/genetic_primus_final_archive_20261027/** -text\n"
    attr = git("hash-object", "-w", "--stdin", data=old_attributes + extra).decode().strip()
    rows.append("100644 " + attr + "\t.gitattributes\0")
    git("update-index", "-z", "--index-info", env=env, data="".join(rows).encode())
    tree = git("write-tree", env=env).decode().strip()
    message = "Archive Genetic Primus research at 2026-10-27\n\nConserve causal evaluators, model artifacts, negative results and publication plans. NoTrade remains the deployment decision. No trading authority granted.\n"
    commit_sha = git("commit-tree", tree, "-p", plan["source_head"], data=message.encode()).decode().strip()
    git("update-ref", "refs/heads/" + BRANCH, commit_sha, "0" * 40)
    assert sha(ROOT / ".git/index") == plan["source_index_sha256"]
    write(DEST / "git_archive.json", dict(branch=BRANCH, tag=TAG, commit=commit_sha, tree=tree, source_head=plan["source_head"], original_index_unchanged=True, original_branch_unchanged=True))
    print(json.dumps(dict(branch=BRANCH, commit=commit_sha, included_files=len(rows), original_index_unchanged=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["inventory", "prepare", "pack-public", "pack-local", "verify-public", "verify-local", "commit"])
    args = parser.parse_args()
    if args.stage == "inventory":
        inv = inventory()
        print(json.dumps({key: dict(files=len(ps), mb=round(sum(p.stat().st_size for p in ps) / 1e6, 2)) for key, ps in inv.items()}))
    elif args.stage == "prepare":
        prepare()
    elif args.stage.startswith("pack-"):
        pack(args.stage.split("-")[1])
    elif args.stage.startswith("verify-"):
        verify_pack(args.stage.split("-")[1])
    else:
        commit()
