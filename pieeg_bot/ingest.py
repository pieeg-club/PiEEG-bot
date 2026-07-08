"""Ingestion: turn the documentation repositories into ``data/index.json``.

Pipeline:

1. ``git clone`` (or pull) each source repo into ``data/sources/``.
2. Walk its content directory for ``.md`` / ``.mdx`` files.
3. Split each file into chunks by markdown heading. A heading is a natural
   unit and gives us a deep link (``page#anchor``) back to the live docs.
4. Embed every chunk and write text + vector + source URL to a single JSON file.

The corpus is small (a few dozen files), so there is no vector database and no
incremental indexing — a full rebuild takes seconds and keeps the code trivial.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config, embeddings

# Files that are navigation/config rather than content.
_SKIP_NAMES = {"_meta", "readme"}
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TITLE_RE = re.compile(r'^title:\s*["\']?(.*?)["\']?\s*$', re.MULTILINE)


@dataclass
class Chunk:
    """A single retrievable unit of documentation."""

    text: str
    url: str
    source: str
    title: str


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def clone_or_update(source: config.Source) -> Path:
    """Ensure ``source`` is cloned locally and up to date; return its path."""
    config.CLONE_DIR.mkdir(parents=True, exist_ok=True)
    # Derive a stable folder name from the repo URL.
    name = source.repo.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = config.CLONE_DIR / name
    if (dest / ".git").exists():
        _run_git(["fetch", "--depth", "1", "origin"], cwd=dest)
        _run_git(["reset", "--hard", "origin/HEAD"], cwd=dest)
    else:
        _run_git(["clone", "--depth", "1", source.repo, str(dest)])
    return dest


def _slugify_anchor(heading: str) -> str:
    """GitHub-style heading anchor: lowercase, punctuation stripped, spaces -> '-'."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def _page_url(source: config.Source, content_root: Path, path: Path) -> str:
    """Map a markdown file path to its public page URL."""
    rel = path.relative_to(content_root).with_suffix("")
    parts = list(rel.parts)
    # An index file resolves to its parent route.
    if parts and parts[-1].lower() == "index":
        parts = parts[:-1]
    suffix = "/".join(parts)
    return f"{source.url_base}/{suffix}" if suffix else source.url_base


def _doc_title(text: str, path: Path) -> str:
    """Prefer the frontmatter title, then the first H1, then the filename."""
    match = _TITLE_RE.search(text[:1000])
    if match:
        return match.group(1).strip()
    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            return heading.group(2).strip()
    return path.stem.replace("-", " ").title()


def chunk_file(source: config.Source, content_root: Path, path: Path) -> list[Chunk]:
    """Split one markdown file into heading-delimited chunks."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    body = _FRONTMATTER_RE.sub("", raw, count=1)
    page_url = _page_url(source, content_root, path)
    doc_title = _doc_title(raw, path)

    chunks: list[Chunk] = []
    cur_heading = doc_title
    cur_anchor = ""
    cur_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(cur_lines).strip()
        if not text:
            return
        url = f"{page_url}#{cur_anchor}" if cur_anchor else page_url
        # Prefix the breadcrumb so the embedding captures document context.
        body_text = f"# {doc_title} — {cur_heading}\n\n{text}"
        chunks.append(
            Chunk(text=body_text, url=url, source=source.name, title=cur_heading)
        )

    for line in body.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            cur_lines = []
            cur_heading = heading.group(2).strip()
            cur_anchor = _slugify_anchor(cur_heading)
        else:
            cur_lines.append(line)
    flush()
    return chunks


def collect_chunks() -> list[Chunk]:
    """Clone every source and return all chunks across the corpus."""
    all_chunks: list[Chunk] = []
    for source in config.SOURCES:
        repo_path = clone_or_update(source)
        content_root = repo_path / source.content_dir
        if not content_root.exists():
            print(f"  ! {source.name}: {source.content_dir} not found, skipping")
            continue
        files = sorted(
            p
            for p in content_root.rglob("*")
            if p.suffix in {".md", ".mdx"} and p.stem.lower() not in _SKIP_NAMES
        )
        source_chunks: list[Chunk] = []
        for path in files:
            source_chunks.extend(chunk_file(source, content_root, path))
        print(f"  - {source.name}: {len(files)} files -> {len(source_chunks)} chunks")
        all_chunks.extend(source_chunks)
    return all_chunks


def _embed_all(chunks: list[Chunk]) -> list[list[float]]:
    """Embed all chunk texts locally, preserving order."""
    vectors: list[list[float]] = []
    batch_size = 256
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embeddings.embed_passages([c.text for c in batch]))
        print(f"    embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")
    return vectors


def build_index() -> None:
    """Full ingestion run: clone, chunk, embed and write ``index.json``."""
    print("Collecting documentation chunks...")
    chunks = collect_chunks()
    if not chunks:
        raise SystemExit("No documentation chunks found; nothing to index.")

    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL}...")
    vectors = _embed_all(chunks)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": config.EMBED_MODEL,
        "chunks": [asdict(c) for c in chunks],
        "vectors": vectors,
    }
    config.INDEX_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {config.INDEX_PATH}")


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
