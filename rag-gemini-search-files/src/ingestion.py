"""Document ingestion helpers for Gemini File Search.

This module includes utilities to create/resolve a file search store,
upload and import files with chunking configuration, and poll long running
operations until completion.
"""

from __future__ import annotations

import os
import time
import re
import unicodedata
import tempfile
import shutil
from collections.abc import Sequence
import argparse
from pathlib import Path

from google import genai
from google.genai import types

from config import RagConfig, load_rag_config


def create_client() -> genai.Client:
    """Create a Gemini API client using the ``GOOGLE_API_KEY`` environment variable.

    This does not rely on any dotenv loader; it expects the variable to be exported.

    Returns:
        genai.Client: Authenticated client.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is not defined.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Export it before running: export GOOGLE_API_KEY=..."
        )
    return genai.Client(api_key=api_key)


def ensure_store(client: genai.Client, display_name: str, *, verbose: bool = False) -> types.FileSearchStore:
    """Find an existing File Search Store by display name or create it.

    Args:
        client: Gemini API client.
        display_name: Human friendly display name to locate the store.

    Returns:
        types.FileSearchStore: Resolved or newly created store.
    """
    # Try to find existing store by display name (list + match)
    for store in client.file_search_stores.list():
        if getattr(store, "display_name", None) == display_name:
            if verbose:
                print(f"Found existing File Search Store: '{display_name}' -> {store.name}")
            return store
    created = client.file_search_stores.create(config={"display_name": display_name})
    if verbose:
        print(f"Created File Search Store: '{display_name}' -> {created.name}")
    return created


def _ascii_safe_name(name: str, max_len: int = 120) -> str:
    """Return an ASCII-only safe version of a file/display name.

    - Strips accents via NFKD normalization
    - Removes non [A-Za-z0-9._ -]
    - Trims and truncates to max_len
    """
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = re.sub(r"[^A-Za-z0-9._ \-]+", "", ascii_only).strip()
    return (ascii_only or "file")[:max_len]


def _needs_ascii_rename(name: str) -> bool:
    try:
        name.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def upload_and_import(
    client: genai.Client,
    store: types.FileSearchStore,
    file_path: Path,
    chunking_cfg: dict | None = None,
    display_file_name: str | None = None,
    *,
    verbose: bool = False,
) -> types.Operation:
    """Upload a file and import it into the given store.

    Args:
        client: Gemini API client.
        store: Target File Search Store.
        file_path: Path to the source file.
        chunking_cfg: Optional white space chunking configuration dict with
            keys "max_tokens_per_chunk" and "max_overlap_tokens".
        display_file_name: Optional display name used for citations.

    Returns:
        types.Operation: Long running operation representing the import.
    """
    safe_display = _ascii_safe_name(display_file_name or file_path.name)

    temp_dir: str | None = None
    temp_file: Path | None = None
    file_arg = file_path
    if _needs_ascii_rename(file_path.name):
        # Create a temporary ASCII-only copy for the multipart filename header
        temp_dir = tempfile.mkdtemp(prefix="ingest_")
        temp_file = Path(temp_dir) / _ascii_safe_name(file_path.name)
        shutil.copyfile(file_path, temp_file)
        if verbose:
            print(f"Using temp ASCII filename: '{temp_file.name}' for upload")
        file_arg = temp_file

    try:
        return client.file_search_stores.upload_to_file_search_store(
            file=str(file_arg),
            file_search_store_name=store.name,
            config={
                "display_name": safe_display,
                **({"chunking_config": {"white_space_config": chunking_cfg}} if chunking_cfg else {}),
            },
        )
    finally:
        # Cleanup temporary file if we created one
        if temp_file is not None:
            try:
                temp_file.unlink(missing_ok=True)
                Path(temp_dir).rmdir()  # type: ignore[arg-type]
            except Exception:
                pass


def poll_operation(
    client: genai.Client,
    operation: types.Operation,
    timeout_seconds: int,
    interval_seconds: int,
    *,
    verbose: bool = False,
) -> types.Operation:
    """Poll a long running operation until done or timeout.

    Args:
        client: Gemini API client.
        operation: Operation to poll.
        timeout_seconds: Maximum time to wait.
        interval_seconds: Sleep interval between checks.

    Returns:
        types.Operation: Final operation state.

    Raises:
        TimeoutError: If the operation does not finish in time.
    """
    start = time.time()
    current = operation
    while not current.done:
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Upload/import operation timed out")
        time.sleep(interval_seconds)
        current = client.operations.get(current)
        if verbose:
            print("… still processing")
    return current


def ingest_files(
    config: RagConfig,
    paths: Sequence[Path],
    *,
    verbose: bool = False,
    keep_going: bool = False,
    failures: list[tuple[Path, str]] | None = None,
) -> tuple[types.FileSearchStore, list[str]]:
    """Ingest a collection of files into a File Search Store.

    Args:
        config: RAG configuration with chunking and store parameters.
        paths: Iterable of paths to files to ingest.

    Returns:
        tuple: (store, imported_file_names)
    """
    client = create_client()
    store = ensure_store(client, config.store.display_name, verbose=verbose)

    chunking_cfg = {
        "max_tokens_per_chunk": config.chunking.max_tokens_per_chunk,
        "max_overlap_tokens": config.chunking.max_overlap_tokens,
    }

    imported_docs: list[str] = []
    for p in paths:
        if verbose:
            print(f"Uploading '{p.name}' to store {store.name}…")
        try:
            op = upload_and_import(
                client,
                store,
                p,
                chunking_cfg=chunking_cfg,
                display_file_name=p.stem,
                verbose=verbose,
            )
            final = poll_operation(
                client,
                op,
                timeout_seconds=config.store.timeout_seconds,
                interval_seconds=config.store.polling_interval_seconds,
                verbose=verbose,
            )
            if not final.done:
                raise RuntimeError(f"Operation not finished for file: {p}")
            if verbose:
                print(f"Imported '{p.name}' successfully.")
            imported_docs.append(p.name)
        except Exception as e:
            if verbose:
                print(f"FAILED to import '{p.name}': {e}")
            if failures is not None:
                failures.append((p, str(e)))
            if not keep_going:
                raise

    return store, imported_docs


def ingest_folder(
    config: RagConfig,
    folder: Path,
    *,
    verbose: bool = False,
    keep_going: bool = False,
    failures: list[tuple[Path, str]] | None = None,
) -> tuple[types.FileSearchStore, list[str]]:
    """Ingest all supported files in a folder (non-recursive).

    Args:
        config: RAG configuration.
        folder: Directory containing files to ingest.

    Returns:
        tuple: (store, imported_file_names)
    """
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    supported_exts = {
        ".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".json", ".csv",
    }
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
    if verbose:
        print(f"Found {len(files)} file(s) to ingest in '{folder}'.")
    return ingest_files(config, files, verbose=verbose, keep_going=keep_going, failures=failures)


def main() -> None:
    """Simple CLI to ingest files with optional verbosity and custom folder.

    Usage examples:
        - Ingest default 'files' folder with progress:
            python src/ingestion.py --verbose
        - Ingest a specific folder:
            python src/ingestion.py --folder /path/to/folder --verbose
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into a Gemini File Search Store",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default folder with progress\n"
            "  python src/ingestion.py --verbose\n\n"
            "  # Custom folder and continue on errors\n"
            "  python src/ingestion.py --folder /data/docs --verbose --keep-going\n\n"
            "  # Only ingest markdown and txt files\n"
            "  python src/ingestion.py --extensions .md .txt --verbose\n\n"
            "  # Dry run (show what would be ingested without uploading)\n"
            "  python src/ingestion.py --dry-run --verbose\n"
        ),
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder to ingest files from (defaults to project_root/files)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose progress output",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue on per-file errors and summarize failures at the end",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include (override default set). Example: --extensions .txt .md .pdf",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be ingested without performing uploads",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    files_dir = Path(args.folder) if args.folder else (base / "files")
    cfg = load_rag_config()
    start_ts = time.time()
    failures: list[tuple[Path, str]] = []

    # Resolve file list first if dry-run or extensions override is used.
    if args.extensions is not None:
        # Normalize extensions (ensure leading dot)
        exts = {e if e.startswith('.') else f'.{e}' for e in args.extensions}
    else:
        exts = None

    # Collect candidate files
    supported_default = {
        ".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".json", ".csv",
    }
    active_exts = exts or supported_default
    candidate_files = [
        p for p in files_dir.iterdir() if p.is_file() and p.suffix.lower() in active_exts
    ]

    if args.verbose:
        print(f"Resolved {len(candidate_files)} file(s) for ingestion with extensions: {sorted(active_exts)}")

    if args.dry_run:
        print("\nDry run file list:")
        for p in candidate_files:
            print(f"  - {p.name}")
        print("\nNo uploads performed (dry-run mode).")
        return

    # Perform actual ingestion using pre-filtered list
    store, docs = ingest_files(
        cfg,
        candidate_files,
        verbose=args.verbose,
        keep_going=args.keep_going,
        failures=failures,
    )
    elapsed = time.time() - start_ts
    total = len(docs) + len(failures)
    print("\nIngestion finished.")
    print(f"Store display name: {cfg.store.display_name}")
    print(f"Store resource name: {store.name}")
    print(f"Imported files ({len(docs)}/{total}): {docs}")
    if failures:
        print(f"Failures ({len(failures)}/{total}):")
        for p, err in failures:
            print(f"  - {p.name}: {err}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":  # Ensure script runs when executed directly
    main()
