"""Download and verify the OdorNet Zenodo data archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
from urllib.parse import unquote, urlparse
import zipfile
from pathlib import Path


ZENODO_ARCHIVE_URL = "https://zenodo.org/records/19838456/files/OdorNet_v1.0.0.zip?download=1"
ZENODO_ARCHIVE_SHA256 = "56b71debe259001bfbc0f54ddeabe88dae347f319a4f3758318423c4c993e4a4"
DEFAULT_OUTPUT_DIR = Path("data_releases")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def ensure_within_directory(base_dir: Path, target: Path) -> None:
    base_resolved = base_dir.resolve()
    target_resolved = target.resolve()
    if os.path.commonpath([base_resolved, target_resolved]) != str(base_resolved):
        raise ValueError(f"Unsafe archive member path: {target}")


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                ensure_within_directory(output_dir, output_dir / member)
            archive.extractall(output_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                ensure_within_directory(output_dir, output_dir / member.name)
            archive.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    checksum_files = sorted(output_dir.rglob("checksums_sha256.txt"))
    if not checksum_files:
        raise FileNotFoundError("No checksums_sha256.txt found after extraction.")
    return checksum_files[0].parent


def verify_manifest(package_dir: Path) -> None:
    manifest_path = package_dir / "checksums_sha256.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest_path}")

    checked = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative_path = line.split(maxsplit=1)
        relative_path = relative_path.strip().lstrip("*")
        file_path = package_dir / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest entry is missing: {relative_path}")
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        checked += 1

    print(f"Verified {checked} files from {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=ZENODO_ARCHIVE_URL, help="Zenodo archive URL.")
    parser.add_argument(
        "--archive-sha256",
        default=ZENODO_ARCHIVE_SHA256,
        help="Optional SHA-256 checksum for the downloaded archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the downloaded archive and extracted files.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Optional local archive path. Defaults to output-dir / URL filename.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive and verify checksums_sha256.txt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if archive-path already exists.",
    )
    args = parser.parse_args()

    if not args.url:
        raise SystemExit(
            "Zenodo archive URL is blank. Fill ZENODO_ARCHIVE_URL or pass --url."
        )

    archive_name = Path(unquote(urlparse(args.url).path)).name or "OdorNet_v1.0.0.zip"
    archive_path = args.archive_path or (args.output_dir / archive_name)
    if archive_path.exists() and not args.force:
        print(f"Using existing archive: {archive_path}")
    else:
        download_file(args.url, archive_path)
        print(f"Downloaded: {archive_path}")

    if args.archive_sha256:
        actual = sha256_file(archive_path)
        if actual != args.archive_sha256:
            raise ValueError(
                f"Archive checksum mismatch: expected {args.archive_sha256}, got {actual}"
            )
        print("Archive checksum verified.")

    if args.extract:
        package_dir = extract_archive(archive_path, args.output_dir)
        verify_manifest(package_dir)


if __name__ == "__main__":
    main()
