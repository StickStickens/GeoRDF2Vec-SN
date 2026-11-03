#!/usr/bin/env python3
"""Download and unpack the KGBench dmg777k dataset into data/kgbench_dmg777k/.

Uses the same URL as the official kgbench-loader (kgbench/urls.py), without
installing the loader, which depends on torch and torchvision.

    uv run scripts/download_dmg777k.py          # skips work that is already done
    uv run scripts/download_dmg777k.py --force  # downloads and unpacks again
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.dropbox.com/s/fal1nobf1etxtfu/dmg777k.tgz?dl=1"
SHA256 = "825517b81d6fd420706fbc2d19d8f90c6a84daa60381bdcc30fed9150b9e3c58"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "kgbench_dmg777k"
EXPECTED_FILES = {
    "nodes.int.csv",
    "relations.int.csv",
    "training.int.csv",
    "validation.int.csv",
    "testing.int.csv",
}
TRIPLES_FILES = {"triples.int.csv", "triples.int.csv.gz"}
CHUNK_SIZE = 1024 * 1024


def download(url: str, target: Path) -> None:
    """Download to a .part file, so an interrupted run leaves no partial archive."""
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "geordf2vec-sn-downloader"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(CHUNK_SIZE):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if total and done != total:
        partial.unlink()
        raise RuntimeError(f"incomplete download: got {done} of {total} bytes")
    partial.replace(target)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, dest: Path) -> None:
    """Unpack only the data files, flat into dest.

    Skips the archive's own scripts/ folder and macOS "._*" metadata entries.
    Using base names keeps every write inside dest.
    """
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            is_data = name.endswith((".csv", ".csv.gz")) and not name.startswith(".")
            if not member.isfile() or not is_data:
                continue
            source = tar.extractfile(member)
            with source, (dest / name).open("wb") as out:
                shutil.copyfileobj(source, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="download and unpack again")
    args = parser.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "dmg777k.tgz"

    if args.force or not archive.exists():
        print(f"Downloading {URL}", file=sys.stderr)
        download(URL, archive)
    else:
        print(f"Archive already present: {archive}", file=sys.stderr)

    actual = sha256_of(archive)
    print(f"sha256 {actual}")
    if SHA256 is not None and actual != SHA256:
        print(f"Checksum mismatch: expected {SHA256}", file=sys.stderr)
        return 1

    extract(archive, dest)

    present = {path.name for path in dest.iterdir()}
    missing = EXPECTED_FILES - present
    if not TRIPLES_FILES & present:
        missing.add("triples.int.csv[.gz]")
    if missing:
        print(f"Missing after unpacking: {sorted(missing)}", file=sys.stderr)
        return 1

    print(f"dmg777k ready in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
