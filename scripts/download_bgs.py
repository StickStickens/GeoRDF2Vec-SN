#!/usr/bin/env python3
"""Download the BGS+ dataset (Wilcke et al., 2020) into data/bgs/.

The files live in Git LFS in the mmkg repository, so they are fetched through
GitLab's raw URLs, which serve the LFS content rather than the pointer files.
The main file's checksum equals the LFS object ID published in that repository.

    uv run scripts/download_bgs.py          # skips files that are already present
    uv run scripts/download_bgs.py --force  # downloads again
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://gitlab.com/wxwilcke/mmkg/-/raw/master/bgs/{name}?inline=false"
FILES = {
    "bgs+.nt.gz": "3cf13b2a85c676e92f898612dd00ac00fc33871b4a4a373adf20dbd58ae2406d",
    "bgs+_train_set.nt.gz": "82f533fb4bdb2185b86fd728b1a63e97c81c17c93790259952c80bfd7bdf9e9f",
    "bgs+_test_valid.nt.gz": "f3ceb5028af0ee67491fcc39db7ed97030d460b838d5a5a3602ac17d1b10ffd4",
    "bgs+_test_set.nt.gz": "5c98cbdb3b5e096ca31101243fda096d259d150e89239dd980752ffc3b9455dc",
}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "bgs"
CHUNK_SIZE = 1024 * 1024


def download(url: str, target: Path) -> None:
    """Download to a .part file, so an interrupted run leaves no partial file."""
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "geordf2vec-sn-downloader"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        while chunk := response.read(CHUNK_SIZE):
            out.write(chunk)
    partial.replace(target)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="download again")
    args = parser.parse_args()
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    failed = False
    for name, expected in FILES.items():
        target = dest / name
        if args.force or not target.exists():
            print(f"Downloading {name}", file=sys.stderr)
            download(BASE_URL.format(name=name.replace("+", "%2B")), target)
        actual = sha256_of(target)
        if actual != expected:
            print(f"Checksum mismatch for {name}: got {actual}", file=sys.stderr)
            failed = True
    if failed:
        return 1
    print(f"BGS+ ready in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
