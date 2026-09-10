#!/usr/bin/env python3
"""Download only the official YCB Rubik's cube Google 16k model."""

import argparse
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


OBJECT = "077_rubiks_cube"
URL = f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{OBJECT}_google_16k.tgz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "assets" / "ycb",
        help="Destination directory (default: assets/ycb next to this script)",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / OBJECT
    if destination.exists():
        parser.exit(1, f"Already exists; not overwriting: {destination}\n")

    try:
        with tempfile.TemporaryDirectory(prefix=".rubiks-", dir=output) as tmp:
            temp = Path(tmp)
            archive = temp / "model.tgz"
            print(f"Downloading: {URL}", flush=True)
            request = urllib.request.Request(URL, headers={"User-Agent": "YCB-model-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                with archive.open("wb") as target:
                    shutil.copyfileobj(response, target)

            extracted = temp / "extracted"
            extracted.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                members = bundle.getmembers()
                for member in members:
                    path = (extracted / member.name).resolve()
                    if not path.is_relative_to(extracted.resolve()):
                        raise ValueError(f"Unsafe archive path: {member.name}")
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unsupported archive entry: {member.name}")
                # Only validated regular files and directories are extracted.
                for member in members:
                    path = extracted / member.name
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as source, path.open("wb") as target:
                            shutil.copyfileobj(source, target)

            model = extracted / OBJECT / "google_16k"
            for filename in ("textured.obj", "textured.mtl", "texture_map.png"):
                if not (model / filename).is_file():
                    raise ValueError(f"Missing expected model file: {filename}")
            shutil.move(str(extracted / OBJECT), str(destination))
        print(f"Ready. FoundationPose mesh: {destination / 'google_16k' / 'textured.obj'}")
    except (OSError, urllib.error.URLError, tarfile.TarError, ValueError) as error:
        parser.exit(1, f"Download/extraction failed: {error}\n")


if __name__ == "__main__":
    main()
