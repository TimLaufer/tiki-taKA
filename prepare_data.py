"""Unpack downloaded tracking ZIPs into the DATA.md layout; link optional videos."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent


def prepare(data_dir: Path) -> list[Path]:
    data_dir = data_dir.resolve()
    expected = {
        f"BAR-ATH_{source}_{suffix}": source
        for source in ("tactical", "broadcast")
        for suffix in ("atd.xml", "atd.txt", "homography.csv", "smart_tagging.xml")
    }
    written = []
    for archive in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = Path(member.filename).name
                if name not in expected or member.is_dir():
                    continue
                target = data_dir / expected[name] / name
                if target.exists():
                    crc = 0
                    with target.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            crc = zlib.crc32(chunk, crc)
                    if target.stat().st_size != member.file_size or crc != member.CRC:
                        raise ValueError(f"Existing file differs from archive: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".extracting")
                try:
                    with bundle.open(member) as source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                written.append(target)
                print(f"Extracted {target.relative_to(data_dir)}")
    for source in ("tactical", "broadcast"):
        target = data_dir / source / f"BAR-ATH_{source}.mp4"
        candidates = sorted(p for p in data_dir.glob(f"BAR-ATH_{source}*.mp4")
                            if p.stat().st_size > 0)
        if not target.exists() and len(candidates) == 1:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(candidates[0], target.parent))
            print(f"Linked {target.relative_to(data_dir)}")
        required = [data_dir / source / f"BAR-ATH_{source}_atd.{ext}" for ext in ("xml", "txt")]
        print(f"{source}: tracking {'ready' if all(p.is_file() for p in required) else 'missing'}; "
              f"video {'available' if target.is_file() and target.stat().st_size else 'unavailable (optional)'}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")))
    args = parser.parse_args()
    prepare(args.data_dir)
