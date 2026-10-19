"""Download MultiDoc2Dial into data/raw/. eCFR versions are fetched on demand by vpqa.data.ecfr."""
import io
import zipfile
from pathlib import Path

import httpx

URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


def main() -> None:
    out = Path("data/raw")
    if (out / "multidoc2dial" / "multidoc2dial_doc.json").exists():
        print("MultiDoc2Dial already present")
        return
    r = httpx.get(URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    print(f"extracted to {out}/multidoc2dial")


if __name__ == "__main__":
    main()
