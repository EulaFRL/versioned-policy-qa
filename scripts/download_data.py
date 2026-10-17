"""Download MultiDoc2Dial into data/raw/. eCFR versions are fetched on demand by vpqa.data.ecfr."""
from pathlib import Path

URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


def main() -> None:
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Download {URL} and unzip into {out}/ (manual step for now).")


if __name__ == "__main__":
    main()
