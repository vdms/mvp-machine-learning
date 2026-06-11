from __future__ import annotations

import csv
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_raw_files() -> list[Path]:
    patterns = ["*.zip", "*.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(RAW_DIR.glob(pattern)))
    return files


def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except csv.Error:
        return ";"


def inspect_csv(path: Path) -> None:
    print(f"Arquivo CSV encontrado: {path.name}")
    print(f"Tamanho: {path.stat().st_size / (1024 ** 2):.2f} MB")
    with path.open("r", encoding="latin1", errors="replace") as fh:
        sample = fh.read(100_000)
    delimiter = sniff_delimiter(sample)
    header = sample.splitlines()[0].split(delimiter)
    print(f"Delimitador provável: {delimiter!r}")
    print(f"Total de colunas: {len(header)}")
    print("Primeiras colunas:")
    for col in header[:40]:
        print(f"- {col}")


def inspect_zip(path: Path) -> None:
    print(f"Arquivo ZIP encontrado: {path.name}")
    print(f"Tamanho: {path.stat().st_size / (1024 ** 2):.2f} MB")
    with zipfile.ZipFile(path) as zf:
        members = zf.infolist()
        csv_members = [m for m in members if m.filename.lower().endswith(".csv")]
        print(f"Arquivos CSV internos: {len(csv_members)}")
        for member in csv_members[:10]:
            print(f"- {member.filename} ({member.file_size / (1024 ** 2):.2f} MB)")

        target = None
        for priority_term in ["microdados", "resultados", "participantes"]:
            matches = [
                m for m in csv_members
                if priority_term in Path(m.filename).name.lower()
            ]
            if matches:
                target = matches[0]
                break
        if target is None:
            target = csv_members[0] if csv_members else None
        if target is None:
            print("Nenhum CSV interno encontrado.")
            return

        with zf.open(target) as fh:
            sample_bytes = fh.read(100_000)
        sample = sample_bytes.decode("latin1", errors="replace")
        delimiter = sniff_delimiter(sample)
        header = sample.splitlines()[0].split(delimiter)
        print(f"\nCSV recomendado para leitura: {target.filename}")
        print(f"Delimitador provável: {delimiter!r}")
        print(f"Total de colunas: {len(header)}")
        print("Primeiras colunas:")
        for col in header[:50]:
            print(f"- {col}")


def main() -> None:
    files = find_raw_files()
    if not files:
        print("Nenhum arquivo .zip ou .csv encontrado em data/raw/.")
        print("Baixe os Microdados do ENEM no portal oficial do INEP e coloque o arquivo em data/raw/.")
        raise SystemExit(0)

    for path in files:
        print("=" * 80)
        if path.suffix.lower() == ".zip":
            inspect_zip(path)
        elif path.suffix.lower() == ".csv":
            inspect_csv(path)


if __name__ == "__main__":
    main()
