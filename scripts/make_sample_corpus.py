"""Generate the bundled sample corpus.

Writes a small, deterministic parts + suppliers catalog under samples/ so the
whole benchmark runs on a fresh clone with no network and no acme-parts-cloud
container. Same seed always produces the same files.

All data is synthetic. Meridian Fabrication Co. is fictional.

    python scripts/make_sample_corpus.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

MATERIALS = ["Stainless steel", "Zinc-plated", "Brass", "Carbon steel", "Nylon", "Aluminum"]
FASTENERS = ["hex bolt", "socket cap screw", "flat washer", "lock nut", "cotter pin"]
CATEGORIES = {
    "Fasteners": ["hex bolt", "socket cap screw", "flat washer", "lock nut", "cotter pin"],
    "Springs": ["compression spring", "torsion spring", "extension spring"],
    "Seals": ["o-ring", "gasket", "oil seal"],
    "Brackets": ["mounting bracket", "angle bracket", "shelf bracket"],
    "Fittings": ["elbow fitting", "tee fitting", "hose barb"],
    "Bearings": ["ball bearing", "bearing race", "thrust washer"],
}
UOMS = ["EA", "BOX", "PK", "FT", "LB"]
STATUSES = ["active", "active", "active", "obsolete", "superseded", "prototype"]


def _part_number(rng: random.Random, n: int) -> str:
    """Pick one of three naming eras for variety (see QUIRKS section 3)."""
    era = rng.random()
    if era < 0.6:
        return f"PN-{n:04d}"  # current
    if era < 0.85:
        return f"2019-PN-{n}"  # migration era, no zero-pad
    return f"P{n}"  # legacy


def _name(rng: random.Random, base: str) -> str:
    """Mix short labels and long engineering descriptions on purpose."""
    if rng.random() < 0.35:
        return base.title()  # short label like "Hex Bolt"
    material = rng.choice(MATERIALS)
    if base in FASTENERS:
        size = rng.choice(["M6", "M8", "1/4-20", "3/8-16", "1/2-13"])
        length = rng.choice(["12mm", "20mm", "1in", "2in", "3in"])
        return f"{material} {base}, {size} thread, {length} length"
    dims = rng.choice(["25mm bore", "1in ID", "50mm OD", "2in stroke"])
    return f"{material} {base}, {dims}"


def generate_parts(rng: random.Random, count: int) -> list[dict]:
    rows = []
    used = set()
    n = 1
    for _ in range(count):
        while True:
            pn = _part_number(rng, n)
            n += 1
            if pn not in used:
                used.add(pn)
                break
        category = rng.choice(list(CATEGORIES))
        base = rng.choice(CATEGORIES[category])
        rows.append(
            {
                "part_number": pn,
                "name": _name(rng, base),
                "category": category,
                "uom": rng.choice(UOMS),
                "status": rng.choice(STATUSES),
            }
        )
    return rows


# a handful of suppliers with deliberate near-duplicate variants (QUIRKS section 5)
SUPPLIER_BASES = [
    ("Vortex Metals", "US"),
    ("Cascade Fasteners", "US"),
    ("Nord Precision", "DE"),
    ("Kanto Seiki", "JP"),
    ("Ironline Supply", "CA"),
    ("Meridian Stock", "US"),
]
NEAR_DUPES = {
    "Vortex Metals": ["VORTEX METALS", "Vortex Metals Inc."],
    "Cascade Fasteners": ["Cascade Fasteners LLC"],
}


def generate_suppliers(rng: random.Random) -> list[dict]:
    rows = []
    code = 1
    for name, country in SUPPLIER_BASES:
        variants = [name] + NEAR_DUPES.get(name, [])
        for v in variants:
            handle = v.lower().replace(" ", "").replace(".", "").replace(",", "")[:10]
            rows.append(
                {
                    "supplier_code": f"SUP-{code:03d}",
                    "name": v,
                    "country": country,
                    "contact_email": f"sales@{handle}.example",
                    "active": rng.choice([1, 1, 1, 0]),
                }
            )
            code += 1
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):>3} rows -> {path.relative_to(path.parent.parent)}")


def main(seed: int = 42, n_parts: int = 90) -> None:
    rng = random.Random(seed)
    SAMPLES.mkdir(exist_ok=True)
    parts = generate_parts(rng, n_parts)
    suppliers = generate_suppliers(rng)
    write_csv(
        SAMPLES / "parts_sample.csv", parts, ["part_number", "name", "category", "uom", "status"]
    )
    write_csv(
        SAMPLES / "suppliers_sample.csv",
        suppliers,
        ["supplier_code", "name", "country", "contact_email", "active"],
    )


if __name__ == "__main__":
    main()
