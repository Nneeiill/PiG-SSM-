"""Immutable split manifests for the revision-v2 controlled protocol.

The minimum split unit is a complete CSV sequence (never a trajectory
segment). The grouping key is the SHA-256 content hash: alloy-1 contains
byte-identical exports of the same simulation under different material
prefixes (e.g. ``2-0-Job_CBT496.csv`` is byte-identical to
``16-0-Job_CBT600.csv``), and ``simulation_case_id`` is the file stem, which
already includes the material prefix, so case-id grouping degenerates to
file-level splitting and leaks identical content across splits. All files
sharing a digest therefore move as one group.

Conformance: a file must carry exactly 101 data rows and 12 or 13 columns
(headerless numeric profile). Non-conforming files are excluded and recorded
in the manifest's ``excluded`` section with a reason (they are never padded
or truncated).

Manifests are deterministic: groups are sorted by their lexicographically
smallest member path, shuffled with the seed, and walked once; identical
(files, seed, ratios) always produce the identical manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ALLOWED_LOADING_FAMILIES = ("CBT", "CS", "PSC", "TCT", "linear", "path")
_ALLOWED_FAMILIES_LOWER = frozenset(f.lower() for f in ALLOWED_LOADING_FAMILIES)

_FILENAME_RE = re.compile(
    r"^(?P<material>\d+-\d+)-Job_(?P<family>[A-Za-z]+)(?P<suffix>\d*)\.csv$"
)


def parse_filename_metadata(filename: str) -> dict:
    """Parse ``14-0-Job_CBT569.csv`` into material / loading-family metadata.

    Returns ``{"material_id", "loading_family", "simulation_case_id"}`` where
    ``simulation_case_id`` is the file stem (``14-0-Job_CBT569``). Raises
    ``ValueError`` with a clear message for unparseable names.
    """
    match = _FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"unparseable filename: {filename!r}")
    family = match.group("family")
    if family.lower() not in _ALLOWED_FAMILIES_LOWER:
        raise ValueError(
            f"unparseable filename {filename!r}: unknown loading family {family!r} "
            f"(allowed: {', '.join(ALLOWED_LOADING_FAMILIES)})"
        )
    stem = filename[: -len(".csv")]
    return {
        "material_id": match.group("material"),
        "loading_family": family,  # canonical name as written in the filename
        "simulation_case_id": stem,
    }


def build_split_manifest(files, seed: int, ratios=(0.70, 0.10, 0.20)) -> dict:
    """Split complete CSV sequences into train/validation/test.

    ``files`` is an iterable of file paths (str or Path); relative or absolute.
    Deterministic: the input is sorted by simulation_case_id before a seeded
    shuffle, and integer counts exhaust the complete list. Entries in the
    returned split lists are the input items as given (stringified).
    """
    ratios = tuple(float(r) for r in ratios)
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    items = [str(f) for f in files]
    by_case: dict[str, str] = {}
    for item in items:
        case_id = parse_filename_metadata(Path(item).name)["simulation_case_id"]
        if case_id in by_case:
            raise ValueError(f"duplicate simulation_case_id in input: {case_id!r}")
        by_case[case_id] = item

    case_ids = sorted(by_case)
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 0:
        raise ValueError(f"ratios {ratios} infeasible for {n} sequences")

    return {
        "train": [by_case[c] for c in case_ids[:n_train]],
        "validation": [by_case[c] for c in case_ids[n_train : n_train + n_val]],
        "test": [by_case[c] for c in case_ids[n_train + n_val :]],
        "seed": seed,
        "ratios": list(ratios),
        "counts": {"train": n_train, "validation": n_val, "test": n_test},
    }


def _profile_file(path: Path) -> dict:
    """One-pass SHA-256 digest plus row/column profile of a CSV file.

    Reads the file once so hashing and conformance checking cost a single
    I/O pass. Row counting is newline-based (the dataset is headerless,
    newline-terminated numeric ASCII), matching the text-mode count in the
    leakage audit.
    """
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if not data.strip():
        return {"sha256": digest, "rows": 0, "columns": None}
    first_line = data.split(b"\n", 1)[0].rstrip(b"\r")
    rows = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
    return {"sha256": digest, "rows": rows, "columns": len(first_line.split(b","))}


def check_csv_shape_from_profile(
    profile: dict,
    expected_rows: int = 101,
    expected_columns=(12, 13),
) -> dict | None:
    """Conformance of a headerless CSV profile; ``None`` when conforming.

    Returns a violation dict in the same shape the leakage audit reports:
    ``{"error": "empty" | "row_count" | "column_count", ...}``. Row-count
    violations take precedence over column-count violations (audit parity).
    """
    if profile["rows"] == 0:
        return {"error": "empty"}
    if profile["rows"] != expected_rows:
        return {"error": "row_count", "rows": profile["rows"], "expected_rows": expected_rows}
    if profile["columns"] is not None and profile["columns"] not in tuple(expected_columns):
        return {
            "error": "column_count",
            "columns": profile["columns"],
            "expected_columns": list(expected_columns),
        }
    return None


def check_csv_shape(
    path: Path,
    expected_rows: int = 101,
    expected_columns=(12, 13),
) -> dict | None:
    """Path-based conformance check (single source of truth for the audit)."""
    return check_csv_shape_from_profile(_profile_file(path), expected_rows, expected_columns)


def build_enriched_manifest(
    project_root: Path,
    dataset_rel_dir: str,
    seed: int,
    ratios=(0.70, 0.10, 0.20),
    protocol_version: str = "revision-v2-2026-08-24",
    expected_rows: int = 101,
    expected_columns=(12, 13),
) -> dict:
    """Hash, parse, conformance-check, and split every CSV under the dataset.

    Paths in the returned manifest are relative to ``project_root``.

    Grouping: content-hash groups (see module docstring) — whole groups are
    assigned to train while below the 70% target, then to validation while
    below the 10% target, and the remainder to test. Actual counts may
    therefore exceed the target by at most ``max_group_size - 1``; the
    targets are recorded in ``expected_counts`` for the audit's drift check.
    """
    project_root = Path(project_root)
    data_dir = project_root / dataset_rel_dir
    rel_paths = sorted(
        p.relative_to(project_root).as_posix() for p in data_dir.glob("*.csv")
    )
    if not rel_paths:
        raise FileNotFoundError(f"no CSV files under {data_dir}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        profiles = list(pool.map(lambda rel: _profile_file(project_root / rel), rel_paths))

    entries = []
    excluded = []
    for rel, profile in zip(rel_paths, profiles):
        name = Path(rel).name
        meta = parse_filename_metadata(name)  # raises on unparseable names
        violation = check_csv_shape_from_profile(profile, expected_rows, expected_columns)
        if violation is not None:
            excluded.append({"path": rel, "sha256": profile["sha256"], **violation})
            continue
        entry = {"path": rel, "sha256": profile["sha256"]}
        entry.update(meta)
        entries.append(entry)

    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["sha256"], []).append(entry)

    order = sorted(groups.items(), key=lambda kv: min(e["path"] for e in kv[1]))
    rng = random.Random(seed)
    rng.shuffle(order)

    ratios = tuple(float(r) for r in ratios)
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    n = len(entries)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 0:
        raise ValueError(f"ratios {ratios} infeasible for {n} sequences")

    train, validation, test = [], [], []
    for _digest, members in order:
        if len(train) < n_train:
            train.extend(members)
        elif len(validation) < n_val:
            validation.extend(members)
        else:
            test.extend(members)

    return {
        "protocol_version": protocol_version,
        "dataset": dataset_rel_dir,
        "seed": seed,
        "ratios": list(ratios),
        "split_unit": "complete_csv_sequence",
        "group_unit": "sha256_content",
        "expected_rows": expected_rows,
        "expected_columns": list(expected_columns),
        "dataset_file_total": len(rel_paths),
        "total_sequences": n,
        "max_group_size": max(len(v) for v in groups.values()),
        "expected_counts": {"train": n_train, "validation": n_val, "test": n_test},
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "excluded": excluded,
        "train": train,
        "validation": validation,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a revision-v2 split manifest")
    parser.add_argument("--project_root", type=Path, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    args = parser.parse_args()

    if args.project_root is None:
        from pipeline.paths import resolve_project_root

        project_root = resolve_project_root()
    else:
        project_root = args.project_root

    protocol_version = "revision-v2-2026-08-24"
    if args.protocol is not None:
        protocol_version = json.loads(args.protocol.read_text())["protocol_version"]

    ratios = (0.70, 0.10, 0.20)
    if args.protocol is not None:
        split = json.loads(args.protocol.read_text())["split"]
        ratios = (split["train"], split["validation"], split["test"])

    manifest = build_enriched_manifest(
        project_root, args.dataset, args.seed, ratios, protocol_version
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(
        f"wrote {args.output}: total={manifest['total_sequences']} "
        f"train={manifest['counts']['train']} "
        f"validation={manifest['counts']['validation']} "
        f"test={manifest['counts']['test']}"
    )


if __name__ == "__main__":
    main()
