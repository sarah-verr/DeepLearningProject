import argparse
import copy
import json
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pick_first_existing(candidates: list[str]) -> str:
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return candidates[0] if candidates else ""


def _find_b_ann_paths(base_dir: str, level: str) -> list[str]:
    ann_dir = os.path.join(base_dir, level, "ann")
    if not os.path.isdir(ann_dir):
        return []
    paths = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.endswith("_b.json")]
    return sorted(paths)


def _center_from_obj(obj: dict) -> tuple[int, int] | None:
    center = obj.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return int(center[0]), int(center[1])
    bbox = obj.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        return int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
    return None


def _build_objective_caption(objects: list[dict]) -> str:
    parts = []
    for obj in sorted(objects, key=lambda o: int(o.get("id", 0))):
        color = obj.get("color", "unknown")
        shape = obj.get("shape", "object")
        center = _center_from_obj(obj)
        if center is None:
            parts.append(f"There is a {color} {shape}.")
        else:
            x, y = center
            parts.append(f"There is a {color} {shape} at coordinates ({x}, {y}).")
    return " ".join(parts)


def _make_captions_meta(caption: str) -> list[dict]:
    return [
        {
            "id": 0,
            "caption": caption,
            "subject_id": None,
            "object_id": None,
            "rel_type": None,
            "rel_group": None,
            "rel_phrase": None,
            "entailed_qa_ids": [],
            "contradicted_qa_ids": [],
        }
    ]


def _rewrite_ann(ann: dict) -> dict:
    new_ann = copy.deepcopy(ann)
    caption = _build_objective_caption(new_ann.get("objects", []))

    new_ann["orig_captions"] = ann.get("captions")
    new_ann["orig_captions_meta"] = ann.get("captions_meta")

    new_ann["captions"] = [caption]
    new_ann["captions_meta"] = _make_captions_meta(caption)

    qa = new_ann.get("qa", []) or []
    for item in qa:
        item["caption_id"] = 0
    new_ann["qa"] = qa

    meta = new_ann.get("meta") or {}
    meta["objective_caption"] = caption
    new_ann["meta"] = meta

    return new_ann


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build objective-only captions for synthetic annotations.")
    ap.add_argument(
        "--base_data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "Synthetic-Data", "vlm_levels"),
    )
    ap.add_argument(
        "--out_base_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "vlm_levels_objective"),
    )
    ap.add_argument("--levels", nargs="*", default=None, help="Levels to process (e.g., level_1 level_2).")
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of *_b.json files per level.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.base_data_path):
        candidates = [
            args.base_data_path,
            os.path.join(_REPO_ROOT, "Synthetic-Data", "vlm_levels"),
            os.path.join(_REPO_ROOT, "data", "vlm_levels"),
        ]
        picked = _pick_first_existing(candidates)
        if picked and picked != args.base_data_path:
            args.base_data_path = picked
    levels = args.levels
    if not levels:
        levels = [d for d in os.listdir(args.base_data_path) if d.startswith("level_")]
        levels = sorted(levels)

    if not levels:
        if not levels:
            raise SystemExit(f"No level_* directories found in: {args.base_data_path}")

    total = 0
    for lvl in levels:
        ann_paths = _find_b_ann_paths(args.base_data_path, lvl)
        if args.max_files is not None:
            ann_paths = ann_paths[: args.max_files]
        if not ann_paths:
            continue

        out_ann_dir = os.path.join(args.out_base_path, lvl, "ann")
        os.makedirs(out_ann_dir, exist_ok=True)

        for ann_path in ann_paths:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            new_ann = _rewrite_ann(ann)

            out_path = os.path.join(out_ann_dir, os.path.basename(ann_path))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(new_ann, f, indent=2)
            total += 1

    print(f"Wrote {total} objective annotation files to: {args.out_base_path}")


if __name__ == "__main__":
    main()
