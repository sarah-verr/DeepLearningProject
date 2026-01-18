#!/usr/bin/env python3
import argparse
import json
import os


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    levels = [d for d in os.listdir(base_dir) if d.startswith("level_")]
    return sorted(levels)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Filter occluded annotations to only keep QA where removed object is subject or object."
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        default="data/vlm_levels_occluded",
        help="Occluded dataset root (expects level_*/ann/*.json).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    levels = _find_level_dirs(args.base_dir)
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    kept_files = 0
    removed_files = 0
    removed_images = 0
    removed_qas = 0
    for lvl in levels:
        ann_dir = os.path.join(args.base_dir, lvl, "ann")
        if not os.path.isdir(ann_dir):
            continue
        ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        for ann_file in ann_files:
            ann_path = os.path.join(ann_dir, ann_file)
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
            except Exception:
                continue

            removed_id = ann.get("removed_object_id")
            if removed_id is None:
                continue
            qa_list = ann.get("qa", []) or []
            filtered = []
            for qa in qa_list:
                subj_id = qa.get("subject_id")
                obj_id = qa.get("object_id")
                if subj_id == removed_id or obj_id == removed_id:
                    filtered.append(qa)
                else:
                    removed_qas += 1
            if not filtered:
                os.remove(ann_path)
                removed_files += 1
                image_base = os.path.splitext(ann_file)[0]
                img_dir = os.path.join(args.base_dir, lvl, "images")
                for ext in (".png", ".jpg", ".jpeg"):
                    img_path = os.path.join(img_dir, image_base + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        removed_images += 1
                        break
                continue
            ann["qa"] = filtered
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, indent=2)
            kept_files += 1

    print(
        f"Filtered QAs in {kept_files} files; removed {removed_files} annotations and {removed_images} images."
    )
    print(f"Removed {removed_qas} QA entries not involving the removed object.")


if __name__ == "__main__":
    main()
