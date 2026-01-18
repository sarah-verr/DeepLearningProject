#!/usr/bin/env python3
import argparse
import json
import os

from PIL import Image, ImageDraw


def _find_level_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    levels = [d for d in os.listdir(base_dir) if d.startswith("level_")]
    return sorted(levels)


def _background_color(ann: dict, image_id: str) -> tuple[int, int, int]:
    bg = (ann.get("background") or "").strip().lower()
    if not bg and image_id.endswith("_b"):
        bg = "b"
    if not bg and image_id.endswith("_w"):
        bg = "w"
    if bg == "w":
        return (255, 255, 255)
    return (0, 0, 0)


def _occlude_bbox(image: Image.Image, bbox: list[int], color: tuple[int, int, int]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], fill=color)
    return img


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Occlude single objects in images using bbox fill.")
    ap.add_argument(
        "--base_dir",
        type=str,
        default="data/vlm_levels",
        help="Input dataset root (expects level_*/ann and level_*/images).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/vlm_levels_occluded",
        help="Output dataset root.",
    )
    ap.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Optional list of levels to process (e.g., level_0 level_1).",
    )
    ap.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Limit number of objects to occlude per image.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = args.levels or _find_level_dirs(args.base_dir)
    if not levels:
        raise SystemExit(f"No level_* directories found in: {args.base_dir}")

    for lvl in levels:
        ann_dir = os.path.join(args.base_dir, lvl, "ann")
        img_dir = os.path.join(args.base_dir, lvl, "images")
        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            continue

        out_level_dir = os.path.join(args.out_dir, lvl)
        out_img_dir = os.path.join(out_level_dir, "images")
        out_ann_dir = os.path.join(out_level_dir, "ann")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_ann_dir, exist_ok=True)

        ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")])
        for ann_file in ann_files:
            ann_path = os.path.join(ann_dir, ann_file)
            image_id = os.path.splitext(ann_file)[0]

            img_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = os.path.join(img_dir, image_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            objects = ann.get("objects", []) or []
            if not objects:
                continue
            if args.max_objects is not None:
                objects = objects[: args.max_objects]

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                bg_color = _background_color(ann, image_id)

                for obj in objects:
                    bbox = obj.get("bbox")
                    obj_id = obj.get("id")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    if obj_id is None:
                        continue

                    occluded = _occlude_bbox(img, bbox, bg_color)
                    out_id = f"{image_id}_rm{obj_id}"
                    out_img_path = os.path.join(out_img_dir, f"{out_id}.png")
                    occluded.save(out_img_path, "PNG")

                    ann_out = dict(ann)
                    ann_out["removed_object_id"] = obj_id
                    ann_out["removed_object"] = {
                        "id": obj_id,
                        "shape": obj.get("shape"),
                        "color": obj.get("color"),
                        "bbox": bbox,
                    }
                    with open(os.path.join(out_ann_dir, f"{out_id}.json"), "w", encoding="utf-8") as f:
                        json.dump(ann_out, f, indent=2)

        print(f"[{lvl}] Wrote occluded images to: {out_level_dir}")


if __name__ == "__main__":
    main()
