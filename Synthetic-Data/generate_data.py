import os, argparse, random
from core_config import ALL_SHAPES, COLORS_RGB, LEVEL_CONFIG
from core_utils import assign_colors, render_and_save
from levels import get_level_fns

def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="vlm_levels")
    ap.add_argument("--levels", type=int, nargs="+", default=[1,2,3,4,5,6])
    ap.add_argument("--scenes_per_level", type=int, default=20)
    
    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--colors", type=str, default="")
    
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.base_dir, exist_ok=True)

    # Process Shapes
    shapes = parse_list(args.shapes)
    shapes = [s for s in shapes if s in ALL_SHAPES]
    if not shapes: shapes = ALL_SHAPES[:]

    # Process Colors
    user_colors = parse_list(args.colors)
    valid_colors = [c for c in user_colors if c in COLORS_RGB]
    if not valid_colors: valid_colors = list(COLORS_RGB.keys())

    for level in args.levels:
        # 1. Get Config
        cfg = LEVEL_CONFIG.get(level)
        if not cfg:
            print(f"Warning: Config for level {level} not found. Skipping.")
            continue

        # 2. Get Functions
        scene_fn, rel_fn = get_level_fns(level)
        
        out_dir = os.path.join(args.base_dir, f"level_{level}")
        img_dir = os.path.join(out_dir, "images")
        ann_dir = os.path.join(out_dir, "ann")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        print(f"Generating level {level} ({args.scenes_per_level} scenes)...")
        random.seed(args.seed + level)

        for sid in range(args.scenes_per_level):
            # 3. Generate
            n_range = (cfg["min_shapes"], cfg["max_shapes"])
            
            objs = scene_fn(args.img_size, args.patch, n_range, shapes)
            
            # This now enforces STRICT uniqueness
            assign_colors(objs, valid_colors)

            for bg_key in ("w", "b"):
                fname = f"{sid:05d}_{bg_key}.png"
                render_and_save(
                    objs=objs,
                    bg_key=bg_key,
                    img_size=args.img_size,
                    out_img=os.path.join(img_dir, fname),
                    out_json=os.path.join(ann_dir, fname.replace(".png",".json")),
                    rel_fn=rel_fn,
                    patch=args.patch,
                    allow_advanced_language=cfg["adv_lang"],
                )

    print("Done.")

if __name__ == "__main__":
    main()