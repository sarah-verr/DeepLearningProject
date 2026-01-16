"""
Generate vlm_levels_v3 dataset with relational position questions.

Questions are in the format: "Where is the [S_Color] [S_Shape] relative to the [O_Color] [O_Shape]?"
Answers are relational terms like "above", "below", "left", "right".
"""

import os, argparse, random, json
from core_config import ALL_SHAPES, COLORS_RGB, LEVEL_CONFIG
from core_utils import assign_colors, render_and_save, Obj, PRIMARY, rel_group
from levels import get_level_fns
from typing import List, Dict, Any


def get_combined_answer(vertical_rel: str, horizontal_rel: str) -> str:
    """Convert vertical and horizontal relations to combined answer format.
    
    Only returns combined answers when both relations exist (not aligned in either dimension).
    Returns one of: "top-left", "top-right", "bottom-left", "bottom-right"
    """
    if vertical_rel and horizontal_rel:
        # Both exist: use 4-way combined answers
        if vertical_rel == "above" and horizontal_rel == "left_of":
            return "top-left"
        elif vertical_rel == "above" and horizontal_rel == "right_of":
            return "top-right"
        elif vertical_rel == "below" and horizontal_rel == "left_of":
            return "bottom-left"
        elif vertical_rel == "below" and horizontal_rel == "right_of":
            return "bottom-right"
    
    return None  # Only generate combined questions when both relations exist


def _generate_relational_qa(
    objects: List[Obj],
    relations: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Generate relational position questions for each object pair.
    
    For each pair (A, B), generates:
    - Vertical (A → B): "What is the vertical relation of A to B?" → "above"/"below"
    - Vertical (B → A): "What is the vertical relation of B to A?" → "below"/"above"
    - Horizontal (A → B): "What is the horizontal relation of A to B?" → "left"/"right"
    - Horizontal (B → A): "What is the horizontal relation of B to A?" → "right"/"left"
    - Combined (A → B): "Where is A relative to B?" → "top-left"/"top-right"/"bottom-left"/"bottom-right"
      (Only generated when both vertical AND horizontal relations exist)
    - Combined (B → A): "Where is B relative to A?" → inverse of A→B
      (Only generated when both vertical AND horizontal relations exist)
    
    Only generates questions for PRIMARY relations (above/below/left_of/right_of).
    """
    # Create object lookup by ID
    obj_by_id = {obj.id: obj for obj in objects}
    
    # Build relation lookup: (subject_id, object_id) -> set of relation types
    rel_by_pair: Dict[Tuple[int, int], Dict[str, str]] = {}
    for rel in relations:
        if rel["type"] not in PRIMARY:
            continue
        key = (rel["subject_id"], rel["object_id"])
        if key not in rel_by_pair:
            rel_by_pair[key] = {}
        rel_by_pair[key][rel["type"]] = rel["type"]
    
    qa = []
    
    # Get all unique pairs (unordered)
    all_ids = sorted(obj_by_id.keys())
    processed_pairs = set()
    
    for i, id_a in enumerate(all_ids):
        for id_b in all_ids[i+1:]:
            # Ensure we process each pair only once
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            obj_a = obj_by_id[id_a]
            obj_b = obj_by_id[id_b]
            
            # Find relations for both directions
            # A → B relations: check what relations exist from A to B
            rels_ab = rel_by_pair.get((id_a, id_b), {})
            vert_ab = rels_ab.get("above") or rels_ab.get("below")
            horiz_ab = rels_ab.get("left_of") or rels_ab.get("right_of")
            
            # B → A relations: check what relations exist from B to A
            rels_ba = rel_by_pair.get((id_b, id_a), {})
            vert_ba = rels_ba.get("above") or rels_ba.get("below")
            horiz_ba = rels_ba.get("left_of") or rels_ba.get("right_of")
            
            # Generate questions for A → B direction
            if vert_ab:
                # Vertical question: A → B
                qa.append({
                    "question": f"What is the vertical relation of {obj_a.color} {obj_a.shape} to {obj_b.color} {obj_b.shape}?",
                    "answer": vert_ab,  # "above" or "below"
                    "question_type": "vertical",
                    "subject_id": id_a,
                    "object_id": id_b,
                    "rel_type": vert_ab,
                    "rel_group": rel_group(vert_ab),
                })
            
            if horiz_ab:
                # Horizontal question: A → B
                horiz_answer = "left" if horiz_ab == "left_of" else "right"
                qa.append({
                    "question": f"What is the horizontal relation of {obj_a.color} {obj_a.shape} to {obj_b.color} {obj_b.shape}?",
                    "answer": horiz_answer,  # "left" or "right"
                    "question_type": "horizontal",
                    "subject_id": id_a,
                    "object_id": id_b,
                    "rel_type": horiz_ab,
                    "rel_group": rel_group(horiz_ab),
                })
            
            # Combined question: only when both vertical AND horizontal relations exist
            if vert_ab and horiz_ab:
                combined_ab = get_combined_answer(vert_ab, horiz_ab)
                if combined_ab:
                    qa.append({
                        "question": f"Where is {obj_a.color} {obj_a.shape} relative to {obj_b.color} {obj_b.shape}?",
                        "answer": combined_ab,  # "top-left", "top-right", "bottom-left", "bottom-right"
                        "question_type": "combined",
                        "subject_id": id_a,
                        "object_id": id_b,
                        "rel_type": vert_ab or horiz_ab,  # Use one for tracking
                        "rel_group": "PRIMARY",
                    })
            
            # Generate questions for B → A direction
            if vert_ba:
                # Vertical question: B → A
                qa.append({
                    "question": f"What is the vertical relation of {obj_b.color} {obj_b.shape} to {obj_a.color} {obj_a.shape}?",
                    "answer": vert_ba,  # "above" or "below"
                    "question_type": "vertical",
                    "subject_id": id_b,
                    "object_id": id_a,
                    "rel_type": vert_ba,
                    "rel_group": rel_group(vert_ba),
                })
            
            if horiz_ba:
                # Horizontal question: B → A
                horiz_answer_ba = "left" if horiz_ba == "left_of" else "right"
                qa.append({
                    "question": f"What is the horizontal relation of {obj_b.color} {obj_b.shape} to {obj_a.color} {obj_a.shape}?",
                    "answer": horiz_answer_ba,  # "left" or "right"
                    "question_type": "horizontal",
                    "subject_id": id_b,
                    "object_id": id_a,
                    "rel_type": horiz_ba,
                    "rel_group": rel_group(horiz_ba),
                })
            
            # Combined question: only when both vertical AND horizontal relations exist
            if vert_ba and horiz_ba:
                combined_ba = get_combined_answer(vert_ba, horiz_ba)
                if combined_ba:
                    qa.append({
                        "question": f"Where is {obj_b.color} {obj_b.shape} relative to {obj_a.color} {obj_a.shape}?",
                        "answer": combined_ba,  # "top-left", "top-right", "bottom-left", "bottom-right"
                        "question_type": "combined",
                        "subject_id": id_b,
                        "object_id": id_a,
                        "rel_type": vert_ba or horiz_ba,  # Use one for tracking
                        "rel_group": "PRIMARY",
                    })
    
    # Assign final IDs
    for i, item in enumerate(qa):
        item["id"] = i
    
    return qa


def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="vlm_levels_v3")
    ap.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--scenes_per_level", type=int, default=20)
    
    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--colors", type=str, default="")
    
    # For compatibility with generate_data.py interface
    ap.add_argument("--primary", action="store_true", help="Include PRIMARY questions (always used for v3)")
    ap.add_argument("--advanced", action="store_true", help="Not used for v3")

    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.base_dir, exist_ok=True)

    question_groups = ["primary"]  # Always use primary for v3

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
                json_path = os.path.join(ann_dir, fname.replace(".png",".json"))
                
                # Use exact same render_and_save as generate_data.py
                render_and_save(
                    objs=objs,
                    bg_key=bg_key,
                    img_size=args.img_size,
                    out_img=os.path.join(img_dir, fname),
                    out_json=json_path,
                    rel_fn=rel_fn,
                    patch=args.patch,
                    allow_advanced_language=cfg["adv_lang"],
                    dedup_qa=True,
                    question_groups=question_groups,
                )
                
                # Now replace questions with relational position questions
                with open(json_path, "r") as f:
                    ann = json.load(f)
                
                # Convert objects back to Obj instances for question generation
                obj_list = []
                for obj_dict in ann["objects"]:
                    obj = Obj(
                        id=obj_dict["id"],
                        shape=obj_dict["shape"],
                        color=obj_dict["color"],
                        center=tuple(obj_dict["center"]),
                        size=obj_dict["size"],
                        bbox=tuple(obj_dict["bbox"]),
                    )
                    obj_list.append(obj)
                
                # Generate relational position questions
                new_qa = _generate_relational_qa(obj_list, ann["relations"])
                
                # Replace qa field
                ann["qa"] = new_qa
                
                # Save back
                with open(json_path, "w") as f:
                    json.dump(ann, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()