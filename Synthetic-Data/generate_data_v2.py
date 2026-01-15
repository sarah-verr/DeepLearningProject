"""
Generate vlm_levels_v2 dataset with attribute questions instead of yes/no questions.

Uses the same generation logic as generate_data.py to ensure identical images/objects/relations,
then replaces the questions with attribute questions.
"""

import os, argparse, random, json
from core_config import ALL_SHAPES, COLORS_RGB, LEVEL_CONFIG
from core_utils import assign_colors, render_and_save, Obj, rel_phrase, PRIMARY, INV
from levels import get_level_fns
from typing import List, Dict, Any, Tuple


def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _find_second_objects(objects: List[Obj]) -> Tuple[Obj | None, Obj | None, Obj | None, Obj | None]:
    """
    Find second from left, second from right, second from top, second from bottom.
    For 3 objects, this returns the middle object (which is the second from both directions).
    Returns (second_left, second_right, second_top, second_bottom) or None if not available.
    """
    if len(objects) < 3:
        return None, None, None, None
    
    sorted_by_x = sorted(objects, key=lambda o: o.center[0])
    sorted_by_y = sorted(objects, key=lambda o: o.center[1])
    
    second_left = sorted_by_x[1] if len(sorted_by_x) >= 2 else None
    second_right = sorted_by_x[-2] if len(sorted_by_x) >= 2 else None
    second_top = sorted_by_y[1] if len(sorted_by_y) >= 2 else None
    second_bottom = sorted_by_y[-2] if len(sorted_by_y) >= 2 else None
    
    return second_left, second_right, second_top, second_bottom


def _generate_attribute_qa(
    objects: List[Obj],
    relations: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Generate attribute questions (color/shape) instead of yes/no questions.
    
    Strategy:
    - 2 objects: Direct questions for all relations
    - 3 objects: Use middle object as reference (horizontal middle for left/right, vertical middle for above/below)
    - 4+ objects: Use unambiguous positions (second from left/right/top/bottom)
    """
    by_id = {o.id: o for o in objects}
    qa = []
    
    num_objects = len(objects)
    
    # Build relation maps - store ALL relation types per pair (objects can have multiple relations)
    rel_by_subject_obj: Dict[Tuple[int, int], set] = {}
    for r in relations:
        key = (r["subject_id"], r["object_id"])
        if key not in rel_by_subject_obj:
            rel_by_subject_obj[key] = set()
        rel_by_subject_obj[key].add(r["type"])
    
    if num_objects == 2:
        # Simple case: ask about the other object
        obj0, obj1 = objects[0], objects[1]
        
        # Check all primary relations between them
        for rel_type in PRIMARY:
            key = (obj0.id, obj1.id)
            if key in rel_by_subject_obj and rel_type in rel_by_subject_obj[key]:
                phrase = rel_phrase(rel_type)
                # Color question
                qa.append({
                    "question": f"What is the color of the object {phrase} the {obj1.color} {obj1.shape}?",
                    "answer": obj0.color,
                    "question_type": "color",
                    "subject_id": obj0.id,
                    "object_id": obj1.id,
                    "rel_type": rel_type,
                    "rel_group": "PRIMARY",
                    "rel_phrase": phrase,
                })
                # Shape question
                qa.append({
                    "question": f"What is the shape of the object {phrase} the {obj1.color} {obj1.shape}?",
                    "answer": obj0.shape,
                    "question_type": "shape",
                    "subject_id": obj0.id,
                    "object_id": obj1.id,
                    "rel_type": rel_type,
                    "rel_group": "PRIMARY",
                    "rel_phrase": phrase,
                })
            
            # Inverse relation
            inv_type = INV[rel_type]
            key_inv = (obj1.id, obj0.id)
            if key_inv in rel_by_subject_obj and inv_type in rel_by_subject_obj[key_inv]:
                phrase_inv = rel_phrase(inv_type)
                # Color question
                qa.append({
                    "question": f"What is the color of the object {phrase_inv} the {obj0.color} {obj0.shape}?",
                    "answer": obj1.color,
                    "question_type": "color",
                    "subject_id": obj1.id,
                    "object_id": obj0.id,
                    "rel_type": inv_type,
                    "rel_group": "PRIMARY",
                    "rel_phrase": phrase_inv,
                })
                # Shape question
                qa.append({
                    "question": f"What is the shape of the object {phrase_inv} the {obj0.color} {obj0.shape}?",
                    "answer": obj1.shape,
                    "question_type": "shape",
                    "subject_id": obj1.id,
                    "object_id": obj0.id,
                    "rel_type": inv_type,
                    "rel_group": "PRIMARY",
                    "rel_phrase": phrase_inv,
                })
    
    else:  # 3+ objects - use second objects for unambiguous questions
        # For 3 objects: second objects are the middle objects
        # For 4+ objects: second from left/right/top/bottom
        second_left, second_right, second_top, second_bottom = _find_second_objects(objects)
        
        # Process each reference object with its natural direction
        # second_left: check "left_of" (what's to the left of second from left)
        # second_right: check "right_of" (what's to the right of second from right)
        # second_top: check "above" (what's above second from top)
        # second_bottom: check "below" (what's below second from bottom)
        ref_objects = [
            (second_left, "left_of"),
            (second_right, "right_of"),
            (second_top, "above"),
            (second_bottom, "below"),
        ]
        
        # Deduplicate by (ref_obj.id, rel_type) to handle cases where same object appears
        # with different relation types (e.g., for 3 objects, second_left == second_right)
        seen_refs = set()
        for ref_obj, rel_type in ref_objects:
            if ref_obj is None:
                continue
            
            # Create a unique key for this reference object and relation type
            ref_key = (ref_obj.id, rel_type)
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            
            # Check the relation type for this reference object
            # Find objects that have this specific relation to the reference object
            candidates = []
            for obj in objects:
                if obj.id == ref_obj.id:
                    continue
                
                key = (obj.id, ref_obj.id)
                if key in rel_by_subject_obj and rel_type in rel_by_subject_obj[key]:
                    candidates.append(obj)
            
            # If there's only one candidate, it's unambiguous - generate both color and shape questions
            if len(candidates) == 1:
                obj = candidates[0]
                phrase = rel_phrase(rel_type)
                qa.append({
                    "question": f"What is the color of the object {phrase} the {ref_obj.color} {ref_obj.shape}?",
                    "answer": obj.color,
                    "question_type": "color",
                    "subject_id": obj.id,
                    "object_id": ref_obj.id,
                    "rel_type": rel_type,
                    "rel_group": "PRIMARY",
                    "rel_phrase": phrase,
                })
                qa.append({
                    "question": f"What is the shape of the object {phrase} the {ref_obj.color} {ref_obj.shape}?",
                    "answer": obj.shape,
                    "question_type": "shape",
                    "subject_id": obj.id,
                    "object_id": ref_obj.id,
                    "rel_type": rel_type,
                    "rel_group": "PRIMARY",
                    "rel_phrase": phrase,
                })
    
    # Assign IDs
    for i, item in enumerate(qa):
        item["id"] = i
    
    return qa


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="vlm_levels_v2")
    ap.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--scenes_per_level", type=int, default=20)
    
    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--colors", type=str, default="")
    
    # For compatibility with generate_data.py interface
    ap.add_argument("--primary", action="store_true", help="Include PRIMARY questions (always used for v2)")
    ap.add_argument("--advanced", action="store_true", help="Not used for v2")

    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.base_dir, exist_ok=True)

    question_groups = ["primary"]  # Always use primary for v2

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
                
                # Save random state before generating attribute questions
                # (since rel_phrase uses random.choice which consumes state)
                random_state = random.getstate()
                
                # Now replace questions with attribute questions
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
                
                # Generate attribute questions (this may consume random state via rel_phrase)
                new_qa = _generate_attribute_qa(obj_list, ann["relations"])
                
                # Restore random state so subsequent scenes are not affected
                random.setstate(random_state)
                
                # Replace qa field
                ann["qa"] = new_qa
                
                # Save back
                with open(json_path, "w") as f:
                    json.dump(ann, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
