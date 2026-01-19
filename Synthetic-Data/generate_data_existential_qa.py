"""
Generate vlm_levels_existential_qa dataset with existential questions instead of relational questions.

Uses the same generation logic as generate_data.py to ensure identical images/objects/relations,
then replaces the questions with existential questions (existence yes/no + attribute/identification questions).
"""

import os, argparse, random, json
from core_config import ALL_SHAPES, COLORS_RGB, LEVEL_CONFIG
from core_utils import assign_colors, render_and_save
from levels import get_level_fns
from typing import List, Dict, Any

# Available colors and shapes (matching add_existential_questions.py)
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "pink", "lime"]
SHAPES = ["square", "circle", "triangle", "star"]


def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def generate_existential_yesno_questions(objects: List[Dict]) -> List[Dict]:
    """Generate 4 balanced existential yes/no questions (2 yes, 2 no) about object existence.
    
    Args:
        objects: List of object dictionaries with 'color' and 'shape' keys
        
    Returns:
        List of QA dictionaries with 'question', 'answer', and 'id' keys
    """
    # Get set of existing (color, shape) pairs
    existing_objects = set((obj['color'], obj['shape']) for obj in objects)
    
    # All possible color-shape combinations
    all_combinations = [(color, shape) for color in COLORS for shape in SHAPES]
    
    # Generate yes questions (objects that exist)
    yes_questions = []
    for color, shape in existing_objects:
        question = f"Is there a {color} {shape}?"
        yes_questions.append({"question": question, "answer": "yes"})
    
    # Generate no questions (objects that don't exist)
    no_questions = []
    for color, shape in all_combinations:
        if (color, shape) not in existing_objects:
            question = f"Is there a {color} {shape}?"
            no_questions.append({"question": question, "answer": "no"})
    
    # Sample 2 yes and 2 no questions
    selected_yes = random.sample(yes_questions, min(2, len(yes_questions)))
    selected_no = random.sample(no_questions, min(2, len(no_questions)))
    
    # Combine and shuffle
    all_questions = selected_yes + selected_no
    random.shuffle(all_questions)
    
    return all_questions[:4]


def generate_attribute_identification_questions(objects: List[Dict]) -> List[Dict]:
    """Generate questions that ask for color, shape, or object count.
    
    IMPORTANT: Data generation strategy ensures:
    - Colors are UNIQUE per image (no two objects share the same color)
    - Shapes CAN repeat (multiple objects can have the same shape)
    
    Question types:
    1. "How many objects are present in the image?" → answer: "<number>" (1 question)
    2. "What color is the {shape}?" → answer: "<color>" (2 questions, only if shape is unique)
    3. "What shape is the {color} object?" → answer: "<shape>" (2 questions, always unambiguous - colors are unique!)
    
    Args:
        objects: List of object dictionaries with 'color' and 'shape' keys
        
    Returns:
        List of QA dictionaries with 'question' and 'answer' keys (exactly 5 questions: 1 number + 2 color + 2 shape)
    """
    questions = []
    
    # Type 1: "How many objects are present in the image?" → answer: "<number>" (exactly 1)
    num_objects = len(objects)
    questions.append({
        "question": "How many objects are present in the image?",
        "answer": str(num_objects)
    })
    
    # Build indices for efficient lookup
    shape_to_colors = {}  # shape -> list of colors (can have multiple)
    color_to_shape = {}   # color -> shape (one-to-one mapping since colors are unique)
    
    for obj in objects:
        color = obj['color']
        shape = obj['shape']
        shape_to_colors.setdefault(shape, []).append(color)
        # Colors are unique, so this is a one-to-one mapping
        color_to_shape[color] = shape
    
    # Type 2: "What color is the {shape}?" → answer: "<color>" (exactly 2 questions)
    # Only ask if shape is unique (unambiguous)
    color_questions = []
    for shape, colors in shape_to_colors.items():
        if len(colors) == 1:  # Unique shape, unambiguous answer
            question = f"What color is the {shape}?"
            color_questions.append({"question": question, "answer": colors[0]})
    
    # Sample exactly 2 color questions (if available)
    if len(color_questions) >= 2:
        questions.extend(random.sample(color_questions, 2))
    elif len(color_questions) == 1:
        # If only 1 unique shape, use it (but we need 2 color questions)
        questions.append(color_questions[0])
        # Note: If there's only 1 unique shape, we can't generate 2 color questions
        # In this case, we'll have fewer than 2, but that's the best we can do
    # If no unique shapes, no color questions can be generated
    
    # Type 3: "What shape is the {color} object?" → answer: "<shape>" (exactly 2 questions)
    # ALWAYS unambiguous because colors are unique!
    shape_questions = []
    for color, shape in color_to_shape.items():
        question = f"What shape is the {color} object?"
        shape_questions.append({"question": question, "answer": shape})
    
    # Sample exactly 2 shape questions
    if len(shape_questions) >= 2:
        questions.extend(random.sample(shape_questions, 2))
    elif len(shape_questions) == 1:
        questions.append(shape_questions[0])
        # Note: If there's only 1 object, we can only generate 1 shape question
    # If no objects, no shape questions can be generated
    
    # Shuffle and return (should be exactly 5 questions if possible: 1 number + 2 color + 2 shape)
    random.shuffle(questions)
    return questions




def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="vlm_levels_existential_qa")
    ap.add_argument("--levels", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--scenes_per_level", type=int, default=20)
    
    ap.add_argument("--img_size", type=int, default=336)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--colors", type=str, default="")
    
    # For compatibility with generate_data.py interface
    ap.add_argument("--primary", action="store_true", help="Not used for existential_qa, kept for compatibility")
    ap.add_argument("--advanced", action="store_true", help="Not used for existential_qa, kept for compatibility")

    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.base_dir, exist_ok=True)

    question_groups = ["primary"]  # For compatibility with render_and_save, but qa will be replaced

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
                # This creates the base annotation with relational questions
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
                
                # Save random state before generating existential questions
                random_state = random.getstate()
                
                # Now replace questions with existential questions
                with open(json_path, "r") as f:
                    ann = json.load(f)
                
                # Generate and add existential questions (same as add_existential_questions.py)
                objects = ann.get("objects", [])
                if objects:
                    # Generate yes/no questions
                    yesno_qa = generate_existential_yesno_questions(objects)
                    ann["qa_existential_yesno"] = yesno_qa
                    
                    # Generate attribute/object identification questions
                    attribute_qa = generate_attribute_identification_questions(objects)
                    ann["qa_existential_attribute"] = attribute_qa
                
                # Remove or keep the original qa field (remove it since we're replacing relational questions)
                if "qa" in ann:
                    del ann["qa"]
                
                # Restore random state so subsequent scenes are not affected
                random.setstate(random_state)
                
                # Save back
                with open(json_path, "w") as f:
                    json.dump(ann, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
