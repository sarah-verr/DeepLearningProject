import json, os, random
from typing import List, Dict
import sys
sys.path.append('../Synthetic-Data')
from core_config import COLORS_RGB, ALL_SHAPES

def qa_existence(objects: List[Dict], strategy: int = 1) -> List[Dict]:
    """Produces 2 yes and 2 no QA about existence of colored shapes with different prompting strategies."""
    existing = set((o['color'], o['shape']) for o in objects)
    
    all_possible = [(c, s) for c in COLORS_RGB.keys() for s in ALL_SHAPES]
    
    yes_qa = []
    no_qa = []
    
    for color, shape in existing:
        question = generate_question(color, shape, strategy, is_yes=True)
        yes_qa.append({"question": question, "answer": "yes"})
    
    for color, shape in all_possible:
        if (color, shape) not in existing:
            question = generate_question(color, shape, strategy, is_yes=False)
            no_qa.append({"question": question, "answer": "no"})
    
    # Select 2 yes and 2 no if possible
    selected_yes = random.sample(yes_qa, min(2, len(yes_qa)))
    selected_no = random.sample(no_qa, min(2, len(no_qa)))
    
    combined = selected_yes + selected_no
    random.shuffle(combined)
    return combined[:4]

def generate_question(color: str, shape: str, strategy: int, is_yes: bool) -> str:
    """Generate question based on strategy."""
    base = f"{color} {shape}"
    
    if strategy == 0:  # Minimalist
        return f"Is there a {base}?"
    elif strategy == 1:  # Contextual
        return f"Does this image have a {base}?"
    elif strategy == 2:  # Strict Format
        return f"Does this image have a {base}? Answer: Yes or No."
    elif strategy == 3:  # Priming
        return f"This is a Yes/No question: Does this image have a {base}?"
    else:
        return f"Does this image have a {base}?"

def add_baseline_strategies():
    base_dir = "../Synthetic-Data/vlm_levels"
    for level in range(7):  # level_0 to level_6
        ann_dir = os.path.join(base_dir, f"level_{level}", "ann")
        if not os.path.exists(ann_dir):
            continue
        json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        for jf in json_files:
            path = os.path.join(ann_dir, jf)
            with open(path, 'r') as f:
                ann = json.load(f)
            objects = ann["objects"]
            
            # Add all 4 strategies
            for strategy in range(4):
                qa_key = f"qa_baseline_{strategy}"
                ann[qa_key] = qa_existence(objects, strategy)
            
            with open(path, "w") as f:
                json.dump(ann, f, indent=2)
            print(f"Added qa_baseline_0 to qa_baseline_3 to {path}")

if __name__ == "__main__":
    add_baseline_strategies()