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
    elif strategy == 4:  # Minimalist+Priming
        return f"This is a Yes/No question: Is there a {base}?"
    else:
        return f"Does this image have a {base}?"

def add_baseline_strategies(strategies: List[int] = None):
    """Add baseline strategies to annotations. If strategies is None, adds all (0-4)."""
    if strategies is None:
        strategies = list(range(5))
    
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
            
            # Add only specified strategies
            for strategy in strategies:
                qa_key = f"qa_baseline_{strategy}"
                ann[qa_key] = qa_existence(objects, strategy)
            
            with open(path, "w") as f:
                json.dump(ann, f, indent=2)
            
            strat_str = ", ".join(f"qa_baseline_{s}" for s in strategies)
            print(f"Added {strat_str} to {path}")

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1:
        try:
            strategies = [int(arg) for arg in sys.argv[1:]]
            print(f"Adding strategies: {strategies}")
            add_baseline_strategies(strategies)
        except ValueError:
            print(f"Error: Invalid strategy number. Expected integers (0-4).")
            sys.exit(1)
    else:
        # Default: add all strategies
        print("No strategies specified. Adding all strategies (0-4)...")
        add_baseline_strategies()