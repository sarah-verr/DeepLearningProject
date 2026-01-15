"""
Add existential questions to annotation files.

This script augments annotation files with questions about object existence.
Each image gets two question types:
1. Yes/no questions (qa_existential_yesno): 4 balanced questions (2 "yes", 2 "no")
   - "Is there a {color} {shape}?" → "yes" or "no"
   
2. Attribute/object identification (qa_existential_attribute): ~4 questions with descriptive answers
   - "How many objects are present in the image?" → "<number>"
   - "What color is the {shape}?" → "<color>"
   - "What shape is the {color} object?" → "<shape>"
   - "Name an object as <color> <shape>" → "<color> <shape>" (accept any object as answer)
"""

import json
import os
import random
import sys
from typing import List, Dict

# Available colors and shapes
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan", "orange", "pink", "lime"]
SHAPES = ["square", "circle", "triangle", "star"]


def generate_existential_yesno_questions(objects: List[Dict]) -> List[Dict]:
    """Generate 4 balanced existential yes/no questions (2 yes, 2 no) about object existence.
    
    Args:
        objects: List of object dictionaries with 'color' and 'shape' keys
        
    Returns:
        List of QA dictionaries with 'question' and 'answer' keys
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
    """Generate questions that ask for color, shape, object count, or object name.
    
    IMPORTANT: Data generation strategy ensures:
    - Colors are UNIQUE per image (no two objects share the same color)
    - Shapes CAN repeat (multiple objects can have the same shape)
    
    Question types:
    1. "How many objects are present in the image?" → answer: "<number>"
    2. "What color is the {shape}?" → answer: "<color>" (only if shape is unique, unambiguous)
    3. "What shape is the {color} object?" → answer: "<shape>" (always unambiguous - colors are unique!)
    4. "Name an object as <color> <shape>" → answer: "<color> <shape>" (accept any object as answer)
    
    Args:
        objects: List of object dictionaries with 'color' and 'shape' keys
        
    Returns:
        List of QA dictionaries with 'question' and 'answer' keys (exactly 4 questions)
    """
    questions = []
    
    # Type 1: "How many objects are present in the image?" → answer: "<number>"
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
    
    # Type 2: "What color is the {shape}?" → answer: "<color>"
    # Only ask if shape is unique (unambiguous)
    for shape, colors in shape_to_colors.items():
        if len(colors) == 1:  # Unique shape, unambiguous answer
            question = f"What color is the {shape}?"
            questions.append({"question": question, "answer": colors[0]})
        # Skip ambiguous cases (multiple objects with same shape)
    
    # Type 3: "What shape is the {color} object?" → answer: "<shape>"
    # ALWAYS unambiguous because colors are unique!
    for color, shape in color_to_shape.items():
        question = f"What shape is the {color} object?"
        questions.append({"question": question, "answer": shape})
    
    # Type 4: "Name an object as <color> <shape>" → answer: list of all possible "<color> <shape>" pairs
    # Accept any object in the image as a valid answer
    all_answers = [f"{obj['color']} {obj['shape']}" for obj in objects]
    questions.append({
        "question": "Name an object as <color> <shape>",
        "answer": all_answers  # List of all valid answers
    })
    
    # Sample exactly 4 questions: 1 counting + (up to 1 color question) + shape questions + 1 name question
    # Note: Color questions may not exist if all shapes are ambiguous
    selected = []
    
    # Always include the counting question
    counting_q = [q for q in questions if "How many" in q["question"]]
    if counting_q:
        selected.append(counting_q[0])
    
    # Sample 1 color question (if available)
    color_questions = [q for q in questions if "What color" in q["question"]]
    if color_questions:
        selected.append(random.choice(color_questions))
    
    # Sample shape questions - add enough to reach 3 (then name question makes 4)
    shape_questions = [q for q in questions if "What shape" in q["question"]]
    if shape_questions:
        # Calculate how many shape questions we need (we'll add name question after)
        num_needed = 3 - len(selected)  # Target 3 before adding name question
        num_shape_questions = min(num_needed, len(shape_questions))
        selected.extend(random.sample(shape_questions, num_shape_questions))
    
    # Always include the "Name an object" question
    name_q = [q for q in questions if "Name an object" in q["question"]]
    if name_q:
        selected.append(name_q[0])
    
    # Shuffle and return exactly 4 questions
    random.shuffle(selected)
    return selected[:4]


def augment_annotations_with_existential_questions(base_dir: str = "../Synthetic-Data/vlm_levels"):
    """Add existential questions to all annotation files.
    
    Args:
        base_dir: Base directory containing level_X subdirectories
    """
    for level in range(5):  # level_0 to level_4
        ann_dir = os.path.join(base_dir, f"level_{level}", "ann")
        if not os.path.exists(ann_dir):
            print(f"Skipping level_{level}: annotation directory not found")
            continue
        
        json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        print(f"Processing level_{level}: {len(json_files)} annotation files")
        
        for json_file in json_files:
            json_path = os.path.join(ann_dir, json_file)
            
            try:
                with open(json_path, 'r') as f:
                    annotation = json.load(f)
                
                # Generate and add existential questions
                objects = annotation.get("objects", [])
                if not objects:
                    print(f"  Warning: No objects found in {json_file}, skipping")
                    continue
                
                # Generate yes/no questions
                yesno_qa = generate_existential_yesno_questions(objects)
                annotation["qa_existential_yesno"] = yesno_qa
                
                # Generate attribute/object identification questions
                attribute_qa = generate_attribute_identification_questions(objects)
                annotation["qa_existential_attribute"] = attribute_qa
                
                # Write back
                with open(json_path, "w") as f:
                    json.dump(annotation, f, indent=2)
                
            except Exception as e:
                print(f"  Error processing {json_file}: {e}")
                continue
        
        print(f"Completed level_{level}")


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "../Synthetic-Data/vlm_levels"
    print(f"Adding existential questions to annotations in: {base_dir}")
    augment_annotations_with_existential_questions(base_dir)
    print("Done!")

