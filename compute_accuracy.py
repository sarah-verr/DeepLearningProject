import torch
import os
import json
import argparse
import time
import csv
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE_DATA_PATH = "/home/kkarthikeyan/deep-learning/DeepLearningProject/Synthetic-Data/vlm_levels"


def infer_rel_group(rel_type: str | None) -> str:
    """Best-effort group inference for older JSONs without rel_group."""
    if not rel_type:
        return "UNKNOWN"
    rel_type = rel_type.strip().lower()
    if rel_type in {"left_of", "right_of", "above", "below"}:
        return "PRIMARY"
    # In this codebase, CONTACT is treated as part of ADVANCED.
    if rel_type in {"touching", "overlapping", "inside", "encapsulates", "near", "far", "next_to", "beside"}:
        return "ADVANCED"
    return "UNKNOWN"

def get_prediction(outputs, processor, prompt_context="ASSISTANT:"):
    """Extracts 'yes' or 'no' based on logit probability."""
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)

    yes_tokens = [
        processor.tokenizer.encode(f"{prompt_context} Yes", add_special_tokens=False)[-1],
        processor.tokenizer.encode(f"{prompt_context} yes", add_special_tokens=False)[-1]
    ]
    no_tokens = [
        processor.tokenizer.encode(f"{prompt_context} No", add_special_tokens=False)[-1],
        processor.tokenizer.encode(f"{prompt_context} no", add_special_tokens=False)[-1]
    ]

    prob_yes = sum([probs[t_id].item() for t_id in yes_tokens if t_id < len(probs)])
    prob_no = sum([probs[t_id].item() for t_id in no_tokens if t_id < len(probs)])

    return "yes" if prob_yes > prob_no else "no"

def main():
    parser = argparse.ArgumentParser(description="Compute per-question accuracy and log model calls")
    parser.add_argument("--levels", type=str, nargs="+", required=True, help="List of levels (e.g. 1 2 3)")
    parser.add_argument("--output_json", type=str, default="evaluation_results.json", help="Path to save JSON stats")
    parser.add_argument("--log_file", type=str, default="model_calls_log.csv", help="CSV to log every prompt/response")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    level_results = {}

    # Initialize CSV Log File
    with open(args.log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Level",
            "Image_Path",
            "Rel_Group",
            "Rel_Type",
            "Prompt",
            "Ground_Truth",
            "Model_Prediction",
            "Result",
        ])

    for level in args.levels:
        level_dir = os.path.join(BASE_DATA_PATH, f"level_{level}")
        ann_path = os.path.join(level_dir, "ann")
        img_path = os.path.join(level_dir, "images")

        if not os.path.exists(ann_path):
            print(f"Skipping Level {level}: Directory not found.")
            continue

        json_files = [f for f in os.listdir(ann_path) if f.endswith('.json')]
        correct_questions = 0
        total_questions = 0

        # Per-group breakdown
        group_stats = {}

        print(f"\n--- Processing Level {level}: {len(json_files)} images ---")

        for filename in tqdm(json_files, desc=f"Level {level}"):
            file_id = filename.replace(".json", "")
            with open(os.path.join(ann_path, filename), 'r') as f:
                data = json.load(f)
            
            image_rel_path = os.path.join(img_path, f"{file_id}.png")
            if not os.path.exists(image_rel_path): continue
            image = Image.open(image_rel_path).convert('RGB')

            for qa in data.get('qa', []):
                question = qa['question']
                gt = qa['answer'].lower().strip()

                rel_type = qa.get("rel_type")
                rel_group = qa.get("rel_group") or infer_rel_group(rel_type)

                if rel_group not in group_stats:
                    group_stats[rel_group] = {"correct": 0, "total": 0}
                
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=1, 
                        output_scores=True, 
                        return_dict_in_generate=True
                    )
                
                pred = get_prediction(outputs, processor)
                total_questions += 1
                is_correct = (pred == gt)
                if is_correct:
                    correct_questions += 1

                group_stats[rel_group]["total"] += 1
                if is_correct:
                    group_stats[rel_group]["correct"] += 1

                # Append to Model Call Log
                with open(args.log_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        level,
                        image_rel_path,
                        rel_group,
                        rel_type if rel_type is not None else "",
                        question,
                        gt,
                        pred,
                        "CORRECT" if is_correct else "INCORRECT"
                    ])

        accuracy = (correct_questions / total_questions) * 100 if total_questions > 0 else 0
        # Add per-group accuracies
        group_acc = {}
        for g, s in group_stats.items():
            g_total = s.get("total", 0)
            g_correct = s.get("correct", 0)
            group_acc[g] = {
                "correct": g_correct,
                "total": g_total,
                "acc": (g_correct / g_total) * 100 if g_total > 0 else 0,
            }

        level_results[level] = {
            "correct": correct_questions,
            "total": total_questions,
            "acc": accuracy,
            "by_rel_group": group_acc,
        }
        print(f"Level {level} Finished: {correct_questions}/{total_questions} correct ({accuracy:.2f}%)")

    # Save summary files
    with open(args.output_json, "w") as jf:
        json.dump(level_results, jf, indent=4)

    report_name = "evaluation_summary.txt"
    with open(report_name, "w") as f:
        f.write(f"Evaluation Summary Report | Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*45 + "\n")
        f.write(f"{'Level':<10} | {'Correct/Total':<18} | {'Accuracy %':<10}\n")
        f.write("-" * 45 + "\n")
        for lvl, stats in level_results.items():
            f.write(f"{lvl:<10} | {stats['correct']}/{stats['total']:<17} | {stats['acc']:.2f}%\n")
            by_group = stats.get("by_rel_group", {})
            if isinstance(by_group, dict) and by_group:
                for g_name, g_stats in by_group.items():
                    f.write(
                        f"  - {g_name:<8} : {g_stats['correct']}/{g_stats['total']} ({g_stats['acc']:.2f}%)\n"
                    )
        f.write("="*45 + "\n")

    print(f"\nResults saved to '{args.output_json}', '{report_name}', and call logs in '{args.log_file}'.")

if __name__ == "__main__":
    main()