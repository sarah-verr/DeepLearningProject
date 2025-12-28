import json
import os
from collections import defaultdict

def aggregate_baseline_strategies():
    base_dir = "baseline_results"
    results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    
    # Walk through all level directories
    for level_dir in os.listdir(base_dir):
        if not level_dir.startswith("level_"):
            continue
        level = int(level_dir.split("_")[1])
        
        level_path = os.path.join(base_dir, level_dir)
        if not os.path.isdir(level_path):
            continue
            
        # Walk through all image directories
        for image_dir in os.listdir(level_path):
            image_path = os.path.join(level_path, image_dir)
            if not os.path.isdir(image_path):
                continue
                
            # Walk through all strategy directories
            for strategy_dir in os.listdir(image_path):
                if not strategy_dir.startswith("strategy_"):
                    continue
                strategy = int(strategy_dir.split("_")[1])  # Extract number from "strategy_X_timestamp"
                
                strategy_path = os.path.join(image_path, strategy_dir)
                if not os.path.isdir(strategy_path):
                    continue
                    
                # Look for overview/results.json
                results_file = os.path.join(strategy_path, "overview", "results.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        
                    # Aggregate results
                    accuracy = data.get("accuracy", 0)
                    num_questions = len(data.get("results", []))
                    
                    results[strategy][level]["total"] += num_questions
                    results[strategy][level]["correct"] += int(accuracy * num_questions)
    
    # Calculate accuracies and organize output
    output = {}
    for strategy in sorted(results.keys()):
        output[str(strategy)] = {}
        for level in sorted(results[strategy].keys()):
            stats = results[strategy][level]
            if stats["total"] > 0:
                stats["acc"] = (stats["correct"] / stats["total"]) * 100
            else:
                stats["acc"] = 0.0
            output[str(strategy)][str(level)] = stats
    
    with open("evaluation_results_baseline_strategies.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Aggregated baseline strategies results saved to evaluation_results_baseline_strategies.json")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    aggregate_baseline_strategies()