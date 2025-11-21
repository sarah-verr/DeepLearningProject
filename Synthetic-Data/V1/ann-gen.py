import json, os, random
from typing import List, Dict, Tuple

PRIMARY_RELS = ("left_of","right_of","above","below")

def _rel_from_centers(a_xy: Tuple[int,int], b_xy: Tuple[int,int]) -> List[str]:
    ax, ay = a_xy; bx, by = b_xy
    rels = []
    if ax < bx: rels.append("left_of")
    if ax > bx: rels.append("right_of")
    if ay < by: rels.append("above")
    if ay > by: rels.append("below")
    return rels

def compute_relations_from_centers(objects: List[Dict]) -> List[Dict]:
    """Return directional relations for all ordered pairs (i != j)."""
    rels = []
    by_id = {o["id"]: o for o in objects}
    ids = sorted(by_id.keys())
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j: continue
            a = by_id[ids[i]]; b = by_id[ids[j]]
            a_xy = tuple(a["center"]); b_xy = tuple(b["center"])
            for r in _rel_from_centers(a_xy, b_xy):
                rels.append({"type": r, "subject_id": a["id"], "object_id": b["id"]})
    # optional: keep only primary relations and dedup
    seen = set(); out = []
    for r in rels:
        key = (r["type"], r["subject_id"], r["object_id"])
        if r["type"] in PRIMARY_RELS and key not in seen:
            seen.add(key); out.append(r)
    return out

def _phrase(r: str) -> str:
    return {"left_of":"left of","right_of":"right of","above":"above","below":"below"}[r]

def _inverse(r: str) -> str:
    return {"left_of":"right_of","right_of":"left_of","above":"below","below":"above"}[r]

def captions_from_relations(objects: List[Dict], relations: List[Dict], max_caps: int = 6) -> List[str]:
    by_id = {o["id"]: o for o in objects}
    caps = []
    for r in relations:
        a = by_id[r["subject_id"]]; b = by_id[r["object_id"]]
        caps.append(f"The {a['color']} {a['shape']} is { _phrase(r['type']) } the {b['color']} {b['shape']}.")
        if len(caps) >= max_caps: break
    return caps

def qa_from_relations(objects: List[Dict], relations: List[Dict], max_qa: int = 8) -> List[Dict]:
    """Produces a balanced set of yes/no QA."""
    by_id = {o["id"]: o for o in objects}
    ids = sorted(by_id.keys())
    
    yes_qa = []
    # 1. Generate all "yes" questions from true relations
    for r in relations:
        a = by_id[r["subject_id"]]; b = by_id[r["object_id"]]
        # forward
        qf = f"Is the {a['color']} {a['shape']} { _phrase(r['type']) } the {b['color']} {b['shape']}?"
        yes_qa.append({"question": qf, "answer": "yes"})
        # inverse
        inv = _inverse(r["type"])
        qi = f"Is the {b['color']} {b['shape']} { _phrase(inv) } the {a['color']} {a['shape']}?"
        yes_qa.append({"question": qi, "answer": "yes"})
    
    no_qa = []
    true_rels_set = set((r["type"], r["subject_id"], r["object_id"]) for r in relations)
    
    # 2. Generate "no" questions from false relations
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j: continue
            a_id = ids[i]; b_id = ids[j]
            a = by_id[a_id]; b = by_id[b_id]
            
            for rel_type in PRIMARY_RELS:
                # If this specific relation is NOT in the true set, it's a false statement
                if (rel_type, a_id, b_id) not in true_rels_set:
                    q_no = f"Is the {a['color']} {a['shape']} { _phrase(rel_type) } the {b['color']} {b['shape']}?"
                    no_qa.append({"question": q_no, "answer": "no"})

    # 3. Balance the lists. Aim for roughly 50/50 yes/no
    num_yes = len(yes_qa)
    num_no = len(no_qa)
    
    # Sample to balance
    if num_yes > 0 and num_no > 0:
        if num_yes > num_no:
            yes_qa = random.sample(yes_qa, num_no)
        else:
            no_qa = random.sample(no_qa, num_yes)
    
    # 4. Combine, shuffle, and take the max
    combined_qa = yes_qa + no_qa
    random.shuffle(combined_qa)
    
    return combined_qa[:max_qa]



ANN_DIR = "base_scenes/ann_base"          # change to your folder
WRITE_NEW = True                     # False = overwrite JSONs

from pathlib import Path
from typing import Dict, Any

# ---- import utilities from above (paste here) ----
# compute_relations_from_centers, captions_from_relations, qa_from_relations
# --------------------------------------------------

def process_one(json_path: str) -> Dict[str, Any]:
    with open(json_path) as f:
        ann = json.load(f)
    objects = ann["objects"]
    relations = compute_relations_from_centers(objects)
    caps = captions_from_relations(objects, relations, max_caps=6)
    qa = qa_from_relations(objects, relations, max_qa=8)

    ann["relations"] = relations
    ann["captions"]  = caps
    ann["qa"]        = qa
    return ann

def main():
    paths = sorted([str(Path(ANN_DIR)/f) for f in os.listdir(ANN_DIR) if f.endswith(".json")])
    for p in paths:
        updated = process_one(p)
        if WRITE_NEW:
            newp = p.replace(".json", "_with_rel.json")
            newp = p.replace("ann_base", "ann")
            with open(newp, "w") as f: json.dump(updated, f, indent=2)
        else:
            with open(p, "w") as f: json.dump(updated, f, indent=2)
    print(f"Updated {len(paths)} files with relations + captions + QA.")

if __name__ == "__main__":
    main()
