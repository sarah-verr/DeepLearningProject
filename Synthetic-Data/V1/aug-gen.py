import os, json, random, math
from typing import Tuple, Dict, List
from PIL import Image, ImageOps

BASE_DIR = "base_scenes"         # input folder with images+ann
IMG_DIR  = os.path.join(BASE_DIR, "images")
ANN_DIR  = os.path.join(BASE_DIR, "ann")

OUT_DIR  = "synthetic_vlm_rel_aug"
OUT_IMG  = os.path.join(OUT_DIR, "images")
OUT_ANN  = os.path.join(OUT_DIR, "ann")
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_ANN, exist_ok=True)

# ----- Relation remapping for transforms -----
REL_MAP = {
    "id":   {"left_of":"left_of","right_of":"right_of","above":"above","below":"below",
                   "touching":"touching","overlapping":"overlapping"},
    "fh":     {"left_of":"right_of","right_of":"left_of","above":"above","below":"below",
                   "touching":"touching","overlapping":"overlapping"},
    "fv":     {"left_of":"left_of","right_of":"right_of","above":"below","below":"above",
                   "touching":"touching","overlapping":"overlapping"},
    "r90cw":   {"left_of":"above","right_of":"below","above":"right_of","below":"left_of",
                   "touching":"touching","overlapping":"overlapping"},
    "r90ccw":  {"left_of":"below","right_of":"above","above":"left_of","below":"right_of",
                   "touching":"touching","overlapping":"overlapping"},
    "r180":     {"left_of":"right_of","right_of":"left_of","above":"below","below":"above",
                   "touching":"touching","overlapping":"overlapping"},
}

def rel_phrase(r: str) -> str:
    return {
        "left_of":"left of","right_of":"right of",
        "above":"above","below":"below",
        "touching":"touching","overlapping":"overlapping with"
    }[r]

def inverse_rel(r: str) -> str:
    return {
        "left_of":"right_of","right_of":"left_of",
        "above":"below","below":"above",
        "touching":"touching","overlapping":"overlapping"
    }[r]

# ---------- bbox transforms ----------
def apply_bbox_flip_h(b, W):
    x0,y0,x1,y1 = b
    return (W - x1, y0, W - x0, y1)

def apply_bbox_flip_v(b, H):
    x0,y0,x1,y1 = b
    return (x0, H - y1, x1, H - y0)

def apply_bbox_rot90_cw(b, W, H):
    # (x,y) -> (y, W-1-x)
    x0,y0,x1,y1 = b
    pts = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    rot = [(p[1], W-1-p[0]) for p in pts]
    xs = [p[0] for p in rot]; ys=[p[1] for p in rot]
    return (min(xs), min(ys), max(xs), max(ys))

def apply_bbox_rot90_ccw(b, W, H):
    # (x,y) -> (H-1-y, x)
    x0,y0,x1,y1 = b
    pts = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    rot = [(H-1-p[1], p[0]) for p in pts]
    xs = [p[0] for p in rot]; ys=[p[1] for p in rot]
    return (min(xs), min(ys), max(xs), max(ys))

def apply_bbox_rot180(b, W, H):
    x0,y0,x1,y1 = b
    return (W - x1, H - y1, W - x0, H - y0)

def bbox_center(b):
    x0,y0,x1,y1 = b
    return ((x0+x1)//2, (y0+y1)//2)

# ---------- text regeneration ----------
def make_captions(objects: List[Dict], relations: List[Dict], max_caps=6) -> List[str]:
    by_id = {o["id"]: o for o in objects}
    out = []
    for r in relations:
        a = by_id[r["subject_id"]]; b = by_id[r["object_id"]]
        out.append(f"The {a['color']} {a['shape']} is {rel_phrase(r['type'])} the {b['color']} {b['shape']}.")
        if len(out) >= max_caps: break
    return out

# (This function REPLACES the old make_qa)
def make_qa(objects: List[Dict], relations: List[Dict], max_qa=8) -> List[Dict]:
    """Produces a balanced set of yes/no QA."""
    by_id = {o["id"]: o for o in objects}
    ids = sorted(by_id.keys())
    
    yes_qa = []
    # 1. Generate all "yes" questions from true relations
    for r in relations:
        a = by_id[r["subject_id"]]; b = by_id[r["object_id"]]
        # forward
        qf = f"Is the {a['color']} {a['shape']} {rel_phrase(r['type'])} the {b['color']} {b['shape']}?"
        yes_qa.append({"question": qf, "answer": "yes"})
        # inverse
        inv = inverse_rel(r["type"])
        qi = f"Is the {b['color']} {b['shape']} {rel_phrase(inv)} the {a['color']} {a['shape']}?"
        yes_qa.append({"question": qi, "answer": "yes"})
    
    no_qa = []
    true_rels_set = set((r["type"], r["subject_id"], r["object_id"]) for r in relations)
    
    # Define primary relations for checking "no" questions
    PRIMARY_RELS = ("left_of","right_of","above","below")

    # 2. Generate "no" questions from false relations
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j: continue
            a_id = ids[i]; b_id = ids[j]
            a = by_id[a_id]; b = by_id[b_id]
            
            for rel_type in PRIMARY_RELS:
                # If this specific relation is NOT in the true set, it's a false statement
                if (rel_type, a_id, b_id) not in true_rels_set:
                    q_no = f"Is the {a['color']} {a['shape']} {rel_phrase(rel_type)} the {b['color']} {b['shape']}?"
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

# ---------- core transform ----------
def transform_ann(ann, tr_name, W, H, new_W, new_H, crop_box=None):
    """
    Transforms objects (bbox+center), remaps ALL relations in ann['relations'] (list),
    then regenerates captions and QA to stay consistent.
    Returns a NEW annotation dict, or None if an object goes invalid after crop.
    """
    if "objects" not in ann:
        return None
    objs = []
    # 1) transform objects
    for obj in ann["objects"]:
        b = tuple(obj["bbox"])
        if tr_name == "fh":
            b2 = apply_bbox_flip_h(b, W)
        elif tr_name == "fv":
            b2 = apply_bbox_flip_v(b, H)
        elif tr_name == "r90cw":
            b2 = apply_bbox_rot90_cw(b, W, H)
        elif tr_name == "r90ccw":
            b2 = apply_bbox_rot90_ccw(b, W, H)
        elif tr_name == "r180":
            b2 = apply_bbox_rot180(b, W, H)
        else:
            b2 = b

        # crop adjust
        if crop_box is not None:
            cx0, cy0, cx1, cy1 = crop_box
            x0,y0,x1,y1 = b2
            x0 -= cx0; x1 -= cx0
            y0 -= cy0; y1 -= cy0
            # clip
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(new_W-1, x1); y1 = min(new_H-1, y1)
            b2 = (x0,y0,x1,y1)

        # reject degenerate box
        if b2[2] <= b2[0] or b2[3] <= b2[1]:
            return None

        o2 = dict(obj)
        o2["bbox"]   = [int(v) for v in b2]
        o2["center"] = [int(v) for v in bbox_center(b2)]
        objs.append(o2)

    # 2) remap ALL relations in list
    new_relations = []
    rels = ann.get("relations", [])
    mapping = REL_MAP[tr_name]
    for r in rels:
        t_old = r["type"]
        t_new = mapping.get(t_old, t_old)
        new_relations.append({
            "type": t_new,
            "subject_id": r["subject_id"],
            "object_id": r["object_id"]
        })

    # 3) rebuild captions + QA
    new_captions = make_captions(objs, new_relations, max_caps=6)
    new_qa = make_qa(objs, new_relations, max_qa=8)

    # 4) assemble new ann
    ann2 = dict(ann)
    ann2["objects"]   = objs
    ann2["relations"] = new_relations
    ann2["captions"]  = new_captions
    ann2["qa"]        = new_qa
    return ann2

def random_crop_box(W, H, min_scale=0.85):
    cw = int(W * random.uniform(min_scale, 1.0))
    ch = int(H * random.uniform(min_scale, 1.0))
    x0 = random.randint(0, max(0, W - cw))
    y0 = random.randint(0, max(0, H - ch))
    return (x0, y0, x0+cw, y0+ch)

def augment_one(img_path, ann_path, idx):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    with open(ann_path) as f:
        ann = json.load(f)

    transforms = [
        ("id", lambda im: im),
        ("fh",   lambda im: ImageOps.mirror(im)),
        ("fv",   lambda im: ImageOps.flip(im)),
        ("r90cw", lambda im: im.transpose(Image.Transpose.ROTATE_270)),
        ("r90ccw",lambda im: im.transpose(Image.Transpose.ROTATE_90)),
        ("r180",   lambda im: im.transpose(Image.Transpose.ROTATE_180)),
    ]

    out_count = 0
    for tname, tf in transforms:
        im2 = tf(img)
        new_W, new_H = im2.size

        # optional crop
        if random.random() < 0.5:
            crop = random_crop_box(new_W, new_H, min_scale=0.9)
            im2c = im2.crop(crop)
            ann2 = transform_ann(ann, tname, W, H, im2c.size[0], im2c.size[1], crop_box=crop)
            if ann2 is None:
                continue
            im_save, ann_save = im2c, ann2
        else:
            ann2 = transform_ann(ann, tname, W, H, new_W, new_H, crop_box=None)
            if ann2 is None:
                continue
            im_save, ann_save = im2, ann2

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_img_name = f"{base}_{tname}.png"
        out_ann_name = out_img_name.replace(".png",".json")
        im_save.save(os.path.join(OUT_IMG, out_img_name), "PNG")
        with open(os.path.join(OUT_ANN, out_ann_name), "w") as f:
            json.dump(ann_save, f, indent=2)
        out_count += 1

    return out_count

def main():
    files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
    # If you only want to augment a small subset, slice here (e.g., [:10]).
    total = 0
    for i, fn in enumerate(files):
        img_path = os.path.join(IMG_DIR, fn)
        ann_path = os.path.join(ANN_DIR, fn.replace(".png",".json"))
        if not os.path.exists(ann_path):
            print(f"⚠️  Missing ann for {fn}, skipping")
            continue
        total += augment_one(img_path, ann_path, i)
    print(f"Augmented samples written: {total} → {OUT_DIR}/")

if __name__ == "__main__":
    main()
