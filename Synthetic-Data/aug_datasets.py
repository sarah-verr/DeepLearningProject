#!/usr/bin/env python3
import os, json, random, math
from typing import Tuple, Dict, List
from PIL import Image, ImageOps

# ------------ CONFIG ------------

<<<<<<< HEAD
# Root folders for your generated 6-level dataset
ROOT_IN  = "vlm_levels"         # where generate_dataset.py wrote level_1..level_6
ROOT_OUT = "vlm_levels_aug"     # where we will write augmented data

LEVELS   = [1, 2, 3, 4, 5, 6]   # which levels to augment
=======
ROOT_IN  = "data/vlm_levels"         # where generate_dataset.py wrote level_1..level_6
ROOT_OUT = "data/vlm_levels_aug"     # where we will write augmented data

LEVELS   = [0, 1, 2, 3, 4]   # which levels to augment
>>>>>>> 5377b2f0425a36e119609f3a4180b3e1e327ba0c
N_AUG_PER_IMAGE = 6             # how many transforms per image (max; we sample from TRANSFORMS)

os.makedirs(ROOT_OUT, exist_ok=True)

# ----- Relation remapping for transforms -----
REL_MAP = {
    "id": {
        "left_of":"left_of","right_of":"right_of",
        "above":"above","below":"below",
        "touching":"touching","overlapping":"overlapping",
        # advanced relations stay the same
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    },
    "fh": {  # flip horizontal (mirror)
        "left_of":"right_of","right_of":"left_of",
        "above":"above","below":"below",
        "touching":"touching","overlapping":"overlapping",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    },
    "fv": {  # flip vertical
        "left_of":"left_of","right_of":"right_of",
        "above":"below","below":"above",
        "touching":"touching","overlapping":"overlapping",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    },
    "r90cw": {  # rotate 90° clockwise
        "left_of":"above","right_of":"below",
        "above":"right_of","below":"left_of",
        "touching":"touching","overlapping":"overlapping",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    },
    "r90ccw": {  # rotate 90° counter-clockwise
        "left_of":"below","right_of":"above",
        "above":"left_of","below":"right_of",
        "touching":"touching","overlapping":"overlapping",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    },
    "r180": {  # rotate 180°
        "left_of":"right_of","right_of":"left_of",
        "above":"below","below":"above",
        "touching":"touching","overlapping":"overlapping",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    },
}

PRIMARY_RELS = ("left_of","right_of","above","below")

# ------------ Text helpers ------------

def rel_phrase(r: str) -> str:
    return {
        "left_of":"left of","right_of":"right of",
        "above":"above","below":"below",
        "touching":"touching","overlapping":"overlapping with",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next to","beside":"beside",
    }[r]

def inverse_rel(r: str) -> str:
    return {
        "left_of":"right_of","right_of":"left_of",
        "above":"below","below":"above",
        # symmetric ones
        "touching":"touching","overlapping":"overlapping",
        "inside":"inside","near":"near","far":"far",
        "next_to":"next_to","beside":"beside",
    }[r]

# ------------ bbox transforms ------------

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

# ------------ text regeneration (captions + QA) ------------

def make_captions(objects: List[Dict], relations: List[Dict], max_caps=6) -> List[str]:
    by_id = {o["id"]: o for o in objects}
    out = []
    for r in relations:
        a = by_id[r["subject_id"]]; b = by_id[r["object_id"]]
        out.append(f"The {a['color']} {a['shape']} is {rel_phrase(r['type'])} the {b['color']} {b['shape']}.")
        if len(out) >= max_caps: break
    return out

def make_qa(objects: List[Dict], relations: List[Dict], max_qa=8) -> List[Dict]:
    """Produces a balanced set of yes/no QA."""
    by_id = {o["id"]: o for o in objects}
    ids = sorted(by_id.keys())
    
    yes_qa = []
    # 1. Generate 'yes' questions from true relations
    for r in relations:
        a = by_id[r["subject_id"]]; b = by_id[r["object_id"]]
        # forward
        qf = f"Is the {a['color']} {a['shape']} {rel_phrase(r['type'])} the {b['color']} {b['shape']}?"
        yes_qa.append({"question": qf, "answer": "yes"})
        # inverse
        inv = inverse_rel(r["type"])
        qi = f"Is the {b['color']} {b['shape']} {rel_phrase(inv)} the {a['color']} {a['shape']}?"
        yes_qa.append({"question": qi, "answer": "yes"})
    
    # 2. Generate 'no' questions from false primary relations
    no_qa = []
    true_rels_set = set((r["type"], r["subject_id"], r["object_id"]) for r in relations)

    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j: 
                continue
            a_id = ids[i]; b_id = ids[j]
            a = by_id[a_id]; b = by_id[b_id]
            for rel_type in PRIMARY_RELS:
                if (rel_type, a_id, b_id) not in true_rels_set:
                    q_no = f"Is the {a['color']} {a['shape']} {rel_phrase(rel_type)} the {b['color']} {b['shape']}?"
                    no_qa.append({"question": q_no, "answer": "no"})

    # 3. Balance yes/no
    num_yes = len(yes_qa)
    num_no  = len(no_qa)
    if num_yes > 0 and num_no > 0:
        if num_yes > num_no:
            yes_qa = random.sample(yes_qa, num_no)
        else:
            no_qa  = random.sample(no_qa, num_yes)

    combined = yes_qa + no_qa
    random.shuffle(combined)
    return combined[:max_qa]

# ------------ annotation transform ------------

def transform_ann(ann, tr_name, W, H, new_W, new_H, crop_box=None):
    """
    Transforms objects (bbox+center), remaps ALL relations in ann['relations'] (list),
    then regenerates captions and QA to stay consistent.
    Returns a NEW annotation dict, or None if an object goes invalid after crop.
    """
    if "objects" not in ann:
        return None

    objs = []
    for obj in ann["objects"]:
        b = tuple(obj["bbox"])
        # 1) geometric transform
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
        else:  # "id"
            b2 = b

        # 2) crop adjustment
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

    # 3) remap relations
    new_relations = []
    rels = ann.get("relations", [])
    mapping = REL_MAP[tr_name]
    for r in rels:
        t_old = r["type"]
        t_new = mapping.get(t_old, t_old)  # unknown relations default to themselves
        new_relations.append({
            "type": t_new,
            "subject_id": r["subject_id"],
            "object_id": r["object_id"]
        })

    # 4) regenerate captions + QA
    new_captions = make_captions(objs, new_relations, max_caps=6)
    new_qa       = make_qa(objs, new_relations, max_qa=8)

    ann2 = dict(ann)
    ann2["objects"]   = objs
    ann2["relations"] = new_relations
    ann2["captions"]  = new_captions
    ann2["qa"]        = new_qa
    return ann2

# ------------ crop helper ------------

def random_crop_box(W, H, min_scale=0.9):
    cw = int(W * random.uniform(min_scale, 1.0))
    ch = int(H * random.uniform(min_scale, 1.0))
    x0 = random.randint(0, max(0, W - cw))
    y0 = random.randint(0, max(0, H - ch))
    return (x0, y0, x0+cw, y0+ch)

# ------------ per-image augmentation ------------

def augment_one(img_path, ann_path, out_img_dir, out_ann_dir):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    with open(ann_path) as f:
        ann = json.load(f)

    # you can tweak which transforms to use here
    transforms = [
        ("id",    lambda im: im),
        ("fh",    lambda im: ImageOps.mirror(im)),
        ("fv",    lambda im: ImageOps.flip(im)),
        ("r90cw", lambda im: im.transpose(Image.Transpose.ROTATE_270)),
        ("r90ccw",lambda im: im.transpose(Image.Transpose.ROTATE_90)),
        ("r180",  lambda im: im.transpose(Image.Transpose.ROTATE_180)),
    ]

    random.shuffle(transforms)
    chosen = transforms[:N_AUG_PER_IMAGE]  # sample subset if you want

    out_count = 0
    for tname, tf in chosen:
        im2 = tf(img)
        new_W, new_H = im2.size

        # optional random crop (50% chance)
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
        im_save.save(os.path.join(out_img_dir, out_img_name), "PNG")
        with open(os.path.join(out_ann_dir, out_ann_name), "w") as f:
            json.dump(ann_save, f, indent=2)
        out_count += 1

    return out_count

# ------------ augment one level ------------

def augment_level(level: int):
    base_dir = os.path.join(ROOT_IN,  f"level_{level}")
    img_dir  = os.path.join(base_dir, "images")
    ann_dir  = os.path.join(base_dir, "ann")

    out_dir   = os.path.join(ROOT_OUT, f"level_{level}")
    out_img   = os.path.join(out_dir, "images")
    out_ann   = os.path.join(out_dir, "ann")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    total = 0
    for i, fn in enumerate(files):
        img_path = os.path.join(img_dir, fn)
        ann_path = os.path.join(ann_dir, fn.replace(".png",".json"))
        if not os.path.exists(ann_path):
            print(f"[level {level}] Missing ann for {fn}, skipping")
            continue
        total += augment_one(img_path, ann_path, out_img, out_ann)
    print(f"[level {level}] Augmented samples written: {total} → {out_dir}/")

# ------------ main ------------

def main():
    random.seed(1234)
    for lvl in LEVELS:
        augment_level(lvl)
    print("All levels done.")

if __name__ == "__main__":
    main()
