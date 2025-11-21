# core_utils.py

import os, json, math, random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw

from core_config import COLORS_RGB, BG_CHOICES, NEAR_GRID_DIST, FAR_GRID_DIST, MAX_CAPTIONS, MAX_QA

# ---------------------- Data Structures ----------------------
@dataclass
class Obj:
    id: int
    shape: str
    color: str
    center: Tuple[int, int]
    size: int
    bbox: Tuple[int, int, int, int]

def make_obj(obj_id: int, shape: str, color: str,
             center: Tuple[int,int], size: int) -> Obj:
    cx, cy = center
    half = size // 2
    return Obj(
        id=obj_id,
        shape=shape,
        color=color,
        center=center,
        size=size,
        bbox=(int(cx-half), int(cy-half), int(cx+half), int(cy+half)),
    )

# ---------------------- Drawing ----------------------
def draw_shape(draw: ImageDraw.ImageDraw, obj: Obj):
    x0, y0, x1, y1 = obj.bbox
    col = COLORS_RGB[obj.color]
    if obj.shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=col)
    elif obj.shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=col)
    elif obj.shape == "triangle":
        cx, cy = obj.center
        s = obj.size
        h = s * math.sqrt(3) / 2
        pts = [
            (cx, cy - 2*h/3),
            (cx - s/2, cy + h/3),
            (cx + s/2, cy + h/3),
        ]
        draw.polygon(pts, fill=col)
    elif obj.shape == "star":
        draw_star(draw, obj.center, obj.size // 2, col)

def draw_star(draw: ImageDraw.ImageDraw, center, radius, fill):
    cx, cy = center
    pts = []
    for i in range(10):
        ang = math.pi/2 + i * math.pi/5
        r = radius if i % 2 == 0 else radius / 2.5
        pts.append((cx + r*math.cos(ang), cy - r*math.sin(ang)))
    draw.polygon(pts, fill=fill)

# ---------------------- Geometry ----------------------
def grid_centers(img_size: int, patch: int) -> List[Tuple[int,int,int,int]]:
    """
    Returns available grid centers with safety margin.
    """
    gw = img_size // patch
    centers = []
    margin = 1 
    for gy in range(margin, gw - margin):
        for gx in range(margin, gw - margin):
            cx = gx*patch + patch//2
            cy = gy*patch + patch//2
            centers.append((gx, gy, cx, cy))
    return centers

def manhattan(a_xy: Tuple[int,int], b_xy: Tuple[int,int], patch: int) -> int:
    ax, ay = a_xy
    bx, by = b_xy
    gx_a, gy_a = ax // patch, ay // patch
    gx_b, gy_b = bx // patch, by // patch
    return abs(gx_a - gx_b) + abs(gy_a - gy_b)

def bboxes_overlap(b1, b2):
    x0,y0,x1,y1 = b1
    a0,b0,a1,b1 = b2
    return not (x1 < a0 or a1 < x0 or y1 < b0 or b1 < y0)

def bbox_contains(outer, inner):
    """Returns True if inner is FULLY inside outer."""
    Ox0, Oy0, Ox1, Oy1 = outer
    Ix0, Iy0, Ix1, Iy1 = inner
    return (Ox0 <= Ix0) and (Oy0 <= Iy0) and (Ox1 >= Ix1) and (Oy1 >= Iy1)

def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(area1+area2-inter+1e-9)

def point_in_bbox(pt, bbox):
    x,y = pt
    x0,y0,x1,y1 = bbox
    return (x0 <= x <= x1) and (y0 <= y <= y1)

# ---------------------- Relations & Language ----------------------
PRIMARY  = {"left_of","right_of","above","below"}
CONTACT  = {"touching","overlapping"}
ADVANCED = {"inside","near","far","next_to","beside"}

PHRASES = {
    "left_of": ["left of", "to the left of"], 
    "right_of": ["right of", "to the right of"],
    "above": ["above"], 
    "below": ["below"],
    "touching": ["touching"], 
    "overlapping": ["overlapping with", "partially covering"],
    
    # Updated "Inside" phrases
    "inside": ["fully inside", "contained within", "enclosed by"],
    
    # New "Encapsulates" phrases (used for Inverse of Inside)
    "encapsulates": ["encapsulating", "encircling", "surrounding", "containing"],

    "next_to": ["next to", "beside"], 
    "beside": ["beside", "next to"],
    "near": ["near", "close to"], 
    "far": ["far from", "distant from"],
}

INV = {
    "left_of":"right_of", "right_of":"left_of",
    "above":"below", "below":"above",
    "touching":"touching", "overlapping":"overlapping",
    
    # Asymmetric Inverse: If A is inside B, B encapsulates A
    "inside": "encapsulates",
    "encapsulates": "inside",
    
    "next_to":"next_to", "beside":"beside",
    "near":"near", "far":"far",
}

def rel_phrase(t: str) -> str:
    return random.choice(PHRASES[t])

def caption_from_rel(a: Obj, b: Obj, rel_type: str) -> str:
    return f"The {a.color} {a.shape} is {rel_phrase(rel_type)} the {b.color} {b.shape}."

def qa_from_rel(a: Obj, b: Obj, rel_type: str) -> List[Dict[str,str]]:
    # Forward Question
    qf = f"Is the {a.color} {a.shape} {rel_phrase(rel_type)} the {b.color} {b.shape}?"
    
    # Inverse Question
    inv_type = INV[rel_type]
    qi = f"Is the {b.color} {b.shape} {rel_phrase(inv_type)} the {a.color} {a.shape}?"
    
    return [{"question": qf, "answer":"yes"}, {"question": qi, "answer":"yes"}]

def sample_language(objects: List[Obj], relations: List[Dict], allow_advanced: bool, max_caps: int=MAX_CAPTIONS, max_qa: int=MAX_QA):
    by_id = {o.id: o for o in objects}
    
    # Filter allowed relation types
    allowed_types = PRIMARY.union(CONTACT)
    if allow_advanced:
        allowed_types = allowed_types.union(ADVANCED)
        
    valid_rels = [r for r in relations if r["type"] in allowed_types]
    random.shuffle(valid_rels)

    captions, qa = [], []
    target_yes = max_qa // 2
    
    # Generate captions and YES questions
    for r in valid_rels:
        a, b = by_id[r["subject_id"]], by_id[r["object_id"]]
        if len(captions) < max_caps:
            captions.append(caption_from_rel(a, b, r["type"]))
        if len(qa) < target_yes:
            qa.extend(qa_from_rel(a, b, r["type"]))

    qa = qa[:target_yes]

    # Generate NO questions
    pair_to_types = {}
    for r in relations: # Use all relations to check truth
        k = (r["subject_id"], r["object_id"])
        if k not in pair_to_types: pair_to_types[k] = set()
        pair_to_types[k].add(r["type"])

    all_types = list(allowed_types)
    # Remove 'encapsulates' from sampling pool if it exists, as it's only an inverse
    if "encapsulates" in all_types: all_types.remove("encapsulates")

    import itertools
    pairs = [(a.id, b.id) for a, b in itertools.product(objects, objects) if a.id != b.id]
    
    attempts = 0
    while len(qa) < max_qa and attempts < 500 and pairs:
        attempts += 1
        sid, oid = random.choice(pairs)
        true_types = pair_to_types.get((sid, oid), set())
        
        # Pick a type that is FALSE for this pair
        candidates = [t for t in all_types if t not in true_types]
        
        if candidates:
            rtype = random.choice(candidates)
            a, b = by_id[sid], by_id[oid]
            q = f"Is the {a.color} {a.shape} {rel_phrase(rtype)} the {b.color} {b.shape}?"
            qa.append({"question": q, "answer": "no"})

    return captions, qa

# ---------------------- Logic Helpers ----------------------

def assign_colors(objs, allowed_colors):
    if not objs: return
    if len(objs) > len(allowed_colors):
        raise ValueError(f"Error: {len(objs)} objects vs {len(allowed_colors)} colors.")
    chosen = random.sample(allowed_colors, len(objs))
    for o, c in zip(objs, chosen):
        o.color = c

def dedup_relations(rels: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in rels:
        k = (r["type"], r["subject_id"], r["object_id"])
        if k not in seen:
            seen.add(k); out.append(r)
    return out

# ---------------------- Render + Save ----------------------
def render_and_save(
    objs: List[Obj],
    bg_key: str,
    img_size: int,
    out_img: str,
    out_json: str,
    rel_fn,
    patch: int,
    allow_advanced_language: bool,
):
    img = Image.new("RGB", (img_size, img_size), BG_CHOICES[bg_key])
    draw = ImageDraw.Draw(img)
    
    # Draw Largest first
    for o in sorted(objs, key=lambda x: x.size, reverse=True):
        draw_shape(draw, o)

    # Unified call
    relations = rel_fn(objs, patch)
    
    caps, qa = sample_language(objs, relations, allow_advanced_language)

    ann = {
        "background": bg_key,
        "objects": [asdict(o) for o in objs],
        "relations": relations,
        "captions": caps,
        "qa": qa,
    }

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    img.save(out_img, "PNG")
    with open(out_json, "w") as f:
        json.dump(ann, f, indent=2)