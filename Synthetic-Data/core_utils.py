# core_utils.py

import os, json, math, random, re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
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

def intersecting_patch_indices(bbox: Tuple[int, int, int, int], img_size: int, patch: int) -> List[int]:
    """Return flat patch indices (row-major) whose patch area intersects bbox.

    Coordinates are assumed to be in the same pixel space as the rendered image.
    """
    x1, y1, x2, y2 = bbox
    grid_dim = img_size // patch
    out: List[int] = []
    for r in range(grid_dim):
        for c in range(grid_dim):
            p_x1 = c * patch
            p_y1 = r * patch
            p_x2 = p_x1 + patch
            p_y2 = p_y1 + patch

            inter_x1 = max(x1, p_x1)
            inter_y1 = max(y1, p_y1)
            inter_x2 = min(x2, p_x2)
            inter_y2 = min(y2, p_y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                out.append(r * grid_dim + c)
    return out

# ---------------------- Relations & Language ----------------------
PRIMARY  = {"left_of","right_of","above","below"}
# NOTE: We include explicit negation types for symmetric relations so we can generate
# strong, controlled NO questions without relying on random fallbacks.
ADVANCED = {
    "inside",
    "encapsulates",
    "touching",
    "not_touching",
    "overlapping",
    "not_overlapping",
}  # removed "near","far" for now as it is quite ambiguous and subjective, also removed "next_to","beside" due to its subjectivity
CONTACT  = set()  # kept for backward-compatibility with older code paths

PHRASES = {
    "left_of": ["left of", "to the left of"], 
    "right_of": ["right of", "to the right of"],
    "above": ["above"], 
    "below": ["below"],
    "touching": ["touching"],
    "not_touching": ["not touching"],
    "overlapping": ["overlapping with"],
    "not_overlapping": ["not overlapping with"],
    
    # Updated "Inside" phrases
    "inside": ["fully inside", "contained within", "enclosed by"],
    
    # New "Encapsulates" phrases (used for Inverse of Inside)
    "encapsulates": ["encapsulating", "surrounding", "containing"],

    "next_to": ["next to", "beside"], 
    "beside": ["beside", "next to"],
    "near": ["near", "close to"], 
    "far": ["far from", "distant from"],
}

INV = {
    "left_of":"right_of", "right_of":"left_of",
    "above":"below", "below":"above",
    "touching":"touching",
    "not_touching":"not_touching",
    "overlapping":"overlapping",
    "not_overlapping":"not_overlapping",
    
    # Asymmetric Inverse: If A is inside B, B encapsulates A
    "inside": "encapsulates",
    "encapsulates": "inside",
    
    "next_to":"next_to", "beside":"beside",
    "near":"near", "far":"far",
}

NEG_PRIMARY = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
}

def rel_group(rel_type: str) -> str:
    """Return the relation family name for a relation type."""
    if rel_type in PRIMARY:
        return "PRIMARY"
    if rel_type in ADVANCED:
        return "ADVANCED"
    return "UNKNOWN"

def rel_phrase(t: str) -> str:
    return random.choice(PHRASES[t])

def caption_from_rel(a: Obj, b: Obj, rel_type: str) -> str:
    return f"The {a.color} {a.shape} is {rel_phrase(rel_type)} the {b.color} {b.shape}."


def caption_meta_from_rel(caption_id: int, a: Obj, b: Obj, rel_type: str) -> Dict[str, Any]:
    phrase = rel_phrase(rel_type)
    cap = f"The {a.color} {a.shape} is {phrase} the {b.color} {b.shape}."
    return {
        "id": int(caption_id),
        "caption": cap,
        "subject_id": a.id,
        "object_id": b.id,
        "rel_type": rel_type,
        "rel_group": rel_group(rel_type),
        "rel_phrase": phrase,
        # Filled after final QA selection + id assignment
        "entailed_qa_ids": [],
        "contradicted_qa_ids": [],
    }

def qa_from_rel(a: Obj, b: Obj, rel_type: str) -> List[Dict[str, str]]:
    # Forward Question (A ?rel? B) — choose and store the exact phrase
    phrase_fwd = rel_phrase(rel_type)
    qf = f"Is the {a.color} {a.shape} {phrase_fwd} the {b.color} {b.shape}?"

    # Inverse Question (B ?inv_rel? A) — choose and store the exact phrase
    inv_type = INV[rel_type]
    phrase_inv = rel_phrase(inv_type)
    qi = f"Is the {b.color} {b.shape} {phrase_inv} the {a.color} {a.shape}?"

    return [
        {
            "question": qf,
            "answer": "yes",
            "subject_id": a.id,
            "object_id": b.id,
            "rel_type": rel_type,
            "rel_group": rel_group(rel_type),
            "rel_phrase": phrase_fwd,  # NEW
        },
        {
            "question": qi,
            "answer": "yes",
            "subject_id": b.id,
            "object_id": a.id,
            "rel_type": inv_type,
            "rel_group": rel_group(inv_type),
            "rel_phrase": phrase_inv,  # NEW
        },
    ]
    

def _qa_item(a: Obj, b: Obj, rel_type: str, answer: str, *, caption_id: int | None = None) -> Dict[str, Any]:
    phrase = rel_phrase(rel_type)
    q = f"Is the {a.color} {a.shape} {phrase} the {b.color} {b.shape}?"
    item: Dict[str, Any] = {
        "question": q,
        "answer": answer,
        "subject_id": a.id,
        "object_id": b.id,
        "rel_type": rel_type,
        "rel_group": rel_group(rel_type),
        "rel_phrase": phrase,
    }
    if caption_id is not None:
        item["caption_id"] = int(caption_id)
    return item


def _paired_yes_no_for_rel(
    a: Obj,
    b: Obj,
    rel_type: str,
    *,
    allowed_types: set,
    true_types_for_pair: set,
    caption_id: int | None = None,
) -> List[Dict[str, str]]:
    """Return [YES, NO] for a true relation instance.

    Rules:
      - PRIMARY: NO is NEG_PRIMARY[rel_type] on the same ordered pair (a,b).
      - ADVANCED: NO is INV[rel_type] on the same ordered pair (a,b) (no swap).
        This works well for asymmetric pairs like inside/encapsulates.
      - Fallback (e.g., touching/overlapping where INV == self): pick any allowed
        relation type that is not true for this ordered pair.
    """
    out: List[Dict[str, Any]] = [_qa_item(a, b, rel_type, "yes", caption_id=caption_id)]

    no_type = None

    if rel_type in PRIMARY:
        cand = NEG_PRIMARY.get(rel_type)
        if cand and cand in allowed_types and cand not in true_types_for_pair:
            no_type = cand

    # Symmetric advanced relations: prefer explicit negation types.
    if no_type is None and rel_type == "touching":
        cand = "not_touching"
        if cand in allowed_types and cand not in true_types_for_pair:
            no_type = cand

    if no_type is None and rel_type == "overlapping":
        cand = "not_overlapping"
        if cand in allowed_types and cand not in true_types_for_pair:
            no_type = cand

    if no_type is None and rel_type in ADVANCED:
        inv = INV.get(rel_type)
        if inv and inv in allowed_types and inv not in true_types_for_pair and inv != rel_type:
            no_type = inv

    if no_type is None:
        # Prefer staying within the same family when possible.
        preferred_pool = list(ADVANCED & allowed_types) if rel_type in ADVANCED else list(PRIMARY & allowed_types)
        preferred = [t for t in preferred_pool if t not in true_types_for_pair]
        candidates = preferred if preferred else [t for t in allowed_types if t not in true_types_for_pair]
        if candidates:
            no_type = random.choice(candidates)

    if no_type is not None:
        out.append(_qa_item(a, b, no_type, "no", caption_id=caption_id))

    return out


def _assign_qa_ids_and_link_captions(
    captions_meta: List[Dict[str, Any]],
    qa: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assign sequential QA ids and populate caption->qa links.

    We do this AFTER dedup/balancing so caption links remain valid.
    """
    for c in captions_meta:
        c["entailed_qa_ids"] = []
        c["contradicted_qa_ids"] = []

    cap_index = {int(c["id"]): c for c in captions_meta if "id" in c}

    for qid, item in enumerate(qa):
        item["id"] = int(qid)
        cap_id = item.get("caption_id")
        if cap_id is None:
            continue
        cap_id_int = int(cap_id)
        c = cap_index.get(cap_id_int)
        if c is None:
            continue
        if item.get("answer") == "yes":
            c["entailed_qa_ids"].append(int(qid))
        elif item.get("answer") == "no":
            c["contradicted_qa_ids"].append(int(qid))

    return captions_meta, qa


def _balance_yes_no(qa: List[Dict[str, str]], max_qa: int) -> List[Dict[str, str]]:
    """Return an exactly balanced list: N yes + N no, interleaved, up to max_qa."""
    yes = [x for x in qa if x.get("answer") == "yes"]
    no = [x for x in qa if x.get("answer") == "no"]
    n = min(len(yes), len(no), max_qa // 2)
    out: List[Dict[str, str]] = []
    for i in range(n):
        out.append(yes[i])
        out.append(no[i])
    return out[: 2 * n]

def sample_language(
    objects: List[Obj],
    relations: List[Dict],
    allow_advanced: bool,
    max_caps: int = MAX_CAPTIONS,
    max_qa: int = MAX_QA,
    dedup_qa: bool = True,
    include_contact: bool = True,
    question_group: str | None = None,
    question_groups: List[str] | None = None,
):
    by_id = {o.id: o for o in objects}

    # Filter allowed relation types
    # If question_groups is provided, it takes precedence and can include multiple families.
    if question_groups is not None:
        selected = [g.lower().strip() for g in question_groups if isinstance(g, str) and g.strip()]
        if not selected:
            return [], []
        allowed_types = set()
        for g in selected:
            if g == "primary":
                allowed_types |= PRIMARY
            elif g == "advanced":
                allowed_types |= ADVANCED
            else:
                # Unknown group token => treat as invalid and generate no questions
                return [], []
    elif question_group is not None:
        qg = question_group.lower().strip()
        if qg == "primary":
            allowed_types = set(PRIMARY)
        elif qg == "advanced":
            allowed_types = set(ADVANCED)
        else:
            # Unknown group => no questions.
            return [], []
    else:
        # Backward-compatible default behavior (not used by current CLI):
        allowed_types = set(PRIMARY)
        if include_contact:
            allowed_types |= CONTACT
        if allow_advanced:
            allowed_types |= ADVANCED

    valid_rels = [r for r in relations if r["type"] in allowed_types]
    random.shuffle(valid_rels)

    captions: List[str] = []
    captions_meta: List[Dict[str, Any]] = []
    qa: List[Dict[str, Any]] = []
    # Build truth map once (used to ensure our paired NO is actually false).
    pair_to_types: Dict[Tuple[int, int], set] = {}
    for r in relations:
        k = (r["subject_id"], r["object_id"])
        pair_to_types.setdefault(k, set()).add(r["type"])

    # Generate paired YES/NO questions. Generate extra to survive dedup.
    budget = max_qa * 3
    for r in valid_rels:
        a, b = by_id[r["subject_id"]], by_id[r["object_id"]]
        caption_id = None
        if len(captions) < max_caps:
            caption_id = len(captions_meta)
            cap_meta = caption_meta_from_rel(caption_id, a, b, r["type"])
            captions_meta.append(cap_meta)
            captions.append(cap_meta["caption"])

        if len(qa) >= budget:
            break

        true_types = pair_to_types.get((a.id, b.id), set())
        qa.extend(
            _paired_yes_no_for_rel(
                a,
                b,
                r["type"],
                allowed_types=allowed_types,
                true_types_for_pair=true_types,
                caption_id=caption_id,
            )
        )

    if dedup_qa:
        qa = _dedup_qa_list(qa)

    qa = _balance_yes_no(qa, max_qa=max_qa)
    captions_meta, qa = _assign_qa_ids_and_link_captions(captions_meta, qa)
    return captions, qa, captions_meta
# ---------------------- Logic Helpers ----------------------

def _norm_question(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation
    return s

def _dedup_qa_list(qa: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for item in qa:
        q = item.get("question", "")
        key = _norm_question(q)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

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
    # NEW:
    dedup_qa: bool = True,
    include_contact: bool = True,
    question_group: str | None = None,
    question_groups: List[str] | None = None,
):
    # Create image + draw objects
    bg = BG_CHOICES.get(bg_key, (255, 255, 255))
    img = Image.new("RGB", (img_size, img_size), color=bg)
    draw = ImageDraw.Draw(img)
    for o in objs:
        draw_shape(draw, o)
    relations = rel_fn(objs, patch)

    caps, qa, caps_meta = sample_language(
        objs,
        relations,
        allow_advanced_language,
        dedup_qa=dedup_qa,
        include_contact=include_contact,
        question_group=question_group,
        question_groups=question_groups,
    )

    ann = {
        "background": bg_key,
        "objects": [
            {
                **asdict(o),
                "patch_indices": intersecting_patch_indices(o.bbox, img_size=img_size, patch=patch),
            }
            for o in objs
        ],
        "relations": relations,
        "captions": caps,
        "captions_meta": caps_meta,
        "qa": qa,
        "meta": {
            "img_size": img_size,
            "patch": patch,
            "grid_dim": img_size // patch,
        },
    }

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    img.save(out_img, "PNG")
    with open(out_json, "w") as f:
        json.dump(ann, f, indent=2)