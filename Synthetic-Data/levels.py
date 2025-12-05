# levels.py

import random
from typing import List, Tuple
from core_utils import (
    Obj, make_obj, grid_centers, manhattan, bboxes_overlap, iou,
    point_in_bbox, bbox_contains, dedup_relations, NEAR_GRID_DIST, FAR_GRID_DIST
)

# ---------- Scene Generators (Unified) ----------

def scene_grid_strict(img_size:int, patch:int, n_range:Tuple[int,int], shapes:List[str]) -> List[Obj]:
    """Level 0 style: strictly small objects, centered in grid cells."""
    n_objs = random.randint(*n_range)
    centers = grid_centers(img_size, patch)
    random.shuffle(centers)
    chosen = centers[:n_objs]
    objs = []
    for i,(_,_,cx,cy) in enumerate(chosen):
        shape = random.choice(shapes)
        size = int(patch - 4)
        objs.append(make_obj(i, shape, "red", (cx,cy), size))
    
    # Adjacency enforcement for level 1
    if len(objs) >= 2:
        gx0, gy0, _, _ = chosen[0]
        neighbors = [(gx0+1,gy0),(gx0-1,gy0),(gx0,gy0+1),(gx0,gy0-1)]
        random.shuffle(neighbors)
        grid_map = {(gx,gy):(cx,cy) for gx,gy,cx,cy in centers}
        count = 1
        for ng in neighbors:
            if count >= len(objs): break
            if ng in grid_map:
                cx,cy = grid_map[ng]
                o = objs[count]
                objs[count] = make_obj(o.id, o.shape, o.color, (cx,cy), o.size)
                count += 1
    return objs

def scene_grid(img_size:int, patch:int, n_range:Tuple[int,int], shapes:List[str]) -> List[Obj]:
    """Level 1 style: strictly small objects, centered in grid cells."""
    n_objs = random.randint(*n_range)
    centers = grid_centers(img_size, patch)
    random.shuffle(centers)
    chosen = centers[:n_objs]
    objs = []
    for i,(_,_,cx,cy) in enumerate(chosen):
        shape = random.choice(shapes)
        size = int(patch - 4)
        objs.append(make_obj(i, shape, "red", (cx,cy), size))
    
    # Adjacency enforcement for level 1
    if len(objs) >= 2:
        gx0, gy0, _, _ = chosen[0]
        n = random.randint(2, 4)
        neighbors = [(gx0+n,gy0),(gx0-n,gy0),(gx0,gy0+n),(gx0,gy0-n)]
        random.shuffle(neighbors)
        grid_map = {(gx,gy):(cx,cy) for gx,gy,cx,cy in centers}
        count = 1
        for ng in neighbors:
            if count >= len(objs): break
            if ng in grid_map:
                cx,cy = grid_map[ng]
                o = objs[count]
                objs[count] = make_obj(o.id, o.shape, o.color, (cx,cy), o.size)
                count += 1
    return objs

def scene_level2_no_overlap(img_size:int, patch:int, n_range:Tuple[int,int], shapes:List[str]) -> List[Obj]:
    """Level 2: Larger objects, but strictly NO overlaps."""
    n_objs = random.randint(*n_range)
    centers = grid_centers(img_size, patch)
    random.shuffle(centers)
    
    objs = []
    for _,_,cx,cy in centers:
        if len(objs) >= n_objs: break
        shape = random.choice(shapes)
        size = random.randint(int(1.2*patch), int(2.2*patch))
        candidate = make_obj(len(objs), shape, "red", (cx,cy), size)
        if any(bboxes_overlap(candidate.bbox, o.bbox) for o in objs): continue
        objs.append(candidate)
    return objs

def scene_grid_loose(img_size:int, patch:int, n_range:Tuple[int,int], shapes:List[str]) -> List[Obj]:
    """Level 3 style: Larger objects, can span patches. Some overlap allowed."""
    n_objs = random.randint(*n_range)
    centers = grid_centers(img_size, patch)
    random.shuffle(centers)
    
    objs = []
    for _,_,cx,cy in centers:
        if len(objs) >= n_objs: break
        shape = random.choice(shapes)
        size = random.randint(int(1.2*patch), int(2.2*patch))
        candidate = make_obj(len(objs), shape, "red", (cx,cy), size)
        overlaps = sum(1 for o in objs if bboxes_overlap(candidate.bbox, o.bbox))
        if overlaps > 1: continue
        objs.append(candidate)
        
    if len(objs) < n_objs:
        used_centers = {o.center for o in objs}
        remaining = [c for c in centers if (c[2],c[3]) not in used_centers]
        for i in range(n_objs - len(objs)):
            if not remaining: break
            gx, gy, cx, cy = remaining.pop()
            shape = random.choice(shapes)
            size = random.randint(int(1.2*patch), int(2.2*patch))
            objs.append(make_obj(len(objs), shape, "red", (cx,cy), size))

    return objs

def scene_nested(img_size:int, patch:int, n_range:Tuple[int,int], shapes:List[str]) -> List[Obj]:
    """Level 4+ style: Anchor + Satellites."""
    n_objs = random.randint(*n_range)
    objs = []
    
    # 1. Anchor
    anchor_size = random.randint(int(4.0 * patch), int(8.0 * patch))
    anchor_shape = random.choice(shapes)
    margin = anchor_size // 2 + 5
    if margin >= img_size - margin: margin = img_size // 2
    ax = random.randint(margin, img_size - margin)
    ay = random.randint(margin, img_size - margin)
    anchor = make_obj(0, anchor_shape, "red", (ax, ay), anchor_size)
    objs.append(anchor)
    
    # 2. Satellites
    n_satellites = random.randint(1, 2)
    n_satellites = min(n_satellites, n_objs - 1)
    
    for i in range(n_satellites):
        rel_type = random.choice(["inside", "overlapping"])
        sat_shape = random.choice(shapes)
        if rel_type == "inside":
            sat_size = random.randint(int(1.0*patch), int(anchor_size * 0.5))
        else:
            sat_size = random.randint(int(1.5*patch), int(anchor_size * 0.7))
            
        for _ in range(20):
            x0, y0, x1, y1 = anchor.bbox
            if rel_type == "inside":
                # Must fit FULLY inside
                min_x, max_x = x0 + sat_size//2 + 1, x1 - sat_size//2 - 1
                min_y, max_y = y0 + sat_size//2 + 1, y1 - sat_size//2 - 1
            else:
                min_x, max_x = x0 - sat_size//2, x1 + sat_size//2
                min_y, max_y = y0 - sat_size//2, y1 + sat_size//2
                
            if min_x >= max_x or min_y >= max_y: continue
            
            nx = random.randint(min_x, max_x)
            ny = random.randint(min_y, max_y)
            
            # Safety bounds check for overlapping
            sm = sat_size//2 + 2
            if rel_type == "overlapping":
                if nx < sm or nx > img_size - sm or ny < sm or ny > img_size - sm: continue
            
            sat = make_obj(len(objs), sat_shape, "red", (nx, ny), sat_size)
            
            if rel_type == "overlapping" and not bboxes_overlap(sat.bbox, anchor.bbox): continue
            if rel_type == "inside" and not bbox_contains(anchor.bbox, sat.bbox): continue

            collision = False
            for o in objs:
                if o.id == anchor.id: continue
                if bboxes_overlap(sat.bbox, o.bbox):
                    collision = True; break
            if collision: continue
            
            objs.append(sat)
            break
            
    # 3. Fillers
    attempts = 0
    while len(objs) < n_objs and attempts < 500:
        attempts += 1
        shape = random.choice(shapes)
        size = random.randint(int(1.5*patch), int(3.5*patch))
        margin = size // 2 + 2
        if margin >= img_size - margin: continue
        rx = random.randint(margin, img_size - margin)
        ry = random.randint(margin, img_size - margin)
        cand = make_obj(len(objs), shape, "red", (rx, ry), size)
        
        if any(bboxes_overlap(cand.bbox, o.bbox) for o in objs): continue
        objs.append(cand)
        
    return objs

# ---------- Relation Generators (Unified Signature) ----------

def _rel_spatial(objs, patch):
    """
    Updated Spatial Logic:
    1. Requires a buffer of 'patch' size to declare a relation.
    2. EXCLUDES relations if one object is INSIDE the other.
    """
    rels = []
    buffer = patch 
    
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i == j: continue
            a, b = objs[i], objs[j]
            
            # NEW: If contained, skip directional spatial relations.
            # This prevents "Circle inside Box is Left of Box".
            if bbox_contains(a.bbox, b.bbox) or bbox_contains(b.bbox, a.bbox):
                continue

            # Strict margin check
            if a.center[0] <= b.center[0] - buffer: 
                rels.append({"type":"left_of", "subject_id":a.id, "object_id":b.id})
            if a.center[0] >= b.center[0] + buffer: 
                rels.append({"type":"right_of", "subject_id":a.id, "object_id":b.id})
            
            if a.center[1] <= b.center[1] - buffer: 
                rels.append({"type":"above", "subject_id":a.id, "object_id":b.id})
            if a.center[1] >= b.center[1] + buffer: 
                rels.append({"type":"below", "subject_id":a.id, "object_id":b.id})
    return rels

def _rel_contact(objs):
    rels = []
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i == j: continue
            a, b = objs[i], objs[j]
            if bboxes_overlap(a.bbox, b.bbox):
                rels.append({"type":"touching","subject_id":a.id,"object_id":b.id})
                if iou(a.bbox,b.bbox) > 0.05:
                    rels.append({"type":"overlapping","subject_id":a.id,"object_id":b.id})
    return rels

def rel_level1(objs: List[Obj], patch: int) -> List[dict]:
    # Level 1 strictly uses grid logic, so we keep the old strict check logic here
    # or simply map it to the new spatial logic. 
    # Since L1 objects are small and centered, the buffer logic works fine there too.
    return dedup_relations(_rel_spatial(objs, patch))

def rel_level2(objs: List[Obj], patch: int) -> List[dict]:
    return dedup_relations(_rel_spatial(objs, patch))

def rel_level3(objs: List[Obj], patch: int) -> List[dict]:
    return dedup_relations(_rel_spatial(objs, patch) + _rel_contact(objs))

def rel_level4(objs: List[Obj], patch: int) -> List[dict]:
    rels = _rel_spatial(objs, patch) + _rel_contact(objs)
    
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i == j: continue
            a, b = objs[i], objs[j]
            
            # Strict fully-inside check
            if bbox_contains(b.bbox, a.bbox):
                rels.append({"type":"inside","subject_id":a.id,"object_id":b.id})
            
            gdist = manhattan(a.center, b.center, patch)
            if gdist <= NEAR_GRID_DIST:
                rels.append({"type":"near","subject_id":a.id,"object_id":b.id})
                if gdist == 1:
                    rels.append({"type":"next_to","subject_id":a.id,"object_id":b.id})
                    rels.append({"type":"beside","subject_id":a.id,"object_id":b.id})
            if gdist >= FAR_GRID_DIST:
                rels.append({"type":"far","subject_id":a.id,"object_id":b.id})
    return dedup_relations(rels)

# ---------- Mapping Dictionary ----------
LEVEL_MAPPING = {
    0: {"scene": scene_grid_strict,      "rel": rel_level1},
    1: {"scene": scene_grid,             "rel": rel_level1},
    2: {"scene": scene_level2_no_overlap,"rel": rel_level1}, 
    3: {"scene": scene_level2_no_overlap,"rel": rel_level2}, 
    4: {"scene": scene_grid_loose,       "rel": rel_level3},
    5: {"scene": scene_nested,           "rel": rel_level4},
    6: {"scene": scene_nested,           "rel": rel_level4},
}

def get_level_fns(level_id):
    if level_id not in LEVEL_MAPPING:
        raise ValueError(f"Level {level_id} not found in mapping.")
    return LEVEL_MAPPING[level_id]["scene"], LEVEL_MAPPING[level_id]["rel"]