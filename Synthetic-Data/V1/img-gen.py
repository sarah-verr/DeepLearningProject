import os, json, math, random
from dataclasses import dataclass, asdict
from typing import List, Tuple
from PIL import Image, ImageDraw

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
OUT_DIR = "base_scenes"
N_SCENES = 20                # number of base geometric arrangements
IMG_SIZE = 256
BG_COLORS = {"white": (255, 255, 255), "black": (0, 0, 0)}
# , "grey": (128, 128, 128)
COLOR_COMBOS = [
    ["red", "blue", "green","yellow"],
    ["pink", "purple", "cyan","orange"],
    # ["orange", "pink", "blue","red"],
]
COLORS_RGB = {
    "red": (220, 60, 60),
    "blue": (70, 110, 240),
    "green": (70, 170, 90),
    "yellow": (250, 230, 80),
    "purple": (150, 90, 200),
    "cyan": (60, 200, 220),
    "orange": (250, 150, 60),
    "pink": (250, 150, 200),
}

SHAPES = ["circle", "square", "triangle", "star"]
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "ann_base"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "ann"), exist_ok=True)

@dataclass
class Obj:
    id: int
    shape: str
    color: str
    center: Tuple[int, int]
    size: int
    bbox: Tuple[int, int, int, int]

# -------------------------------------------------------
# SHAPE DRAWING HELPERS
# -------------------------------------------------------
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
        pts = [(cx, cy - 2*h/3), (cx - s/2, cy + h/3), (cx + s/2, cy + h/3)]
        draw.polygon(pts, fill=col)
    elif obj.shape == "star":
        draw_star(draw, obj.center, obj.size // 2, col)

def draw_star(draw: ImageDraw.ImageDraw, center, radius, fill):
    cx, cy = center
    points = []
    for i in range(10):
        angle = math.pi/2 + i * math.pi/5
        r = radius if i % 2 == 0 else radius / 2.5
        x = cx + r * math.cos(angle)
        y = cy - r * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=fill)

def get_shape_bbox(shape: str, center: Tuple[int, int], size: int) -> Tuple[int, int, int, int]:
    """Calculates the tightest bounding box for a given shape."""
    cx, cy = center
    s = size
    if shape == "circle" or shape == "square":
        half = s // 2
        return (cx - half, cy - half, cx + half, cy + half)
    
    elif shape == "triangle":
        # Vertices calculation from your draw_shape function
        h = s * math.sqrt(3) / 2
        pts = [(cx, cy - 2*h/3), (cx - s/2, cy + h/3), (cx + s/2, cy + h/3)]
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        return (int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))

    elif shape == "star":
        # Vertex calculation from your draw_star function
        radius = s // 2 # Based on draw_shape's call
        points = []
        for i in range(10):
            angle = math.pi/2 + i * math.pi/5
            r = radius if i % 2 == 0 else radius / 2.5
            x = cx + r * math.cos(angle)
            y = cy - r * math.sin(angle)
            points.append((x, y))
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return (int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))
    
    # Fallback just in case
    half = s // 2
    return (cx - half, cy - half, cx + half, cy + half)


def make_obj(obj_id: int, shape: str, color: str, center: Tuple[int,int], size: int) -> Obj:
    cx, cy = center
    # Use the new helper to get the correct bbox
    bbox = get_shape_bbox(shape, center, size)
    return Obj(obj_id, shape, color, center, size, bbox)


# -------------------------------------------------------
# SCENE GENERATION
# -------------------------------------------------------
def generate_scene(scene_id: int):
    # Random layout of 2–5 shapes
    n_shapes = random.randint(2, 4)
    chosen_shapes = random.sample(SHAPES, n_shapes)
    coords = []
    objects = []
    for i, shape in enumerate(chosen_shapes):
        size = random.randint(40, 70)
        valid = False
        while not valid:
            cx = random.randint(size, IMG_SIZE - size)
            cy = random.randint(size, IMG_SIZE - size)
            valid = all(math.dist((cx, cy), c) > 70 for c in coords)
        coords.append((cx, cy))
        objects.append(make_obj(i, shape, "red", (cx, cy), size))  # placeholder color, updated later
    return objects

def render_scene(objects: List[Obj], bg_color: Tuple[int,int,int], color_combo: List[str], scene_id: int, bg_name: str, combo_idx: int):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color)
    draw = ImageDraw.Draw(img)
    # assign colors deterministically from combo
    for i, obj in enumerate(objects):
        obj.color = color_combo[i % len(color_combo)]
        draw_shape(draw, obj)

    ann = {
        "scene_id": scene_id,
        "background": bg_name,
        "color_combo": color_combo,
        "objects": [asdict(o) for o in objects],
    }
    bg_short = "w"
    
    if bg_name == "white": bg_short = "w"
    elif bg_name == "black": bg_short =  "b"
    else: bg_short = "g"

    img_name = f"{scene_id:04d}_{bg_short}{combo_idx}.png" #shorter name
    # img_name = f"{scene_id:04d}_{bg_name}_combo{combo_idx}.png" #longer name
    ann_name = img_name.replace(".png", ".json")
    img.save(os.path.join(OUT_DIR, "images", img_name))
    with open(os.path.join(OUT_DIR, "ann_base", ann_name), "w") as f:
        json.dump(ann, f, indent=2)

def main():
    for sid in range(N_SCENES):
        objs = generate_scene(sid)
        for bg_name, bg_col in BG_COLORS.items():
            for combo_idx, combo in enumerate(COLOR_COMBOS, 1):
                render_scene(objs, bg_col, combo, sid, bg_name, combo_idx)
        # for bg_name, bg_col in BG_COLORS.items():
        #     for combo_idx, combo in enumerate(COLOR_COMBOS, 1):
        #         render_scene(objs, bg_col, combo, sid, bg_name, combo_idx)
    print(f"Generated {N_SCENES * len(BG_COLORS) * len(COLOR_COMBOS)} images in '{OUT_DIR}/images'")

if __name__ == "__main__":
    main()
