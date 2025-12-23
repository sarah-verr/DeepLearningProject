# Define available colors
COLORS_RGB = {
    "red":    (220, 60, 60),
    "blue":   (70, 110, 240),
    "green":  (70, 170, 90),
    "yellow": (250, 230, 80),
    "purple": (150, 90, 200),
    "cyan":   (60, 200, 220),
    "orange": (250, 150, 60),
    "pink":   (250, 150, 200),
    "lime":   (180, 250, 100),
}

ALL_SHAPES = ["square", "circle", "triangle", "star"]

BG_CHOICES = {
    "w": (255, 255, 255),
    "b": (0, 0, 0),
}

# Distance thresholds for "near"/"far" in level 3+ (Manhattan on patch grid)
NEAR_GRID_DIST = 2
FAR_GRID_DIST  = 4

# Max captions / QA per image
MAX_CAPTIONS = 25
MAX_QA       = 50

# --- LEVEL DESIGN CONFIGURATION ---
# specific complexity settings for each level
LEVEL_CONFIG = {
    0: {"min_shapes": 2, "max_shapes": 2, "adv_lang": False},
    1: {"min_shapes": 2, "max_shapes": 2, "adv_lang": False},
    2: {"min_shapes": 2, "max_shapes": 2, "adv_lang": False},
    3: {"min_shapes": 3, "max_shapes": 3, "adv_lang": False},
    4: {"min_shapes": 4, "max_shapes": 4, "adv_lang": False},
    5: {"min_shapes": 5, "max_shapes": 5, "adv_lang": False},
    6: {"min_shapes": 6, "max_shapes": 6, "adv_lang": False},
}