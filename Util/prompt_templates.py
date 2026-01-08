"""Shared prompt templates."""

from typing import Any


Conversation = list[dict[str, Any]]


def build_visual_yesno_prompt(question: str, ) -> Conversation:
    """Return a chat-style conversation for visual yes/no answering."""
    q = (question or "").strip()
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        'Use the spatial layout in the image to answer the following '
                        'relational reasoning question with either "yes" or "no".\n'
                        f"QUESTION: {q}"
                    ),
                },
                {"type": "image"},
            ],
        },
    ]


def build_caption_yesno_prompt(caption: str, question: str) -> Conversation:
    """Return a chat-style conversation for caption-based yes/no answering."""
    cap = (caption or "").strip()
    q = (question or "").strip()
    text = (
        "You are given a caption describing a synthetic scene.\n"
        "Use only the caption as evidence.\n"
        "Answer the question with only \"yes\" or \"no\".\n\n"
        f"Caption: {cap}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    ]


def scene_to_text(annotation_data: dict) -> str:
    """Convert annotation data to scene description text."""
    meta = annotation_data.get("meta", {}) or {}
    patch = int(meta.get("patch", 14))

    lines = []
    for obj in annotation_data.get("objects", []):
        oid = obj.get("id", None)
        color = obj.get("color", "unknown")
        shape = obj.get("shape", "unknown")

        center = obj.get("center", None)
        if isinstance(center, list) and len(center) == 2:
            cx, cy = center
            gx, gy = int(cx) // patch, int(cy) // patch
            pos = f"grid ({gx}, {gy})"
        else:
            pos = "grid (unknown, unknown)"

        lines.append(f"Object {oid}: {color} {shape} at {pos}")

    return "\n".join(lines)
    
def build_scene_yesno_prompt(scene_text: str, question: str) -> Conversation:
    """Return a chat-style conversation for scene-description yes/no answering."""
    scene = (scene_text or "").strip()
    q = (question or "").strip()
    text = (
        "You are given a synthetic scene description.\n"
        "Answer the question with only \"yes\" or \"no\".\n\n"
        f"Scene: {scene}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    ]
