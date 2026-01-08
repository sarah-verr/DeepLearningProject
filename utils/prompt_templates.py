"""Shared prompt templates."""

from typing import Any


Conversation = list[dict[str, Any]]


def build_visual_yesno_prompt(question: str,) -> Conversation:
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
                        'relational reasoning question with either "yes" or "no". Explain your reasoning. \n'
                        f"QUESTION: {q}\n"
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


def get_caption_for_qa(annotation_data: dict, qa_item: dict) -> str | None:
    """Fetch the caption text associated with this QA item, if available."""
    cap_id = qa_item.get("caption_id")
    if cap_id is None:
        # Fallback to first caption
        captions = annotation_data.get("captions", []) or []
        if captions and isinstance(captions[0], str):
            return captions[0].strip()
        return None

    captions_meta = annotation_data.get("captions_meta")
    if not isinstance(captions_meta, list) or not captions_meta:
        return None

    try:
        cap_id_int = int(cap_id)
    except Exception:
        return None

    # Direct indexing
    if 0 <= cap_id_int < len(captions_meta):
        c = captions_meta[cap_id_int]
        if isinstance(c, dict):
            cap_text = c.get("caption")
            return cap_text.strip() if isinstance(cap_text, str) and cap_text.strip() else None

    # Fallback scan by explicit id
    for c in captions_meta:
        if not isinstance(c, dict):
            continue
        if c.get("id") == cap_id_int:
            cap_text = c.get("caption")
            return cap_text.strip() if isinstance(cap_text, str) and cap_text.strip() else None

    return None

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