"""Shared prompt templates.

This repo uses the same prompt strings across:
- main.py (visual inference + attention plots)
- compute_accuracy.py (batch evaluation)
- text_only_phrase_attention.py (text-only attention analysis)

Centralizing them here makes it easy to change prompt wording without the
scripts drifting apart.
"""


def build_visual_yesno_prompt(question: str) -> str:
    """Prompt for visual (image + question) yes/no answering.

    Keep the question on its own line so token-subsequence matching against
    `question` continues to work.
    """
    q = (question or "").strip()
    return (
        "USER: <image>\n"
        "USER: Use the spatial layout in the image to answer the following relational reasoning question with either \"yes\" or \"no\".\n"
        f"{q}\n"
        "ASSISTANT:"
    )


def build_caption_yesno_prompt(caption: str, question: str) -> str:
    """Prompt for text-only caption evidence yes/no answering."""
    cap = (caption or "").strip()
    q = (question or "").strip()
    return (
        "You are given a caption describing a synthetic scene.\n"
        "Use only the caption as evidence.\n"
        "Answer the question with only \"yes\" or \"no\".\n\n"
        f"Caption: {cap}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )


def build_scene_yesno_prompt(scene_text: str, question: str) -> str:
    """Prompt for text-only scene-description evidence yes/no answering."""
    scene = (scene_text or "").strip()
    q = (question or "").strip()
    return (
        "You are given a synthetic scene description.\n"
        "Answer the question with only \"yes\" or \"no\".\n\n"
        f"Scene: {scene}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )
