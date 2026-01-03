import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_ID = "llava-hf/llava-1.6-7b-hf"
DEFAULT_BASE_DATA_PATH = "/Users/sarah/GitHub/DeepLearningProject/Synthetic-Data/vlm_levels"



def _pick_model_input_device(model) -> torch.device:
    """Best-effort device for input tensors.

    With `device_map='auto'`, `model.device` can be misleading. The input embedding
    weights device is typically the correct place for `input_ids`.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and hasattr(emb.weight, "device"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    """Return start index of needle in haystack, or None."""
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _tokenize_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _locate_span(tokenizer, full_ids: list[int], span_text: str) -> list[int]:
    """Locate span_text (tokenized) inside full_ids and return absolute positions."""
    span_ids_a = _tokenize_ids(tokenizer, span_text)
    span_ids_b = _tokenize_ids(tokenizer, " " + span_text)

    start = _find_subsequence(full_ids, span_ids_a)
    span_ids = span_ids_a
    if start is None:
        start = _find_subsequence(full_ids, span_ids_b)
        span_ids = span_ids_b
    if start is None:
        return []
    return list(range(start, start + len(span_ids)))


def _locate_rel_phrase_token_positions(tokenizer, full_input_ids_1d, question_text: str, rel_phrase: str) -> list[int]:
    """Locate rel_phrase token positions within the full prompt input_ids.

    Strategy:
      1) find question token span inside full prompt tokens
      2) find rel_phrase token span inside question tokens
      3) convert to absolute positions

    Returns absolute positions (indices into full prompt token ids).
    """
    if not question_text or not rel_phrase:
        return []

    full_ids = full_input_ids_1d.tolist() if hasattr(full_input_ids_1d, "tolist") else list(full_input_ids_1d)

    q_ids = _tokenize_ids(tokenizer, question_text)
    q_start = _find_subsequence(full_ids, q_ids)

    phrase_ids_a = _tokenize_ids(tokenizer, rel_phrase)
    phrase_ids_b = _tokenize_ids(tokenizer, " " + rel_phrase)

    if q_start is not None:
        p_start = _find_subsequence(q_ids, phrase_ids_a)
        p_len = len(phrase_ids_a)
        if p_start is None:
            p_start = _find_subsequence(q_ids, phrase_ids_b)
            p_len = len(phrase_ids_b)
        if p_start is None:
            return []
        abs_start = q_start + p_start
        return list(range(abs_start, abs_start + p_len))

    # fallback: search in full prompt directly
    p_start = _find_subsequence(full_ids, phrase_ids_a)
    p_len = len(phrase_ids_a)
    if p_start is None:
        p_start = _find_subsequence(full_ids, phrase_ids_b)
        p_len = len(phrase_ids_b)
    if p_start is None:
        return []
    return list(range(p_start, p_start + p_len))


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class Example:
    caption: str
    question: str
    rel_phrase: str
    gt: str | None
    rel_group: str | None
    rel_type: str | None


def load_example_from_ann(
    *,
    ann_path: str,
    qa_index: int,
    caption_fallback_index: int = 0,
    rel_phrase_override: str | None = None,
) -> Example:
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    qa_list = ann.get("qa", []) or []
    if not qa_list:
        raise ValueError(f"No qa entries found in: {ann_path}")
    if qa_index < 0 or qa_index >= len(qa_list):
        raise ValueError(f"qa_index out of range: {qa_index} (len={len(qa_list)})")

    qa = qa_list[qa_index]
    question = (qa.get("question") or "").strip()
    if not question:
        raise ValueError(f"Empty question at qa_index={qa_index}")

    rel_phrase = rel_phrase_override or (qa.get("rel_phrase") or "").strip()
    if not rel_phrase:
        raise ValueError(
            "rel_phrase is missing for this QA item. "
            "Regenerate data with rel_phrase or pass --rel_phrase_override."
        )

    # Prefer caption linkage via captions_meta + caption_id
    caption = None
    cap_id = qa.get("caption_id")
    caps_meta = ann.get("captions_meta")
    if cap_id is not None and isinstance(caps_meta, list) and caps_meta:
        try:
            cap_id_int = int(cap_id)
        except Exception:
            cap_id_int = None
        if cap_id_int is not None:
            if 0 <= cap_id_int < len(caps_meta) and isinstance(caps_meta[cap_id_int], dict):
                caption = (caps_meta[cap_id_int].get("caption") or "").strip() or None
            if caption is None:
                for c in caps_meta:
                    if isinstance(c, dict) and c.get("id") == cap_id_int:
                        caption = (c.get("caption") or "").strip() or None
                        break

    # Fallback: use plain captions list
    if caption is None:
        caps = ann.get("captions", []) or []
        if isinstance(caps, list) and caps:
            caption = (caps[caption_fallback_index] if caption_fallback_index < len(caps) else caps[0]).strip()

    if not caption:
        raise ValueError("Could not resolve a caption for this QA item.")

    return Example(
        caption=caption,
        question=question,
        rel_phrase=rel_phrase,
        gt=(qa.get("answer") or None),
        rel_group=(qa.get("rel_group") or None),
        rel_type=(qa.get("rel_type") or None),
    )


def build_caption_question_prompt(caption: str, question: str) -> str:
    # Keep this prompt stable so we can reliably locate spans.
    return (
        "You are given a caption describing a synthetic scene.\n"
        "Use only the caption as evidence.\n"
        "Answer the question with only \"yes\" or \"no\".\n\n"
        f"Caption: {caption}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _get_tokens_text(tokenizer, input_ids_1d: Iterable[int]) -> list[str]:
    toks = tokenizer.convert_ids_to_tokens(list(input_ids_1d))
    # Keep raw tokens; downstream we will build readable labels.
    return [str(t) for t in toks]


def _token_labels_for_positions(tokenizer, input_ids: list[int], positions: list[int]) -> list[str]:
    toks = _get_tokens_text(tokenizer, input_ids)
    labels = []
    for p in positions:
        if 0 <= p < len(toks):
            labels.append(toks[p])
        else:
            labels.append("<oob>")
    return labels


def _sp_token_to_display(token: str) -> str:
    # Common token boundary markers:
    # - SentencePiece:  (shown as '▁')
    # - GPT2/BPE: '' (shown as '')
    return token.replace("▁", " ").replace("Ġ", " ")


def _group_sentencepiece_words(tokens: list[str]) -> tuple[list[str], list[list[int]]]:
    """Group SentencePiece-like tokens into word groups.

    Returns:
      - word labels (strings)
      - groups: list of lists of token indices (relative to `tokens`)
    """
    groups: list[list[int]] = []
    words: list[str] = []

    current: list[int] = []
    for i, tok in enumerate(tokens):
        if tok.startswith("▁") and current:
            groups.append(current)
            words.append("".join(_sp_token_to_display(tokens[j]) for j in current).strip() or tokens[current[0]])
            current = []
        current.append(i)

    if current:
        groups.append(current)
        words.append("".join(_sp_token_to_display(tokens[j]) for j in current).strip() or tokens[current[0]])

    return words, groups


def _aggregate_scores_by_groups(scores_per_layer: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Aggregate [L, T] token scores into [L, W] word scores by mean over each group."""
    if scores_per_layer.ndim != 2:
        raise ValueError("scores_per_layer must be [num_layers, num_tokens]")
    out = np.zeros((scores_per_layer.shape[0], len(groups)), dtype=np.float32)
    for wi, g in enumerate(groups):
        if not g:
            continue
        out[:, wi] = scores_per_layer[:, g].mean(axis=1)
    return out


def compute_phrase_to_caption_attention(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    phrase_positions: list[int],
    caption_positions: list[int],
) -> dict[str, Any]:
    """Compute phrase->caption attention maps.

    Returns per-layer scores and an overall average.
    """
    if not phrase_positions:
        raise ValueError("phrase_positions empty")
    if not caption_positions:
        raise ValueError("caption_positions empty")

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attns = out.attentions
    if not attns:
        raise RuntimeError("Model did not return attentions. Ensure attn_implementation='eager'.")

    seq_len = int(input_ids.shape[1])
    phrase_pos = [p for p in phrase_positions if 0 <= p < seq_len]
    cap_pos = [p for p in caption_positions if 0 <= p < seq_len]
    if not phrase_pos or not cap_pos:
        raise ValueError("Phrase/caption positions out of range for the tokenized prompt.")

    per_layer: list[np.ndarray] = []
    for layer_attn in attns:
        # [batch, heads, seq, seq] -> [heads, seq, seq]
        layer = layer_attn[0]
        # Extract phrase queries to caption keys: [heads, P, C]
        sub = layer[:, phrase_pos, :][:, :, cap_pos]
        # Avg over heads + phrase tokens -> [C]
        scores = sub.mean(dim=0).mean(dim=0).float().cpu().numpy()
        per_layer.append(scores)

    per_layer_arr = np.stack(per_layer, axis=0)  # [L, C]
    mean_scores = per_layer_arr.mean(axis=0)  # [C]

    return {
        "per_layer": per_layer_arr,
        "mean": mean_scores,
        "num_layers": per_layer_arr.shape[0],
    }


def plot_phrase_to_caption(
    *,
    out_dir: str,
    caption_tokens: list[str],
    phrase_tokens: list[str],
    scores_per_layer: np.ndarray,
    scores_mean: np.ndarray,
    title_prefix: str,
):
    _safe_mkdir(out_dir)

    # Convert caption token axis to word axis for plotting.
    caption_words, groups = _group_sentencepiece_words(caption_tokens)
    scores_per_layer_w = _aggregate_scores_by_groups(scores_per_layer, groups)
    scores_mean_w = scores_per_layer_w.mean(axis=0)

    # Plot 1: heatmap (layers x caption words)
    plt.figure(figsize=(max(10, 0.45 * len(caption_words)), 6))
    plt.imshow(scores_per_layer_w, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Avg attention (phrasecaption)")
    plt.yticks(range(scores_per_layer_w.shape[0]), [f"L{l}" for l in range(scores_per_layer_w.shape[0])], fontsize=7)
    plt.xticks(range(len(caption_words)), caption_words, rotation=90, fontsize=7)
    plt.title(f"{title_prefix}\nPhrase tokens: {' '.join(phrase_tokens)}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phrase_to_caption_heatmap.png"), dpi=160)
    plt.close()

    # Plot 2: bar chart of mean scores
    plt.figure(figsize=(max(10, 0.45 * len(caption_words)), 4))
    plt.bar(range(len(caption_words)), scores_mean_w)
    plt.xticks(range(len(caption_words)), caption_words, rotation=90, fontsize=7)
    plt.ylabel("Avg attention (mean over layers)")
    plt.title(f"{title_prefix} | Mean over layers")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phrase_to_caption_mean.png"), dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Text-only attention analysis: relational phrase (question) -> caption tokens"
    )

    ap.add_argument("--ann_path", type=str, default=None, help="Path to ann JSON (overrides level/id)")
    ap.add_argument("--base_data_path", type=str, default=None, help="Base dataset path (default: Synthetic-Data/vlm_levels)")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--id", type=str, default=None, help="Image id like 00001_w")

    ap.add_argument("--qa_index", type=int, default=0, help="Which QA index to analyze")
    ap.add_argument(
        "--caption_fallback_index",
        type=int,
        default=0,
        help="If qa.caption_id is missing, use captions[caption_fallback_index]",
    )
    ap.add_argument(
        "--rel_phrase_override",
        type=str,
        default=None,
        help="Override rel_phrase when not present in qa",
    )

    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--out_dir", type=str, default="vis_results/text_only_phrase_attention")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.ann_path:
        ann_path = args.ann_path
    else:
        if args.level is None or args.id is None:
            raise SystemExit("Provide either --ann_path OR both --level and --id")
        base = args.base_data_path or DEFAULT_BASE_DATA_PATH.format(user=os.environ.get("USER", "kkarthikeyan"))
        ann_path = os.path.join(base, f"level_{args.level}", "ann", f"{args.id}.json")

    if not os.path.exists(ann_path):
        raise SystemExit(f"Annotation JSON not found: {ann_path}")

    ex = load_example_from_ann(
        ann_path=ann_path,
        qa_index=args.qa_index,
        caption_fallback_index=args.caption_fallback_index,
        rel_phrase_override=args.rel_phrase_override,
    )

    # Model + tokenizer
    print(f"Loading model: {args.model_id}")
    if torch.cuda.is_available():
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.float32,
            device_map=None,
            attn_implementation="eager",
        )
        model = model.to("cpu")
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    model.eval()

    prompt = build_caption_question_prompt(ex.caption, ex.question)
    tok = tokenizer(prompt, return_tensors="pt")

    model_device = _pick_model_input_device(model)
    tok = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in tok.items()}

    full_ids = tok["input_ids"][0]
    full_ids_list = full_ids.detach().cpu().tolist()

    # Locate caption span inside the prompt
    caption_positions = _locate_span(tokenizer, full_ids_list, ex.caption)

    # Locate phrase positions inside the prompt via question span
    phrase_positions = _locate_rel_phrase_token_positions(
        tokenizer, full_ids, question_text=ex.question, rel_phrase=ex.rel_phrase
    )

    if not caption_positions:
        raise SystemExit("Could not locate caption tokens in prompt. Try simplifying prompt template.")
    if not phrase_positions:
        raise SystemExit(f"Could not locate rel_phrase tokens in prompt: '{ex.rel_phrase}'")

    # Labels
    caption_tok_labels = _token_labels_for_positions(tokenizer, full_ids_list, caption_positions)
    phrase_tok_labels = _token_labels_for_positions(tokenizer, full_ids_list, phrase_positions)

    stats = compute_phrase_to_caption_attention(
        model=model,
        tokenizer=tokenizer,
        input_ids=tok["input_ids"],
        attention_mask=tok.get("attention_mask"),
        phrase_positions=phrase_positions,
        caption_positions=caption_positions,
    )

    out_dir = args.out_dir
    if args.level is not None and args.id is not None:
        out_dir = os.path.join(out_dir, f"level_{args.level}", f"{args.id}", f"qa_{args.qa_index:03d}")
    else:
        out_dir = os.path.join(out_dir, os.path.splitext(os.path.basename(ann_path))[0], f"qa_{args.qa_index:03d}")

    title_prefix = f"Phrase→Caption Attention | rel_phrase='{ex.rel_phrase}'"
    if ex.rel_type or ex.rel_group:
        title_prefix += f" | {ex.rel_group or ''} {ex.rel_type or ''}".strip()

    plot_phrase_to_caption(
        out_dir=out_dir,
        caption_tokens=caption_tok_labels,
        phrase_tokens=phrase_tok_labels,
        scores_per_layer=stats["per_layer"],
        scores_mean=stats["mean"],
        title_prefix=title_prefix,
    )

    # Save a small JSON summary for debugging
    summary = {
        "ann_path": ann_path,
        "qa_index": args.qa_index,
        "caption": ex.caption,
        "question": ex.question,
        "rel_phrase": ex.rel_phrase,
        "rel_group": ex.rel_group,
        "rel_type": ex.rel_type,
        "caption_token_positions": caption_positions,
        "phrase_token_positions": phrase_positions,
        "caption_tokens": caption_tok_labels,
        "phrase_tokens": phrase_tok_labels,
        "token_scores_mean": stats["mean"].tolist(),
    }
    _safe_mkdir(out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plots to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
