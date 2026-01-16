"""Utilities for logit-lens style probing."""

from typing import Any

import torch


def _token_ids(tokenizer, variants: list[str]) -> set[int]:
    ids: set[int] = set()
    for s in variants:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids.add(int(enc[-1]))
    return ids


def yes_no_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """Return token id sets for yes/no variants."""
    yes_ids = _token_ids(tokenizer, ["Yes", " yes", "yes"])
    no_ids = _token_ids(tokenizer, ["No", " no", "no"])
    return yes_ids, no_ids


def _option_token_ids(tokenizer, option: str) -> set[int]:
    variants = [
        option,
        f" {option}",
        option.capitalize(),
        f" {option.capitalize()}",
    ]
    return _token_ids(tokenizer, variants)


def attribute_token_ids(tokenizer, options: list[str]) -> dict[str, set[int]]:
    ids: dict[str, set[int]] = {}
    for opt in options:
        ids[opt] = _option_token_ids(tokenizer, opt)
    return ids


def logit_lens_yesno(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return per-layer yes/no probabilities at a specific position.

    Uses each layer's hidden state -> LM head projection.
    """
    call_kwargs = dict(model_kwargs or {})
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **call_kwargs,
        )

    hidden_states = out.hidden_states
    if not hidden_states:
        return []

    if position is None:
        position = int(input_ids.shape[1] - 1)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        return []

    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yes_ids_list = sorted(yes_ids)
    no_ids_list = sorted(no_ids)
    per_layer = []
    for h in hidden_states:
        h_pos = h[:, position, :]
        logits = lm_head(h_pos)
        logit_total = float(torch.logsumexp(logits[0], dim=-1).item())
        logit_yes = 0.0
        logit_no = 0.0
        if yes_ids_list:
            yes_tensor = logits[0, yes_ids_list]
            logit_yes = float(torch.logsumexp(yes_tensor, dim=0).item())
        if no_ids_list:
            no_tensor = logits[0, no_ids_list]
            logit_no = float(torch.logsumexp(no_tensor, dim=0).item())
        probs = torch.softmax(logits, dim=-1)[0]
        p_yes = float(sum(probs[i].item() for i in yes_ids if i < probs.numel()))
        p_no = float(sum(probs[i].item() for i in no_ids if i < probs.numel()))
        denom = p_yes + p_no
        if denom > 0:
            p_yes /= denom
            p_no /= denom
        per_layer.append(
            {
                "p_yes": p_yes,
                "p_no": p_no,
                "logit_yes": logit_yes,
                "logit_no": logit_no,
                "logit_total": logit_total,
            }
        )

    return per_layer


def logit_lens_attribute(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    options: list[str],
    attention_mask: torch.Tensor | None = None,
    position: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return per-layer attribute probabilities for a fixed option set."""
    call_kwargs = dict(model_kwargs or {})
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **call_kwargs,
        )

    hidden_states = out.hidden_states
    if not hidden_states:
        return []

    if position is None:
        position = int(input_ids.shape[1] - 1)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        return []

    option_ids = attribute_token_ids(tokenizer, options)
    per_layer = []
    for h in hidden_states:
        h_pos = h[:, position, :]
        logits = lm_head(h_pos)
        logit_total = float(torch.logsumexp(logits[0], dim=-1).item())
        logit_by_choice: dict[str, float | None] = {}
        valid_logits = []
        valid_labels = []
        for opt, ids in option_ids.items():
            ids_list = sorted(ids)
            if not ids_list:
                logit_by_choice[opt] = None
                continue
            opt_logits = logits[0, ids_list]
            logit_val = float(torch.logsumexp(opt_logits, dim=0).item())
            logit_by_choice[opt] = logit_val
            valid_logits.append(logit_val)
            valid_labels.append(opt)

        p_by_choice: dict[str, float | None] = {opt: None for opt in option_ids.keys()}
        if valid_logits:
            probs = torch.softmax(torch.tensor(valid_logits), dim=0).tolist()
            for opt, prob in zip(valid_labels, probs):
                p_by_choice[opt] = float(prob)

        per_layer.append(
            {
                "p_by_choice": p_by_choice,
                "logit_by_choice": logit_by_choice,
                "logit_total": logit_total,
            }
        )

    return per_layer
