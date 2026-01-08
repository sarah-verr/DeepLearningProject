import os
import json
from typing import Any, Dict


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file.

    Supports .yaml/.yml (requires PyYAML) and .json. Returns an empty dict if the
    file is empty. Raises FileNotFoundError if the path does not exist and
    ValueError for unsupported extensions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import]
        except Exception as e:  # pragma: no cover - import-time error path
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}

    raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml/.yml or .json")


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults and validate required fields for the main script.

    This mirrors the previous _normalize_config in main.py but is reusable.
    Required keys: level, id.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/object at the top level.")

    if "level" not in cfg or "id" not in cfg:
        raise ValueError("Config must include required keys: level, id")

    out: Dict[str, Any] = dict(cfg)

    # Basic options
    out.setdefault("num_questions", None)
    out.setdefault("plot_trends", False)

    # attention_mode controls attention extraction detail level
    # None     -> no attention plots
    # simple   -> lightweight summaries (e.g., layer-thirds maps), no per-layer grids
    # detailed -> full per-layer grids (and optional summaries)
    out.setdefault("attention_mode", None)
    if out["attention_mode"] not in {None, "simple", "detailed"}:
        raise ValueError("attention_mode must be one of: None, simple, detailed")

    # optional overrides for paths/model
    out.setdefault("model_id", None)
    out.setdefault("base_output_dir", None)
    out.setdefault("base_data_path", None)

    # attention_source controls which token(s) act as the query for
    # image attention when building maps/trends.
    #   - "first_generated_token" (default): use first generated token
    #   - "rel_phrase": use relational phrase tokens from the prompt
    out.setdefault("attention_source", "first_generated_token")
    if out["attention_source"] not in {"first_generated_token", "rel_phrase"}:
        raise ValueError(
            "attention_source must be one of: first_generated_token, rel_phrase"
        )

    # Normalize types
    out["level"] = str(out["level"])
    out["id"] = str(out["id"])
    if out["num_questions"] is not None:
        out["num_questions"] = int(out["num_questions"])

    out["plot_trends"] = bool(out["plot_trends"])

    return out
