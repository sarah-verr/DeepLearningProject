# LLaVA Attention and Evaluation Framework

This project provides a suite of tools to run inference on the LLaVA model for visual question-answering tasks. It includes scripts to generate synthetic data, evaluate model performance, and debug/visualize image features (attention maps).

---

## Quick Start: Key Results

If you just want the main results, use the commands below after setup.

**1) Accuracy runs (all datasets)**
```bash
python experiments/run_all_accuracy_experiments.py
```
Outputs CSVs and summaries under `results_llava_hf/<model>/accuracy_question_ablation/`.

**2) Masked attention accuracy**
```bash
python experiments/run_masked_accuracy_experiments.py
```
Writes masked-accuracy CSVs in the same results tree.

**3) Logit lens (yes/no)**
```bash
python experiments/logit_lens/run_visual_logit_lens.py
```
Outputs per-example JSONL under `results_llava-hf/llava-1.5-7b-hf/visual_logit_lens/`.

**4) Logit lens (attribute: shape/color)**
```bash
python experiments/logit_lens/run_visual_logit_lens_attribute.py
```
Outputs per-example JSONL under `results_llava-hf/llava-1.5-7b-hf/visual_logit_lens_attribute/`.

**5) Plotting (logit lens)**
```bash
python experiments/logit_lens/plots/plot_logit_lens.py
python experiments/logit_lens/plots/plot_logit_lens_raw.py
python experiments/logit_lens/plots/plot_logit_lens_attribute.py
```
Plots are written under `results_llava-hf/llava-1.5-7b-hf/logit_lens*`.

**6) Occlusion bias analysis (optional)**
```bash
python experiments/logit_lens/plots/plot_occlusion_bias.py \
  --base_dir results_llava-hf/llava-1.5-7b-hf/visual_logit_lens_occluded \
  --out_dir results_llava-hf/llava-1.5-7b-hf/analysis_occlusion_bias
```

**Cluster runs (Slurm)**
- `run_eval.sh`: batch evaluation job (accuracy script).
- `run_experiment.sh`: batch run for `main.py`.
- `run_logit.sh`: batch logit-lens runs; edit the active command in the script before submission.

## 1. Environment Setup

Before running any scripts, ensure your environment is configured correctly.

> **_NOTE:_** The instructions below are specific to running the code on the **ETH student cluster**.

### Cluster Configuration (ETH)
If you are running this code on the student cluster, follow these steps to avoid quota issues and allocate resources:

1. **Set the Hugging Face Cache:** Store models in scratch space rather than the home directory.
   ```zsh
   export HF_HOME="/work/scratch/{your user name}"
   ```

2. **Start an Interactive Job:** Run on a compute node.
   ```zsh
   srun -A deep_learning --pty bash
   ```

### Python Environment
Once on the compute node, set up the dependencies. You can simply run `source setup.sh` if available, or manually set up the virtual environment:

```bash
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Synthetic Data Generation

Use the following scripts to generate and augment the synthetic datasets required for the framework.

* **Generate Synthetic Data:**
  The generator now requires selecting at least one question group:
  - `--primary` (left/right/above/below)
  - `--advanced` (includes touching/overlapping + inside/encapsulates)

  Note: phrases for `near`/`far`/`next_to`/`beside` exist in code, but these relation types are currently not sampled (they are intentionally excluded from the active ADVANCED set).

  Examples:
  ```bash
  # PRIMARY only
  python Synthetic-Data/generate_data.py --primary --levels 2 --scenes_per_level 20

  # ADVANCED only
  python Synthetic-Data/generate_data.py --advanced --levels 2 --scenes_per_level 20

  # BOTH
  python Synthetic-Data/generate_data.py --primary --advanced --levels 2 --scenes_per_level 20
  ```

  Optional flags:
  - `--no_dedup_qa`: disables deduplication of repeated question text.

* **Augment Datasets:** Creates various variations of the above generated dataset (flipping, rotation, etc)
  ```bash
  python Synthetic-Data/aug_datasets.py
  ```

### Annotation JSON schema (high level)
Each `ann/*.json` now stores extra supervision that downstream scripts use:
- `objects[*].patch_indices`: precomputed patch indices (row-major) intersecting each object bbox
- `captions_meta[*]`: structured captions with relation metadata (useful for text-only evaluation)
- `captions_meta[*].entailed_qa_ids` / `captions_meta[*].contradicted_qa_ids`: QA ids linked to that caption (derived from `qa[*].caption_id` + `qa[*].answer`, but stored for convenience)
- `qa[*].subject_id` / `qa[*].object_id`: object ids referenced in the question
- `qa[*].rel_type`: relation type (e.g., `left_of`)
- `qa[*].rel_group`: relation group (`PRIMARY` or `ADVANCED`)
- `qa[*].rel_phrase`: the exact relational phrase used in the question (e.g., `to the left of`)
- `qa[*].id`: stable per-image question id (0..N-1)
- `qa[*].caption_id`: optional link to the caption in `captions_meta` that was used as evidence for this QA item (only present when captions are generated; older JSONs may not have it)
- `meta`: `{img_size, patch, grid_dim}`

---

## 3. Inference and Visualization (`main.py`)

This is the primary script for running the LLaVA model against a specific image and its corresponding questions. It supports tracking attention on specific groups of image patches (e.g., "Target Object" vs "Distractor").

`main.py` is **config-driven** (YAML or JSON).

**Usage:**
```bash
python main.py --config configs/run_configs.yaml
```

Options are other optional flags detailed below in the Arguments section.

### Configuration (Target Groups)

*Important*: You no longer need to manually define `TARGET_GROUPS` in code.

`main.py` now reads target patches directly from the JSON:
- Per-object target patches come from `objects[*].patch_indices`
- Per-question targets come from `qa[*].subject_id` and `qa[*].object_id`

When attention extraction is enabled, it produces:
- Trend plots for attention on all objects
- A per-question plot comparing attention on the subject vs object across layers
- Plot titles include `rel_group` and `rel_type`

### Config file

Example (see `configs/run_configs.yaml`):
```yaml
level: 4
id: "00010_b"

num_questions: 5

plot_trends: true
plot_attention: false

plot_relational_phrase_attention: true
relational_phrase_attention_mode: "simple"  # "simple"|"detailed"

# Optional overrides
model_id: "llava-hf/llava-1.5-7b-hf"
base_data_path: "/path/to/Synthetic-Data/vlm_levels"
base_output_dir: "vis_results"
```

When enabled, `plot_relational_phrase_attention` uses `qa[*].rel_phrase` and produces phrase→image attention visualizations.

---

## 4. Debugging and Analysis (`debug.py`)

This utility maps object bounding boxes to patch indices and visualizes patch coverage.

With the updated dataset, `debug.py` will use `objects[*].patch_indices` from the JSON when available (and fall back to computing them for older JSONs).

**Usage:**
```bash
python debug.py --level <level_number> --id <image_id>
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--level` | Optional | `1` | The dataset level where the image is located. |
| `--id` | Required | - | The identifier for the image file. |

### Examples

**1. Analyze a specific image:**
```bash
python debug.py --level 3 --id 00042_a
```

---

## 5. Bulk Evaluation (`compute_accuracy.py`)

This script evaluates LLaVA across one or more dataset levels and produces:
- `evaluation_results.json`: per-level accuracy
- per-level accuracy breakdown by `rel_group` (`PRIMARY` vs `ADVANCED`)
- `model_calls_log.csv`: per-question log including `Rel_Group` and `Rel_Type`

**Usage:**
```bash
python compute_accuracy.py --levels 0 1 2 3 4 5 6
```

Optional flags:
- `--output_json evaluation_results.json`
- `--log_file model_calls_log.csv`

Text-only baseline:
- `--text_only`: evaluates relational reasoning from a text-only prompt (no image).
  - For newer datasets, the prompt is built from the per-question linked caption (`captions_meta` + `qa[*].caption_id`) followed by the question.
  - For older datasets without caption linkage, it falls back to a simple scene description.
  - The CSV logs the full transcript in the `Prompt` column (input prompt + `MODEL: <completion>`) and also stores the raw generated text in `Completion`.
  - The CSV includes a `Mode` column to distinguish `image` vs `text_only` runs.
  
