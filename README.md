# LLaVA Attention and Evaluation Framework

This project provides a suite of tools to run inference on the LLaVA model for visual question-answering tasks. It includes scripts to generate synthetic data, evaluate model performance, and debug/visualize image features (attention maps).

---

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
  - `--advanced` (includes touching/overlapping + inside/near/far/next_to/beside/encapsulates)

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
- `qa[*].subject_id` / `qa[*].object_id`: object ids referenced in the question
- `qa[*].rel_type`: relation type (e.g., `left_of`)
- `qa[*].rel_group`: relation group (`PRIMARY` or `ADVANCED`)
- `meta`: `{img_size, patch, grid_dim}`

---

## 3. Inference and Visualization (`main.py`)

This is the primary script for running the LLaVA model against a specific image and its corresponding questions. It supports tracking attention on specific groups of image patches (e.g., "Target Object" vs "Distractor").

**Usage:**
```bash
python main.py --level <level> --id <id> [options]
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

### Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--level` | **Required** | Specifies the dataset level to use (e.g., `1`). |
| `--id` | **Required** | The unique identifier for the image file (e.g., `00007_b`). |
| `--plot_trends` | Optional | Generates line graphs showing total attention on Target Groups per layer. Fast (does not generate heatmaps). |
| `--plot_attention` | Optional | If present, generates and saves detailed attention heatmaps for each layer. <br>**Note:** Significantly slows down execution. |
| `--num_questions` | Optional | Number of questions to randomly sample (e.g., 5). If omitted, processes all questions in the JSON |



### Examples

**1. Run a standard evaluation:**
Processes the image, prints results to the console, and saves a summary plot and JSOxwN log in `vis_results/`.
```bash
python main.py --level 1 --id 00007_b
```

**2. Run evaluation with attention plotting:**
Performs standard evaluation and creates subdirectories containing attention heatmaps for each question.
```bash
python main.py --level 1 --id 00007_b --plot_attention
```

**3. Run trend plots only (faster):**
```bash
python main.py --level 1 --id 00007_b --plot_trends
```

**4. Run a subset of questions:**
```bash
python main.py --level 1 --id 00007_b --num_questions 5 --plot_trends
```

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