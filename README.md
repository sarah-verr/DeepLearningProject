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
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Synthetic Data Generation

Use the following scripts to generate and augment the synthetic datasets required for the framework.

* **Generate Synthetic Data:**
  ```bash
  python3 generate_data.py
  ```

* **Augment Datasets:** Creates various variations of the above generated dataset (flipping, rotation, etc)
  ```bash
  python3 aug_datasets.py     
  ```

---

## 3. Inference and Visualization (`main.py`)

This is the primary script for running the LLaVA model against a specific image and its corresponding questions. It supports tracking attention on specific groups of image patches (e.g., "Target Object" vs "Distractor").

**Usage:**
```bash
python main.py --level <level> --id <id> [options]
```

Options are other optional flags detailed below in the Arguments section.

### Configuration (Target Groups)

*Important*: The script tracks specific patches defined in the code. Before running, open main.py and modify the TARGET_GROUPS list to match the patch indices you wish to analyze:

```python 
# Inside main.py
TARGET_GROUPS = [
    [459, 460],       # Group 0 (Cyan)
    [483, 484, 485]   # Group 1 (Magenta)
]
```

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

---

## 4. Debugging and Analysis (`debug.py`)

This utility analyzes an image to identify the most "active" patches based on pixel variance (standard deviation). It helps predict which areas are likely to draw the model's attention.

**Usage:**
```bash
python debug.py [--level <level_number>] [--id <image_id>]
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--level` | Optional | `1` | The dataset level where the image is located. |
| `--id` | Required | - | The identifier for the image file. |

### Examples

**1. Analyze a specific image:**
Analyzes `level_3/images/00042_a.png` and displays a plot showing the top-2 active patches.
```bash
python debug.py --level 3 --id 00042_a
```

**2. Run with default settings:**
Analyzes the default image (`level_1/images/00007_b.png`).
```bash
python debug.py
```