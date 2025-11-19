# Synthetic Dataset for VLM Relational Reasoning
This project generates a synthetic dataset of geometric scenes to test and analyze the relational reasoning capabilities of Vision-Language Models (VLMs).
The goal is to create a controlled environment to probe VLM attention patterns, specifically addressing why they often succeed at entity recognition but fail at understanding spatial relationships (as outlined in the project proposals [cite: Project_Proposal_Yifan_Hou.pdf, DL2025_Project_Proposal.pdf]).
The dataset consists of images with colored shapes (circles, squares, triangles, stars) and detailed JSON annotations. These annotations include:
- Object bounding boxes (with correct tight boxes for all shapes [cite: img-gen.py]).
- A complete list of spatial relations (e.g., left_of, above).
- Descriptive captions (e.g., "The red circle is left of the blue square.").
- A balanced set of "yes" and "no" question-answer pairs to test genuine understanding [cite: aug-gen.py, ann-gen.py].
## Data Generation Pipeline
The data is generated using a three-stage pipeline, orchestrated by main.py [cite: main.py].
- img-gen.py: Generates the base set of N_SCENES (e.g., 20) with unique geometric layouts. It creates multiple versions of each scene with different background colors and color combinations [cite: img-gen.py].
Output: base_scenes/images/ and base_scenes/ann_base/
- ann-gen.py: Reads the basic annotations from ann_base/ and enriches them. It computes all pairwise spatial relations, generates descriptive captions, and creates a balanced set of "yes" and "no" QA pairs [cite: ann-gen.py].
Output: base_scenes/ann/
- aug-gen.py: Takes the complete base scenes and applies a set of geometric augmentations (flips, rotations, crops). For each transformation, it correctly remaps all object bounding boxes, relations (e.g., left_of -> right_of), captions, and QA to be consistent with the new, transformed image [cite: aug-gen.py].
Output: synthetic_vlm_rel_aug/images/ and synthetic_vlm_rel_aug/ann/
## How to Run
To generate the complete dataset, simply run the main pipeline script from your terminal:
``` python main.py ```


This will execute the three stages in order, populating the synthetic_vlm_rel_aug directory with the final augmented dataset.
Output Directories
base_scenes/: Contains the initial, non-augmented images and annotations. This folder is ignored by .gitignore.
synthetic_vlm_rel_aug/: Contains the final, augmented dataset, which is the output you should use for model training and analysis. This folder is also ignored by .gitignore.
