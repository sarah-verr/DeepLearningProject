# Running Synthetic Data Generation and Augmentation

To generate the synthetic dataset, run the following command:
```python
python3 generate_data.py
```

and then for augmenting the datasets, use:
```python
python3 aug_datasets.py     
```

# Running script to generate visualisations
> **_NOTE:_**  These instructions are specific to running the code on ETH student cluster

First of all, if you are running this code on the student-cluster, make sure to store the model in the scratch space, instead of the home directory, to avoid quota issues. To do this, run: 
```zsh
export HF_HOME="/work/scratch/{your user name}"
```
Then, you can run an interactive job on a compute node using: 
```zsh
srun -A deep_learning --pty bash
```

Once you are on the compute node, make sure to create a virtual environment andinstall the required packages using:
```python
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

To generate visualisations from the sample image, run:
```python
python3 main.py
```

This will save the visualisations from all layers in a .pdf file. 

The `debug.py` is used to visualise the patches over a single image, and how the pixel density are distributed across the patches. To run this, use:
```python
python3 debug.py
```

It also gives the patch index of the shapes, which is useful for analysis in `main.py`. Make sure to fill the TARGET_PATCHES variable in `main.py` with those indices. 