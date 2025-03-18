# `notebooks/` Folder
This directory contains Jupyter notebooks demonstrating various EEG analysis methods and workflows. Each notebook focuses on a specific approach or dataset, showcasing how to load data, preprocess signals, run analyses, and interpret results. You can run each notebook independently to test workflows on one pateint before updating the `eeg_auditory_stimulus` modules. Below is a short description of each notebook:

## Notebook Overview

1. **`claassen_analysis.ipynb`**  
   - Demonstrates how to run command-following (CMD) analyses using the Claassen approach.  
   - Provides interactive code cells that load EEG data, apply filtering, epoch the data around motor commands, and run machine-learning classification (e.g., SVM).  
   - Includes functions to visualize predictions and interpret accuracy metrics.

2. **`johnsen_quan.ipynb`**  
   - Shows a quantitative reactivity approach inspired by the Johnsen methodology.  
   - Steps through artifact rejection (ICA), segmentation, and Z-score tests to detect reactivity in specific EEG frequency bands following auditory stimuli (like beeps).  
   - Helps validate consistent reactivity baselines and baseline-subtraction techniques.

3. **`johnsen_statistic.ipynb`**  
   - Presents statistical evaluations related to the Johnsen approach, such as permutation tests or group-level comparisons.  
   - Walks through computing p-values or confidence intervals for reactivity, ITPC, or classification metrics.  
   - Useful for understanding how to interpret significance levels in small-sample EEG studies.

4. **`rodika_analysis.ipynb`**  
   - Focuses on the “Rodika” modularized pipeline, integrating beep reactivity, passive language tracking, and motor command-following in one demonstration.  
   - Illustrates how to call shared functions (from `rodika_modularized.py`) for filtering, epoching, ITPC, or classification.  
   - Offers visual examples of speech-entrainment plots, beep reactivity Z-scores, and SVM classification outputs.

## Getting Started

1. **Environment Setup**  
   - Make sure you’ve installed the Python environment via `pip install -r requirements.txt` or `setup.py`.  
   - Launch Jupyter or an equivalent environment (e.g., JupyterLab, VS Code Notebooks) from the project root directory.

2. **Running a Notebook**  
   - In a terminal or command prompt, navigate to this `notebooks/` folder:  
     ```bash
     cd notebooks
     jupyter notebook
     ```  
   - Open the notebook of your choice (e.g., `claassen_analysis.ipynb`) to explore the analysis steps.

3. **Configuration & Data**  
   - Some notebooks may read parameters from `configs/` or require data in specific folders (like `data/edfs/`).  
   - Adjust file paths in the code cells if your data or config is in a different location.

### Need More Help?**  
Check the main [README.md](../README.md) for environment setup and project overview, or refer to the repository’s issues/pull requests for ongoing discussions.
