# `eeg_auditory_stimulus` Folder

This folder hosts Python modules that implement various EEG analysis paradigms and pipelines. It is part of the broader EEG Auditory Stimulus project, where each file focuses on a specific paradigm or analysis workflow. Below is an overview of each file and how it might be used or extended.

## Files
`__init__.py`
- Makes this directory a proper Python package.

`claassen_analysis.py`
- Scripts and functions specific to the Claassen approach or dataset (e.g., motor command-following or specialized analysis steps).
- Includes machine-learning classification routines (like SVM) or specialized epoching for the command-following paradigm.

`johnsen_quan.py`
- Implements additional or alternative analyses, possibly building on or replicating a study by Johnsen, et al. 
- Define custom spectral, reactivity, or time-frequency analyses.

`rodika_modularized.py`
- A primary module that unifies multiple auditory-stimulus paradigms (e.g., beep reactivity, language tracking, command-following) into one coherent pipeline.
- Contains functions or classes for loading/preprocessing data, performing spectral analysis (like ITPC), and saving output figures/logs.
- Referenced by main scripts (e.g. main.py) or notebooks in the project.

## How to Use
- Import Modules:
```
from eeg_auditory_stimulus import claassen_analysis, rodika_modularized
```
  - Then call the provided functions/classes (e.g., for data loading, artifact rejection, model training).

- Customize Analysis:
  - Adjust parameters (filter ranges, epoch lengths, etc.) in code or via config files in  `configs/`.
  - For specialized tasks, extend these modules by adding new functions or modifying existing workflows.

- Integration:
  - Typically referenced in Jupyter notebooks under `notebooks/`, or invoked by main.py.
  - The pipeline usually expects data to be preprocessed (e.g., with scripts in preprocessing/) and config settings from `configs/`.

### Need Help?
Check the main README.md for installation instructions and an overview of the project’s data flow. If you encounter issues or have improvements, please open a pull request or issue in the repository.








