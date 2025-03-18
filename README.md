# EEG Auditory Stimulus Project

This repository hosts code, configurations, and documentation for analyzing EEG data collected from healthy participants in response to various auditory stimuli. The overarching objective is to replicate findings from established literature and explore novel approaches to understanding the brain’s response to auditory inputs, with an eye toward potential applications in neurorehabilitation and brain-injury prognostication.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Auditory stimulus paradigms provide critical insights into how the brain processes and responds to sounds and speech. By collecting EEG data and applying targeted analyses, this project aims to:

- Replicate documented methodologies from key research on auditory-evoked potentials.

- Integrate multiple paradigms (e.g., beep-based reactivity, language tracking, command-following).

- Refine preprocessing steps (notch filters, ICA for artifact removal, epoching) to ensure data quality.

- Evaluate machine-learning approaches (e.g., SVM) for detecting covert responses in motor imagery tasks.

Ultimately, this repository lays the groundwork for a modular EEG software toolkit that can be adapted for real-time, bedside usage in both healthy individuals and patients with brain injuries.

```
eeg-auditory-stimulus/
├── configs/
│   ├── claassen_cfg.yml
│   ├── johnsen_cfg_test.yml
│   └── johnsen_cfg_test.yml
├── eeg_auditory_stimulus/
│   ├── __init__.py
│   ├── rodika_modularized.py
│   ├── claassen_analysis.py
│   └── johnsen_quan.py
├── notebooks/
│   ├── claassen_analysis.ipynb
│   ├── johnsen_quan.ipynb
│   ├── johnsen_statistic.ipynb
│   └── rodika_analysis.ipynb
├── preprocessing/
│   ├── eeg_loader.py
│   ├── filter_data.py
│   └── save_functions.py
├── .gitignore
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
└── setup.py
```

## Repository Structure
To read more about what each folder contains, please navigate to their respective folders in the repository. Below is a brief explanation of each file. 

- `configs/`: Configuration files for experiments and data processing.
- `eeg_auditory_stimulus/`: Core Python modules for EEG analyses.
- `notebooks/`: Jupyter notebooks detailing data exploration and analysis workflows.
- `preprocessing/`: Scripts for preprocessing raw EEG data.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `LICENSE`: License information for the repository.
- `README.md`: This document.
- `main.py`: Main script to run analyses.
- `requirements.txt`: List of Python dependencies.
- `setup.py`: Script for setting up the Python package.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/EEG-project-capstone/eeg-auditory-stimulus.git
   cd eeg-auditory-stimulus
   ```

2. **Create and activate a virtual environment:**

   ```bash
    conda create -n "eeg"
    conda activate eeg
    conda install pip
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To begin analyzing EEG data:
1. **Configuration:**

   Modify configuration files in the `configs/` directory to adjust parameters for different experiments.

2. **Preprocess the data:**

   Use the scripts in the `preprocessing/` directory to clean and prepare raw EEG data for analysis.

3. **Run analyses:**

    Explore the Jupyter notebooks in the `notebooks/` directory for detailed analysis workflows.
    Run respectives files in the `eeg_auditory_stimulus/` directory for modular functions.

## Contributing

- Khanh Ha
- Nguyen Ha
- Joobee Jung
- Trisha Prasant

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Conclusion

This repository demonstrates how three distinct EEG paradigms—reactivity, language tracking, and motor command-following—can be united in a single software platform to facilitate data collection and analysis at the bedside. By providing standardized preprocessing pipelines, modular analytical modules, and example notebooks, we aim to streamline research into auditory EEG responses. Ongoing development and user contributions will focus on refining artifact removal, expanding machine learning capabilities, and validating these methods in both healthy participants and clinical populations. Ultimately, we hope this project will serve as a robust foundation for further innovation and collaboration in EEG-based neuroprognostication and rehabilitation research.
