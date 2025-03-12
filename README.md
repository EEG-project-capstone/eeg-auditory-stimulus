# EEG Auditory Stimulus Project

This repository focuses on the preliminary analysis of EEG data collected from controlled (healthy) patients subjected to auditory stimuli. The goal is to replicate findings from existing studies and explore the brain's response to auditory inputs.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Understanding how the brain processes auditory stimuli is crucial for insights into neural mechanisms and potential applications in neurorehabilitation. This project aims to analyze EEG data to observe brain responses to controlled auditory inputs, replicating methodologies from established research.

## Repository Structure

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

---
