# `preprocessing/` Folder

This directory contains scripts and utility modules that help prepare and organize raw EEG data before running higher-level analyses (e.g., reactivity tests, language tracking, or machine-learning classification). By separating out these steps, we keep the main analysis code clean and modular.

## Folder Contents

1. **`__init__.py`**  
   - Marks this folder as a Python package.
  
2. **`eeg_loader.py`**  
   - Functions for **reading and loading** EEG data files (e.g. EDF) into memory.  
   - Includes convenient methods for listing files in the raw data directory, converting timestamps to sample indices, or handling channel selection.  
   - Useful for quickly getting EEG data in the right format for further processing (e.g., epoching, filtering).

3. **`save_functions.py`**  
   - Utility methods for **saving** intermediate results or logs (e.g., saving cleaned data, exporting figures, or writing metadata to disk).  
   - Could include specialized logging functions or routines to create well-organized output directories.  
   - Often referenced by analysis scripts (e.g., to automatically save preprocessed data or debugging information).

## How to Use
1. **Load Raw EEG Data**  
   - Call methods from `eeg_loader.py` to read EDF files into memory.  
   - Example:
     ```python
     from preprocessing.eeg_loader import load_edf_files
     raw_data = load_edf_files("path/to/edf_dir")
     ```

2. **Process or Clean the Data**  
   - If you have additional scripts in this folder (e.g., filtering or artifact-removal scripts), run those next.  
   - After you have a cleaned/filtered dataset, you can pass it to the analysis modules (in `eeg_auditory_stimulus/`).

3. **Save Outputs**  
   - Use `save_functions.py` to write the cleaned data, intermediate logs, or debug plots to disk.  
   - This ensures consistent naming conventions and output structures.

## Extending This Folder

- **Add More Scripts**: If you develop new cleaning steps (like advanced artifact removal, channel re-montaging, or baseline corrections), create additional `.py` files here for clarity.  
- **Configuration**: Adjust relevant paths or thresholds in the config files in `configs/` to streamline usage across different data sets or projects.
