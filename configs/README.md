# `/config` Folder

This directory contains configuration files that specify parameters and settings for various EEG analysis workflows. By centralizing these parameters in config files, the code in the main repository remains clean and modular, while allowing for experimentation with different settings.

## Structure

```
configs/
├── claassen_cfg.yml
├── johnsen_cfg_test.yml
└── johnsen_cfg.yml
```

## Typical Contents
**Filtering Parametersz**
- Notch filter frequencies (e.g., 50 Hz, 60 Hz).
- Bandpass range (e.g., 0.1–100 Hz).

**Artifact Removal**
- ICA thresholds for eye-blink or EOG channel detection.
- Channels to exclude from analysis (e.g., Fp1, T7).
- Epoching and Stimuli
- Event markers or trigger codes (e.g., beep_onset, word_onset).
- Time windows before and after each event.
- Directories or filenames for audio stimuli.

**Analysis Parameters**
- Frequencies to analyze for reactivity or ITPC (e.g., delta, alpha, beta).
- Machine-learning settings for the command-following paradigm (number of cross-validation folds, SVM kernel type, etc.).

**Output Directories**
- Where to store figures, logs, or processed data (e.g., data/results/).

### Classen-Specific Parameters
```yaml
dc_threshold: 0.01  # For CON005, it might need to be set as 0.005.