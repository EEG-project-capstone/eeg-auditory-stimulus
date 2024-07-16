import mne
import os

def load_eeg(eeg_path, config=None):
    """
    Load EEG data from a file and apply preprocessing steps
    param:
        eeg_path: Path to the EEG file
        config: Configuration dictionary
    return:
        fname: File name
        raw: Raw EEG data
    """
    # Load EEG data
    fname, extension = os.path.splitext(eeg_path)
    if extension == '.fif':
        raw = mne.io.read_raw_fif(eeg_path, preload=True)
    elif extension == '.edf':
        raw = mne.io.read_raw_edf(eeg_path, preload=True)
    else:
        raise ValueError(f'File extension {extension} not supported')
    
    raw = raw.pick(picks='eeg', exclude='bads')

    # Resample data
    if config.get('sfreq', False):
        print(f"Resampling data to {config['sfreq']} Hz")
        raw.resample(config['sfreq'])

    # Filter data
    lo_pass = config.get('low_pass', None)
    hi_pass = config.get('high_pass', None)    
    raw.filter(lo_pass, hi_pass)

    return fname, raw


