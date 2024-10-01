import mne
import os
import numpy as np

def read_file(eeg_path):
    fname, extension = os.path.splitext(eeg_path)
    if extension.lower() == '.fif':
        raw = mne.io.read_raw_fif(eeg_path, preload=True)
    elif extension.lower() == '.edf':
        raw = mne.io.read_raw_edf(eeg_path, preload=True)
    else:
        raise ValueError(f'File extension {extension} not supported')
    return fname, raw

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
    fname, raw = read_file(eeg_path)
    raw.pick(picks='eeg', exclude='bads')

    # Rename channels
    raw= raw.rename_channels(config['channel_map'])
    missing_channels = [x for x in config['channels'] if x not in raw.info['ch_names']]
    if len(missing_channels) > 0:
        raise ValueError(f"Missing channels: {missing_channels}")
    raw.pick(config['channels'])

    # Resample data
    if config.get('sfreq', False):
        print(f"Resampling data to {config['sfreq']} Hz")
        raw.resample(config['sfreq'])

    # Filter data
    lo_pass = config.get('l_freq', None)
    hi_pass = config.get('h_freq', None)    
    raw.filter(lo_pass, hi_pass)

    return fname, raw

def get_eeg_timestamps(raw_data):
    # Start time of the recording
    start_time = raw_data.info['meas_date']

    # Duration of the recording
    n_samples = raw_data.n_times  # Number of samples
    sampling_frequency = raw_data.info['sfreq']  # Sampling frequency
    duration = timedelta(seconds=n_samples / sampling_frequency)
    # End time of the recording
    end_time = start_time + duration

    # Adjust EEG System time to UTC time
    start_time = start_time + timedelta(hours=7)
    end_time = end_time + timedelta(hours=7)
    
    return start_time, end_time

def load_event(event_path, config):
    """
    Load event data from a file
    param:
        event_full_path: Path to the event file
        config: Configuration dictionary
    return:
        fname: File name
        event: Event data
    """
    instructions = np.array(mne.read_events(event_path))
    print(instructions)
    return
