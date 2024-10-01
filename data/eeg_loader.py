import mne
import os
import numpy as np
import pandas as pd
from datetime import timedelta

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

def load_stimulus(event_full_path, start_time, end_time):
    df = pd.read_csv(event_full_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['start_time'] = pd.to_datetime(df['start_time'], unit='s', utc=True)
    df['end_time'] = pd.to_datetime(df['end_time'], unit='s', utc=True)

    ptc_df = df[['patient_id','start_time','end_time']].groupby('patient_id',as_index=False).agg(['min', 'max'])
    ptc_df['start'] = ptc_df[('start_time', 'min')]
    ptc_df['end'] = ptc_df[('end_time', 'max')]

    ptc_df['start_str'] = ptc_df['start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    ptc_df['end_str'] = ptc_df[('end_time', 'max')].dt.strftime('%Y-%m-%d %H:%M:%S')
    ptc_df = ptc_df.drop(columns=[('start_time', 'max'), ('start_time', 'min'), ('end_time', 'min'), ('end_time', 'max')])

    patient_id = ptc_df.loc[(ptc_df['start'] > start_time) & (ptc_df['end'] < end_time),'patient_id']
    if len(patient_id.index) == 1:
        patient_id = patient_id.values[0]
    else:
        raise ValueError(f"Found more than 1 stimulus activies for patient {patient_id} between {start_time} and {end_time}")
    
    return patient_id

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
