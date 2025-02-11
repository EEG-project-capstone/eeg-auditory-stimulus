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

def get_dc_channels(raw, threshold):
    # Get the channel names and pick only DC channels
    dc_channels = [ch for ch in raw.ch_names if 'DC' in ch]
    dc_picks = mne.pick_channels(raw.ch_names, include=dc_channels)

    signal_channels = []
    # Check which DC channels exceed the threshold
    for idx in dc_picks:
        data = raw.get_data(picks=idx)  # Extract data for the channel
        if np.any(data > threshold):  # Check if any value exceeds the threshold
            signal_channels.append(raw.ch_names[idx])
    print(f"DC channels with signals: {signal_channels}")
    return signal_channels

def load_eeg(eeg_path, config):
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
    print(f"channels: {raw.info['ch_names']}")

    # Resample data
    if config.get('sfreq', False):
        print(f"Resampling data to {config['sfreq']} Hz")
        raw.resample(config['sfreq'])

    # Filter data
    lo_pass = config.get('l_freq', None)
    hi_pass = config.get('h_freq', None)    
    raw.filter(lo_pass, hi_pass)

    # Rename channels
    raw= raw.rename_channels(config['channel_map'])

    # Change DC channels types
    channel_type_mapping = {ch: 'misc' for ch in raw.info['ch_names'] if "DC" in ch}
    raw.set_channel_types(channel_type_mapping)

    # Pick channels
    missing_channels = [x for x in config['channels'] if x not in raw.info['ch_names']]
    if len(missing_channels) > 0:
        raise ValueError(f"Missing channels: {missing_channels}")
    channels = config['channels']
    dc_channel = get_dc_channels(raw, config['dc_threshold'])
    channels.extend(dc_channel)
    channels = list(set(channels))
    raw.pick(channels)
    return fname, raw, dc_channel

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

    # Check that patient only appeared once in the time range
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
        raise ValueError(f"Patient has {len(patient_id.index)} stimulus activies between {start_time} and {end_time}")
    
    return df, patient_id

def trial_start_sec(row, start_time):
    return (row['start_time']-start_time).total_seconds()

def trial_end_sec(row, start_time):
    return (row['end_time']-start_time).total_seconds()

def detect_signal_start(raw, trial_start_sec, dc_channel):
        dc_data = raw.get_data(picks=dc_channel)
        threshold = 2 * np.std(dc_data)
        exceeds_threshold = np.where(dc_data[0] > threshold)[0]
        
        # Convert trial start time (in seconds) to sample index
        start_sample = int(trial_start_sec * raw.info['sfreq'])
        
        # Find the first exceedance after the start_sample
        signal_start_samples = exceeds_threshold[exceeds_threshold > start_sample]
        
        if signal_start_samples.size > 0:
            signal_start_sample = signal_start_samples[0]
            signal_start_time = signal_start_sample / raw.info['sfreq']
            return signal_start_sample, signal_start_time
        else:
            return None, None  # No signal detected
        
def preprocess_stim(raw, stim_csv_path, config, start_time, end_time, dc_channel):
    # Pre-processing EEG data
    try:
        ptc_df, patient_id = load_stimulus(stim_csv_path, start_time, end_time)
    except ValueError as e:
        ptc_df = pd.read_csv(stim_csv_path)
        if 'Unnamed: 0' in ptc_df.columns:
            ptc_df = ptc_df.drop(columns=['Unnamed: 0'])
        ptc_df['start_time'] = pd.to_datetime(ptc_df['start_time'], unit='s', utc=True)
        ptc_df['end_time'] = pd.to_datetime(ptc_df['end_time'], unit='s', utc=True)

        patient_id = ptc_df['patient_id'].values[0]

    patient_trial = ptc_df.loc[(ptc_df['patient_id'] == patient_id) & (ptc_df['trial_type'] == config['trial_type'])]
    patient_trial['start_sec'] = patient_trial.apply(lambda x: trial_start_sec(x, start_time), axis=1)
    patient_trial['end_sec'] = patient_trial.apply(lambda x: trial_end_sec(x, start_time), axis=1)

    signal_start_times = []
    signal_start_samples = []
    for start_sec in patient_trial['start_sec']:
        signal_start_sample, signal_start_time = detect_signal_start(raw, start_sec, dc_channel)
        signal_start_times.append(signal_start_time)
        signal_start_samples.append(signal_start_sample)

    patient_trial['signal_start_time'] = signal_start_times
    patient_trial['signal_start_sample'] = signal_start_samples
    
    return patient_trial
