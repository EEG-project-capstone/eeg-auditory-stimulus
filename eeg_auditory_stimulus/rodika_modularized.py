
import os
import sys
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from scipy.fft import fft
from datetime import timedelta
import sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.save_functions import save_log, save_plot
base_dir = os.path.join("data/results/lang_tracking", patient_id)
# Ensure the directory exists before saving images
os.makedirs(base_dir, exist_ok=True)

def load_and_preprocess_eeg(eeg_file_path, use_channels=None, bad_channels=None):
    """
    Loads and preprocesses EEG data from an EDF file
    
    Parameters:
    -----------
    eeg_file_path : str
        Path to the EEG EDF file
    use_channels : list or None
        List of channels to keep. If None, all channels are kept
    bad_channels : list or None
        List of channels to mark as bad
    
    Returns:
    --------
    raw : mne.io.Raw
        Preprocessed Raw EEG data
    """
    # Load EEG data
    raw = mne.io.read_raw(eeg_file_path, preload=True)
    
    # Define non-EEG channels to drop
    non_eeg_channels = ['IO1', 'IO2', 'EMG1', 'EMG2', 'ECGL', 'ECGR', 'LAT1', 'LAT2', 
                         'RAT1', 'RAT2', 'RESP', 'ABD', 'FLOW', 'SNORE', 'DIF5','DIF6', 
                         'POS', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6', 'DC7','DC8','DC9',
                         'DC10', 'OSAT', 'PR']
    
    # Drop non-EEG channels
    raw.drop_channels(non_eeg_channels)
    
    # Select specific channels if provided
    if use_channels is not None:
        raw.pick_channels(use_channels)
    
    # Mark bad channels if provided
    if bad_channels is not None:
        raw.info["bads"].extend(bad_channels)
    
    # Apply filters
    # Notch filter at specific frequencies
    raw.notch_filter(freqs=[48, 50, 52], filter_length='auto', picks='eeg')
    raw.notch_filter(freqs=[98, 100, 102], filter_length='auto', picks='eeg')
    raw.notch_filter(freqs=[148, 150, 152], filter_length='auto', picks='eeg')
    
    # Band pass filter between 0.01 and 100 hz
    raw.filter(l_freq=0.01, h_freq=100)
    
    return raw

def load_stimulus_data(csv_path, patient_id, trial_type='lang', timezone_offset=8):
    """
    Loads stimulus timing data from a CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing stimulus data
    patient_id : str
        ID of the patient to filter by
    trial_type : str
        Type of trial to filter by (default: 'lang')
    timezone_offset : int
        Number of hours to subtract from timestamps to align with EEG data
    
    Returns:
    --------
    stimulus_df : pandas.DataFrame
        DataFrame containing stimulus data with timestamps
    """
    # Load CSV with stimulus onset times
    stimulus_df = pd.read_csv(csv_path)
    
    # Clean the data
    if 'Unnamed: 0' in stimulus_df.columns:
        stimulus_df = stimulus_df.drop(columns=['Unnamed: 0'])
    
    # Filter by patient ID and trial type
    stimulus_df = stimulus_df[stimulus_df['patient_id'].isin([patient_id])]
    stimulus_df = stimulus_df[stimulus_df['trial_type'] == trial_type]
    
    # Convert Unix timestamps to datetime
    stimulus_df["start_time"] = pd.to_datetime(stimulus_df["start_time"], unit="s")
    stimulus_df["end_time"] = pd.to_datetime(stimulus_df["end_time"], unit="s")
    
    # Adjust for timezone difference
    stimulus_df["start_time"] = stimulus_df["start_time"] - pd.Timedelta(hours=timezone_offset)
    stimulus_df["end_time"] = stimulus_df["end_time"] - pd.Timedelta(hours=timezone_offset)
    
    return stimulus_df

def align_stimulus_to_eeg(stimulus_df, raw):
    """
    Aligns stimulus timing data with EEG recording
    
    Parameters:
    -----------
    stimulus_df : pandas.DataFrame
        DataFrame containing stimulus data with timestamps
    raw : mne.io.Raw
        Raw EEG data object
    
    Returns:
    --------
    stimulus_df : pandas.DataFrame
        Updated DataFrame with sample indices
    """
    # Get EEG recording start time
    eeg_start_time = raw.info["meas_date"]
    eeg_start_time = pd.Timestamp(eeg_start_time)
    eeg_start_time = eeg_start_time.replace(tzinfo=None)
    
    # Compute relative times to find distance from EEG start times
    stimulus_df["relative_start_time"] = (stimulus_df["start_time"] - eeg_start_time).dt.total_seconds()
    stimulus_df["relative_end_time"] = (stimulus_df["end_time"] - eeg_start_time).dt.total_seconds()
    
    # Get sampling frequency
    sfreq = raw.info["sfreq"]
    
    # Calculate sample indices to match with EEG recording
    stimulus_df["start_sample"] = (stimulus_df["relative_start_time"] * sfreq).astype(int)
    stimulus_df["end_sample"] = (stimulus_df["relative_end_time"] * sfreq).astype(int)
    
    return stimulus_df

def create_events_from_stimulus(raw, stimulus_df, event_description='lang'):
    """
    Creates MNE events from stimulus data
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    stimulus_df : pandas.DataFrame
        DataFrame containing stimulus timing data with sample indices
    event_description : str
        Description for the events
    
    Returns:
    --------
    events : ndarray
        MNE events array
    event_id : dict
        Dictionary mapping event descriptions to event codes
    """
    # Extract start samples
    start_samples = np.array(stimulus_df["start_sample"])
    
    # Create annotations for each trial
    annotations = []
    for i, start_sample in enumerate(start_samples):
        annotations.append(mne.Annotations(
            onset=start_sample / raw.info['sfreq'],  # onset in seconds
            duration=stimulus_df['duration'].iloc[i],  # duration in seconds
            description=event_description))  # Event description
    
    # Add annotations to the Raw object
    raw.set_annotations(mne.Annotations(
        onset=np.ravel([a.onset for a in annotations]),
        duration=np.ravel([a.duration for a in annotations]),
        description=np.ravel([a.description for a in annotations])
    ))
    
    # Create events based on the annotations
    events, event_id = mne.events_from_annotations(raw)
    
    return events, event_id

def create_epochs(raw, events, tmin=-1, tmax=None, baseline=None, resample_freq=None):
    """
    Creates epochs from continuous EEG data
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    events : ndarray
        MNE events array
    tmin : float
        Start time of epochs in seconds relative to events
    tmax : float or None
        End time of epochs in seconds relative to events
    baseline : tuple or None
        Baseline correction period
    resample_freq : float or None
        Frequency to resample data to
    
    Returns:
    --------
    epochs : mne.Epochs
        Epoched data
    """
    # If tmax is not specified, use default of 12 sentences * 1.28s each
    if tmax is None:
        tmax = 12 * 1.28  # 12 sentences * 1.28s each = 15.36
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, 
                        baseline=baseline, preload=True)
    
    # Inherit bad channels from the raw data
    epochs.info['bads'] = raw.info['bads']
    
    # Resample if specified
    if resample_freq is not None:
        epochs.resample(resample_freq)
        raw.resample(resample_freq)
    
    return epochs

def apply_ica(data, n_components=15, eog_ch='Fpz', threshold=1):
    """
    Applies Independent Component Analysis to detect and remove artifacts
    
    Parameters:
    -----------
    data : mne.io.Raw or mne.Epochs
        EEG data to apply ICA to
    n_components : int
        Number of ICA components to compute
    eog_ch : str
        Channel name to use for EOG artifact detection
    threshold : float
        Threshold for detecting EOG components
    
    Returns:
    --------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    cleaned_data : mne.io.Raw or mne.Epochs
        Cleaned data with artifacts removed
    """
    
    # Set up and fit ICA
    ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
    ica.fit(data)
    
    # Find eye movement components
    eog_components, eog_scores = ica.find_bads_eog(
        inst=data,
        ch_name=eog_ch,
        threshold=threshold
    )
    
    # Mark components for exclusion
    ica.exclude = eog_components
    
    # Apply ICA to remove artifacts
    cleaned_data = data.copy()
    ica.apply(cleaned_data)
    
    return ica, cleaned_data

def setup_montage_and_reference(epochs, montage_name='standard_1020', ref_type='average'):
    """
    Sets up electrode montage and re-references data
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    montage_name : str
        Name of the standard montage to use
    ref_type : str or list
        Reference type ('average' or list of channel names)
    
    Returns:
    --------
    epochs : mne.Epochs
        Epoched data with montage and reference applied
    """
    # Set up montage
    montage = mne.channels.make_standard_montage(montage_name)
    epochs.set_montage(montage)
    
    # Interpolate bad channels
    epochs.interpolate_bads()
    
    # Re-reference
    epochs.set_eeg_reference(ref_type, projection=True)
    epochs.apply_proj()
    
    return epochs

def prepare_data_for_itpc(epochs, low_freq=0.1, high_freq=25, crop_tmin=None):
    """
    Prepares epoched data for ITPC calculation by filtering and cropping
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched EEG data
    low_freq : float
        Lower cutoff frequency for bandpass filter
    high_freq : float
        Upper cutoff frequency for bandpass filter
    crop_tmin : float or None
        Time to crop from the beginning of epochs (in seconds)
    
    Returns:
    --------
    epochs : mne.Epochs
        Prepared epochs
    epochs_data : ndarray
        Epoched data as a numpy array
    """
    # Filter data
    epochs.filter(l_freq=low_freq, h_freq=high_freq, method='iir')
    
    # Crop data if specified
    if crop_tmin is not None:
        epochs.crop(tmin=crop_tmin)
    
    # Get data as array
    epochs_data = epochs.get_data()
    
    return epochs, epochs_data

def compute_itpc(epochs_data, fs=256):
    """
    Compute Inter-Trial Phase Coherence from epoched data
    
    Parameters:
    -----------
    epochs_data : ndarray
        Shape (n_trials, n_electrodes, n_samples)
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    itpc : ndarray
        Shape (n_frequencies, n_electrodes)
    freqs : ndarray
        Frequency values
    phase_data : ndarray
        Complex phase data with shape (n_frequencies, n_electrodes, n_trials)
    """
    # Transpose to match MATLAB's format (time x channel x trial)
    data = np.transpose(epochs_data, (2, 1, 0))  # (n_samples, n_electrodes, n_trials)
    
    n_samples, n_electrodes, n_trials = data.shape
    
    # Define all frequencies
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    
    # Initialize arrays
    itpc = np.zeros((n_samples, n_electrodes))
    phase_data = np.zeros((n_samples, n_electrodes, n_trials), dtype=complex)
    
    # ITPC loop -> 1 ITPC value per electrode and frequency
    for el in range(n_electrodes):
        # Initialize array for complex exponentials
        exp_k = np.zeros((n_samples, n_trials), dtype=complex)
        
        for tr in range(n_trials):
            # Get data for current channel and trial
            x_chan_trl = data[:, el, tr]
            
            # Compute DFT
            y = fft(x_chan_trl)
            
            # Get phase angles
            p = np.angle(y)
            
            # Compute complex exponential of phase (e^(i*p))
            exp_k[:, tr] = np.exp(1j * p)
            phase_data[:, el, tr] = np.exp(1j * p)
        
        print(f"Computing channel #{el+1}")
        
        # Compute ITPC by taking absolute value of mean across trials
        itpc[:, el] = np.abs(np.mean(exp_k, axis=1))
    
    return itpc, freqs, phase_data

def plot_itpc_each_channel(itpc, freqs, ch_names, base_dir=None,fmin=0.5, fmax=4.0):
    """
    Plot ITPC values for each channel in separate figures, optionally saving them
    via save_plot(), with shaded bands for specific frequencies.

    Parameters
    ----------
    itpc : ndarray
        ITPC array of shape (n_frequencies, n_channels).
    freqs : ndarray
        Frequency values (can include both negative and positive frequencies).
    ch_names : list
        List of channel names.
    patient_id : str or None
        Patient identifier, used to create a folder if saving plots.
    base_dir : str or None
        Base directory for saving plots with save_plot(). If None, figures are shown but not saved.
    fmin : float
        Minimum frequency to display on the x-axis.
    fmax : float
        Maximum frequency to display on the x-axis.
    """
    # Frequency bands to highlight (±0.04 Hz)
    band_width = 0.04
    highlight_data = [
        ('teal',  0.78 - band_width,  0.78 + band_width,  '≈0.78 Hz'),
        ('magenta',  1.56 - band_width,  1.56 + band_width,  '≈1.56 Hz'),
        ('red',  3.125 - band_width, 3.125 + band_width, '≈3.125 Hz')
    ]
    
    pos_idx = (freqs >= 0) & (freqs <= fmax)
    plot_freqs = freqs[pos_idx]
    n_channels = itpc.shape[1]
    
    for ch in range(n_channels):
        channel_itpc = itpc[pos_idx, ch]
        
        # Create new figure for each channel
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(plot_freqs, channel_itpc, color='b', label='ITPC')
        
        if ch_names:
            title = f'ITPC for {ch_names[ch]}'
            filename = f'ITPC_{ch_names[ch]}'  # We'll pass this to save_plot
        else:
            title = f'ITPC for Channel {ch+1}'
            filename = f'ITPC_Channel_{ch+1}'
        
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('ITPC')
        ax.set_xlim(fmin, fmax)
        ax.grid(True)

        # Add shaded bands for each target frequency
        for color, f_start, f_end, label in highlight_data:
            # Only shade if the band is within our displayed range
            if f_end >= fmin and f_start <= fmax:
                ax.axvspan(max(f_start, fmin), min(f_end, fmax),
                           color=color, alpha=0.2, label=label)

        ax.legend(loc='upper right')
        

        # Save or show
        if base_dir is not None:
            plot_path = save_plot(
                base_dir=base_dir, 
                fig=fig, 
                filename=filename
            )
            print(f"Saved {plot_path}")
        else:
            plt.show()

        plt.close(fig)  # Close the figure to free memory after each loop


def plot_itpc_avg(itpc, freqs, base_dir=None, 
                  fmin=0.5, fmax=4.0):
    """
    Plot the average Inter-Trial Phase Coherence (ITPC) across electrodes 
    for frequencies in a specified range, optionally saving the figure,
    including shaded bands for specific frequencies.

    Parameters
    ----------
    itpc : ndarray
        2D array of shape (n_frequencies, n_electrodes) containing the ITPC 
        values across frequencies and electrodes.
    freqs : ndarray
        1D array of frequency values (same length as itpc's first dimension).
    patient_id : str or None
        Patient identifier, used to create a folder if saving plots.
    base_dir : str or None
        Base directory for saving plots with save_plot(). If None, figures are shown but not saved.
    fmin : float
        Minimum frequency to display on the x-axis.
    fmax : float
        Maximum frequency to display on the x-axis.

    Returns
    -------
    avg_itpc : ndarray
        1D array of shape (n_selected_frequencies,) representing the average 
        ITPC values across electrodes within the specified frequency range.
    plot_freqs : ndarray
        1D array of the frequency values corresponding to the selected
        frequency range.
    """   
    # Frequency bands to highlight (±0.04 Hz)
    band_width = 0.04
    highlight_data = [
        ('teal',  0.78 - band_width,  0.78 + band_width,  '≈0.78 Hz'),
        ('magenta',  1.56 - band_width,  1.56 + band_width,  '≈1.56 Hz'),
        ('red',  3.125 - band_width, 3.125 + band_width, '≈3.125 Hz')
    ]
    
    pos_idx = (freqs >= 0) & (freqs <= fmax)
    plot_freqs = freqs[pos_idx]
    plot_itpc = itpc[pos_idx, :]
    
    # Average ITPC across electrodes
    avg_itpc = np.mean(plot_itpc, axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_freqs, avg_itpc, label='Avg ITPC', color='b')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('ITPC')
    ax.set_title('Inter-Trial Phase Coherence (Averaged)')
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(0.01, 0.5)
    ax.grid(True)

    # Add shaded bands for each target frequency
    for color, f_start, f_end, label in highlight_data:
        if f_end >= fmin and f_start <= fmax:
            ax.axvspan(max(f_start, fmin), min(f_end, fmax),
                       color=color, alpha=0.2, label=label)

    ax.legend(loc='upper right')

    # Save or show
    if base_dir is not None:
        filename = "avg_itpc_plot"
        plot_path = save_plot(
            base_dir=base_dir, 
            fig=fig, 
            filename=filename
        )
        print(f"Saved {plot_path}")
    else:
        plt.show()
    
    plt.close(fig)
    return avg_itpc, plot_freqs

def main(eeg_file_path, stimulus_csv_path, patient_id, use_channels, bad_channels, eog_chs, output_dir=None):
    """
    Main function orchestrating the entire EEG analysis pipeline: 
    loading, preprocessing, stimulus alignment, epoching, ICA, 
    referencing, ITPC computation, and plotting.
    """
    
    # 2. Load and preprocess EEG data
    raw = load_and_preprocess_eeg(
        eeg_file_path=eeg_file_path,
        use_channels=use_channels,
        bad_channels=bad_channels
    )
    
    # 3. Load and align stimulus data
    stimulus_df = load_stimulus_data(
        csv_path=stimulus_csv_path,
        patient_id=patient_id,
        trial_type='lang',       # or another trial type
        timezone_offset=8        # offset in hours
    )
    stimulus_df = align_stimulus_to_eeg(stimulus_df, raw)
    
    # 4. Create events and epochs
    events, event_id = create_events_from_stimulus(
        raw=raw, 
        stimulus_df=stimulus_df,
        event_description='lang'
    )
    epochs = create_epochs(
        raw=raw,
        events=events,
        tmin=-1,
        tmax=12 * 1.28,  # 12 sentences * 1.28s each = 15.36
        baseline=None,
        resample_freq=256
    )
    
    # 5. Apply ICA (first to continuous data, then apply to epoched data)
    ica, _ = apply_ica(
        data=raw,   # Usually we fit ICA on continuous data
        n_components=15,
        eog_ch= eog_chs,
        threshold=1
    )
    epochs_clean = epochs.copy()
    ica.apply(epochs_clean)  # Apply same ICA to epoched data
    
    # 6. Montage & Re-reference
    epochs_clean = setup_montage_and_reference(
        epochs=epochs_clean,
        montage_name='standard_1020',
        ref_type='average'
    )
    
    # 7. Prepare data for ITPC (filter to 0.1-25 Hz, crop first 1.28s)
    epochs_clean, epochs_data = prepare_data_for_itpc(
        epochs=epochs_clean,
        low_freq=0.1,
        high_freq=25,
        crop_tmin=1.28
    )
    
    # 8. Compute ITPC
    fs = 256  # sampling rate
    itpc, freqs, phase_data = compute_itpc(epochs_data=epochs_data, fs=fs)
    
    # 9. Plot results (individual channels + average)
    plot_itpc_each_channel(
    itpc=itpc,
    freqs=freqs,
    ch_names=epochs_clean.info['ch_names'],
    fmin=0.5,
    fmax=4,
    base_dir=base_dir
    )

    plot_itpc_avg(
    itpc=itpc,
    freqs=freqs,
    fmin=0.5,
    fmax=4,
    base_dir=base_dir
    )
    
if __name__ == "__main__":
    # Entry point to run the main function
    eeg_file_path = "/Users/trishaprasant/Documents/DATA590/CON004_clipped.EDF"
    stimulus_csv_path = "/Users/trishaprasant/Documents/DATA590/patient_df.csv"
    patient_id = "CON004"
    use_channels = ['C3','C4','O1','O2','FT9','FT10','Cz','F3','F4','F7','F8',
                    'Fz','Fp1','Fp2','Fpz','P3','P4','Pz','T7','T8','P7','P8']
    bad_channels = ['T7', 'Fp1', 'Fp2']
    eog_chs = ['Fp1', 'Fp2', 'T7']

    main(eeg_file_path, stimulus_csv_path, patient_id, use_channels, bad_channels, eog_chs)

'''
EXAMPLE USAGE:

# Define main paths and parameters
eeg_file_path = "CON003_clipped.EDF"
stimulus_csv_path = "patient_df.csv"
patient_id = "CON003"

# Step 1: Load and preprocess EEG data
use_channels = ['C3','C4','O1','O2','FT9','FT10','Cz','F3','F4','F7','F8','Fz',
                'Fp1','Fp2','Fpz','P3','P4','Pz','T7','T8','P7','P8']

bad_channels = ['Fp1', 'Fp2', 'T7', 'F8', 'F7']

raw = load_and_preprocess_eeg(
    eeg_file_path=eeg_file_path,
    use_channels=use_channels,
    bad_channels=bad_channels
)

# Step 2: Load and align stimulus data
stimulus_df = load_stimulus_data(
    csv_path=stimulus_csv_path,
    patient_id=patient_id,
    trial_type='lang',
    timezone_offset=8
)

stimulus_df = align_stimulus_to_eeg(stimulus_df, raw)

# Step 3: Create events and epochs
events, event_id = create_events_from_stimulus(raw, stimulus_df, event_description='lang')
epochs = create_epochs(
    raw=raw,
    events=events,
    tmin=-1,  # Start 1 second before stimulus
    tmax=12 * 1.28,  # 12 sentences * 1.28s each
    baseline=None,
    resample_freq=256  # Downsample to 256 Hz
)

# Step 4: Apply ICA for artifact rejection
ica, epochs_clean = apply_ica(
    data=raw,  # Apply to continuous data first
    n_components=15,
    eog_ch='Fpz',
    threshold=1
)

# Apply ICA to epoched data
epochs_clean = epochs.copy()
ica.apply(epochs_clean)

# Step 5: Setup montage and re-reference
epochs_clean = setup_montage_and_reference(
    epochs=epochs_clean,
    montage_name='standard_1020',
    ref_type='average'
)

# Step 6: Prepare data for ITPC analysis
epochs_clean, epochs_data = prepare_data_for_itpc(
    epochs=epochs_clean,
    low_freq=0.1,
    high_freq=25,
    crop_tmin=1.28  # Skip first part to avoid transient responses
)

# Step 7: Compute ITPC
itpc, freqs, phase_data = compute_itpc(
    epochs_data=epochs_data,
    fs=256  # Sampling frequency
)

# Step 8: Visualize results
plot_itpc_each_channel(
    itpc=itpc,
    freqs=freqs,
    ch_names=epochs_clean.info['ch_names'],
    fmin=0.5,
    fmax=4,
    out_dir = "plots"
)

plot_itpc_avg(
    itpc=itpc,
    freqs=freqs,
    fmin=0.5,
    fmax=4,
    out_dir = "plots"
)

'''