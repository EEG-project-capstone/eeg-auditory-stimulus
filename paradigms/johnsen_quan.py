import os
import yaml
import argparse
import mne
import os
import numpy as np
import pandas as pd
import random
from datetime import timedelta
import matplotlib
import matplotlib.pyplot  as plt
import matplotlib.dates as mdates
import PyQt5
matplotlib.use('Qt5Agg')

def get_johnsen_events(trial_start_time, trial_end_time, signal_start_time, protocol_start_time, sfreq, verbose=False):
    events = []

    signal_start = pd.Timestamp(protocol_start_time) + pd.to_timedelta(signal_start_time, unit="s")
    
    # Extract reference epoch
    if verbose:
        print(f"Protocol start time: {protocol_start_time}")
        print(f"Patient trial start time: {trial_start_time}")
    stimulation_start_time = trial_start_time + timedelta(seconds=10)
    reference_start_time = stimulation_start_time - timedelta(seconds=2.5) #referece epoch
    reference_start_sample = int((reference_start_time - protocol_start_time).total_seconds() * sfreq)
    events.append((reference_start_sample,0,0)) 

    offset = random.uniform(0.5, 2.5)
    active_start_time = stimulation_start_time + timedelta(seconds=offset)
    active_end_time = active_start_time + timedelta(seconds=10)
    active_signal_start_time =  max(active_start_time, signal_start)
    curr_epoch_start_time = active_signal_start_time
    curr_epoch_end_time = curr_epoch_start_time + timedelta(seconds=2)
    i = 1
    while curr_epoch_end_time < active_end_time:
        curr_epoch_start_sample = int((curr_epoch_start_time - protocol_start_time).total_seconds() * sfreq)
        events.append((curr_epoch_start_sample,i,1))
        curr_epoch_start_time = curr_epoch_start_time + timedelta(seconds=1)
        curr_epoch_end_time = curr_epoch_start_time + timedelta(seconds=2)
        i+=1

    return np.array(events)


def plot_epochs(events, start_time, sampling_rate, beep_start_time):
    # Function to convert sample number to time
    def sample_to_time(sample, start_time, sampling_rate):
        return start_time + pd.to_timedelta(sample / sampling_rate, unit='s')

    # Create a time array corresponding to each event sample
    event_times = [sample_to_time(sample, start_time, sampling_rate) for sample in events[-9:, 0]]

    # Create a plot to visualize the events
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each event
    for i, (event_time, event_type) in enumerate(zip(event_times, events[:, 2])):
        # Use event_type (0 or 1) to determine the color/label
        label = "Silence" if event_type == 0 else "Beep"
        color = "b" if event_type == 0 else "r"
        
        # Offset each event to avoid overlap in the plot
        ax.plot([event_time, event_time + pd.to_timedelta(2, unit='s')],
                [i, i], color=color, label=label if i == 0 else "")

    # Add a vertical dashed line at the stimulation time (e.g., start time)
    reference_time = beep_start_time + timedelta(seconds=10)
    ax.axvline(reference_time, color='green', linestyle='--', label="Stimulation start")

    # Set the date format for x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=2))
    fig.autofmt_xdate()

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Epochs')
    ax.set_title('Epochs Visualization')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def get_johnsen_epochs(raw, patient_trial, start_time, config):
    events = []
    for id, row in patient_trial.iterrows():
        events.extend(get_johnsen_events(row['start_time'], row['end_time'], row['signal_start_time'], start_time, config['sfreq']))
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=None, preload=True)
    active_epochs = mne.Epochs(raw, events, event_id=1, tmin=config['tmin'], tmax=config['tmax'], picks='eeg', baseline=None, preload=True)
    reference_epochs = mne.Epochs(raw, events, event_id=0, tmin=config['tmin'], tmax=config['tmax'], picks='eeg', baseline=None, preload=True)
    return events, epochs, active_epochs, reference_epochs

def compute_log_band_power_avg(active_psd, active_epochs, freq_bands):
    # Initialize a dictionary to hold the log-transformed average power per epoch for each band
    log_band_power_avg = {band: [] for band in freq_bands.keys()}

    # Calculate log-transformed average power in each frequency band for each epoch
    for band, (low, high) in freq_bands.items():
        band_log_power_per_epoch = []

        # Iterate through all epochs
        for psd_epoch in active_psd:
            band_power_all_channels = []

            # For each epoch, get the data and compute the power for each channel
            for i in range(len(active_epochs.ch_names)):  # Loop through all channels
                # Find indices of frequencies within the band
                band_idx = np.where((active_psd.freqs >= low) & (active_psd.freqs <= high))

                # Compute average power in the frequency band for the current channel
                band_power_value = np.mean(psd_epoch[i][band_idx])
                band_power_all_channels.append(band_power_value)

            # Compute log-transform of the average power across all channels
            avg_band_power_epoch = np.mean(band_power_all_channels)
            log_avg_band_power_epoch = np.log(avg_band_power_epoch)

            # Store the log-transformed average power for this epoch
            band_log_power_per_epoch.append(log_avg_band_power_epoch)

        # Store the log-transformed average power for each epoch in the band
        log_band_power_avg[band] = np.array(band_log_power_per_epoch)

    return log_band_power_avg

def plot_log_transformed_power(active_log_band_power_avg, output_path=None, verbose=False):
    # Plot the log-transformed power for each frequency band across epochs
    colors = ['b', 'orange', 'r', 'c']
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, (band, log_powers) in enumerate(active_log_band_power_avg.items()):
        axs[i].plot(range(len(log_powers)), log_powers, label=f'{band.capitalize()} Band', marker='o', color=colors[i])
        axs[i].set_title(f'Log-Transformed Average Power in {band.capitalize()} Band Across Epochs')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Log Power (µV²/Hz)')
        axs[i].grid(True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    if verbose:
        plt.show()

def compute_z_scores_for_bands(ref_log_band_power_avg, active_log_band_power_avg, freq_bands):
    z_score_results = {}

    # Iterate over frequency bands
    for band in freq_bands.keys():
        # Get mean and SD for the resting state in the current band
        rest_mean = np.mean(ref_log_band_power_avg[band])
        rest_std = np.std(ref_log_band_power_avg[band])

        # Compute Z-scores for the stimulation epochs
        stim_z_scores = (active_log_band_power_avg[band] - rest_mean) / rest_std

        # Identify significant increases and decreases
        significant_increase = [z > 1.96 for z in stim_z_scores]
        significant_decrease = [z < -1.96 for z in stim_z_scores]

        z_score_results[band] = {
            'z_scores': stim_z_scores,
            'significant_increase': significant_increase,
            'significant_decrease': significant_decrease
        }

    return z_score_results

# Plotting Z-scores for each frequency band
def plot_z_scores(z_score_results, epoch_ids=None, first_n_epochs=0, output_path=None, verbose=False):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    shaded_patch = None

    for idx, (band, results) in enumerate(z_score_results.items()):
        z_scores = results['z_scores']
        significant_increase = results['significant_increase']
        significant_decrease = results['significant_decrease']

        time = np.arange(len(z_scores))  # Assuming epochs are sequential and evenly spaced

        axes[idx].plot(time, z_scores, label='Z-scores', color='blue')
        axes[idx].axhline(1.96, color='green', linestyle='--', label='Significant threshold (+1.96)')
        axes[idx].axhline(-1.96, color='red', linestyle='--', label='Significant threshold (-1.96)')

        # Highlight significant points
        axes[idx].scatter(time[significant_increase], z_scores[significant_increase], color='green', label='Significant increase')
        axes[idx].scatter(time[significant_decrease], z_scores[significant_decrease], color='red', label='Significant decrease')

        if epoch_ids is not None and first_n_epochs > 0:
            shaded_indices = np.where(epoch_ids <= first_n_epochs)[0]  # Get indices where condition is met
            # Find continuous segments
            segments = np.split(shaded_indices, np.where(np.diff(shaded_indices) > 1)[0] + 1)
            # Loop through segments and shade
            for segment in segments:
                if len(segment) > 0:
                    axes[idx].axvspan(segment[0], segment[-1], color='blue', alpha=0.2)

            shaded_patch = mpatches.Patch(color='blue', alpha=0.2, label=f"First {first_n_epochs} epochs")
                
        axes[idx].set_title(f"{band} band", fontsize=15)
        axes[idx].set_xlabel("Time (epochs)", fontsize=12)
        axes[idx].set_ylabel("Z-score", fontsize=12)
        
    handles, labels = axes[3].get_legend_handles_labels()
    if shaded_patch:
        handles.append(shaded_patch)
        labels.append(shaded_patch.get_label())
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.45, 0.03, 0.05, 0.05))
    plt.tight_layout(rect=[0, 0.9, 0, 1.02])
    plt.suptitle("Z-scores of EEG Frequency Bands After Sound Stimulation", fontsize=15, y=0.95)
    if output_path:
        plt.savefig(output_path)
    if verbose:
        plt.show()