import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse
import mne
from data.eeg_loader import load_eeg, load_stimulus, get_eeg_timestamps
from data.eeg_loader import trial_start_sec, trial_end_sec, detect_signal_start
from paradigms.johnsen_quan import get_johnsen_epochs_arr, plot_epochs
from paradigms.johnsen_quan import compute_log_band_power_avg, compute_z_scores_for_bands

def main(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    eeg_file = os.listdir(config['raw_path'])[0]
    eeg_full_path = os.path.join(config['raw_path'], eeg_file)
    _, raw, dc_channel = load_eeg(eeg_full_path, config)
    if config['verbose']:
        print(raw.info)

    start_time, end_time = get_eeg_timestamps(raw)
    protocol_path = os.path.join(config['protocol_path'])
    ptc_df, patient_id = load_stimulus(protocol_path, start_time, end_time)
    patient_trial = ptc_df.loc[(ptc_df['patient_id'] == patient_id) & (ptc_df['trial_type']==config['trial_type'])]
    print(f"Patient ID: {patient_id}")

    patient_trial['start_sec'] = patient_trial.apply(lambda x: trial_start_sec(x, start_time), axis=1)
    patient_trial['end_sec'] = patient_trial.apply(lambda x: trial_end_sec(x, start_time), axis=1)
    print(f"patient_trial: {patient_trial}")

    print(f"Detecting signal start from channel {dc_channel}")
    if config['dc_channel']:
        assert config['dc_channel'].lower() == dc_channel.lower(), \
        f"Config dc_channel {config['dc_channel']} does not match detected dc_channel {dc_channel}"
    signal_start_times = []
    signal_start_samples = []
    for start_sec in patient_trial['start_sec']:
        signal_start_sample, signal_start_time = detect_signal_start(raw, dc_channel, start_sec)
        signal_start_times.append(signal_start_time)
        signal_start_samples.append(signal_start_sample)
    patient_trial.loc[:,'signal_start_time'] = signal_start_times
    patient_trial.loc[:,'signal_start_sample'] = signal_start_samples

    events = []
    for _, row in patient_trial.iterrows():
        events.extend(get_johnsen_epochs_arr(row['start_time'], row['end_time'], row['signal_start_time'], start_time, config['sfreq']))
    
    active_epochs = mne.Epochs(
        raw, events, picks=['eeg'], event_id=1, tmin=config['tmin'], tmax=config['tmax'], baseline=None)
    ref_epochs = mne.Epochs(
        raw, events, picks=['eeg'], event_id=0, tmin=config['tmin'], tmax=config['tmax'], baseline=None)

    if config['verbose']:
        plot_epochs(events, start_time, config['sfreq'], row)

        # Set the parameters
    sfreq = int(active_epochs.info['sfreq'])  # Sampling frequency
    n_fft = sfreq*config['fft_rate']  # Length of FFT

    # Compute PSDs for resting and stimulation epochs
    ref_psd = ref_epochs.compute_psd(method='welch', n_fft=n_fft, n_overlap=n_fft // 2, 
                                    window='hamming', fmin=config['l_freq'], fmax=config['h_freq'])
    active_psd = active_epochs.compute_psd(method='welch', n_fft=n_fft, n_overlap=n_fft // 2, 
                                    window='hamming', fmin=config['l_freq'], fmax=config['h_freq'])

    # Calculate log-transformed average band power for resting and stimulation epochs
    ref_log_band_power_avg = compute_log_band_power_avg(ref_psd, ref_epochs, config['freq_bands'])
    active_log_band_power_avg = compute_log_band_power_avg(active_psd, active_epochs, config['freq_bands'])

    # Print the log-transformed band power for each frequency band across epochs
    if config['verbose']:
        for band, log_powers in active_log_band_power_avg.items():
            print(f"\nLog-transformed average power for {band} band across epochs:")
            print(log_powers)

    # Plot the log-transformed power for each frequency band across epochs
    colors = ['b', 'orange', 'r', 'c']
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, (band, log_powers) in enumerate(active_log_band_power_avg.items()):
        axs[i].plot(range(len(log_powers)), log_powers, label=f'{band.capitalize()} Band', marker='o',color=colors[i])
        axs[i].set_title(f'Log-Transformed Average Power in {band.capitalize()} Band Across Epochs')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Log Power (µV²/Hz)')
        axs[i].grid(True)
        # axs[i].legend()

    plt.tight_layout()
    plt.show()

    # Compute Z-scores for stimulation epochs
    z_score_results = compute_z_scores_for_bands(ref_log_band_power_avg, active_log_band_power_avg, config['freq_bands'])
    # print([x['z_scores'].shape for x in z_score_results.values()])

    # Print the results
    # if config['verbose']:
    for band, results in z_score_results.items():
        print(f"\nBand: {band}")
        # print(f"Z-scores: {results['z_scores']}")
        print(f"Significant increases: {np.sum(results['significant_increase'])}")
        print(f"Significant decreases: {np.sum(results['significant_decrease'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    main(args.config)
