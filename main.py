import os
import yaml
import argparse
from data.eeg_loader import load_eeg, load_stimulus, get_eeg_timestamps
from data.eeg_loader import trial_start_sec, trial_end_sec, detect_signal_start

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
    print(f"Number of trials: {len(patient_trial.index)}")

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
    patient_trial['signal_start_time'] = signal_start_times
    patient_trial['signal_start_sample'] = signal_start_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    main(args.config)
