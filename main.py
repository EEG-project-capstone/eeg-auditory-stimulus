import os
import yaml
import argparse
from data.eeg_loader import load_eeg, get_eeg_timestamps

def main(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    eeg_file = os.listdir(config['raw_path'])[0]
    eeg_full_path = os.path.join(config['raw_path'], eeg_file)
    _, raw = load_eeg(eeg_full_path, config)
    if config['verbose']:
        print(raw.info)

    start_time, end_time = get_eeg_timestamps(raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    main(args.config)
