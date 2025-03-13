import matplotlib.pyplot as plt
import matplotlib
import yaml
import sys
import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneGroupOut
from mne.time_frequency import psd_array_multitaper
from mne.decoding import LinearModel, get_coef

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.eeg_loader import load_eeg, get_eeg_timestamps, load_stimulus, detect_signal_start
from preprocessing.save_functions import save_log, save_plot

# Set the backend for Matplotlib
matplotlib.use('Agg') #Qt5Agg

# # Load Configuration
# Get the absolute path of the project root directory
# CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'claassen_cfg.yml')
# CONFIG_PATH = './configs/claassen_cfg.yml'
CONFIG_PATH = '/Users/joobeejung/eeg-auditory-stimulus/configs/claassen_cfg.yml'

def load_config():
    """Load the configuration file."""
    with open(CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

# pip install --upgrade --force-reinstall git+https://github.com/EEG-project-capstone/eeg-auditory-stimulus.git@jb-modules


config = load_config()

# Load EEG data
def load_eeg_data(subject_id=None, edf_path=None):
    """Load the EEG data for a given subject."""
    eeg_path = edf_path if edf_path else config.get(f"{subject_id}_path", "")
    
    if not eeg_path or not os.path.exists(eeg_path):
        raise FileNotFoundError(f"File not found: {eeg_path}")
    
    raw = mne.io.read_raw_edf(eeg_path, preload=True)
    raw.filter(l_freq=1, h_freq=30)  # Band-pass filter

    fname, raw, dc_channel = load_eeg(eeg_path, config, subject_id)
    return raw, dc_channel, subject_id

# Process trials and segment into epochs
def process_trials(raw, subject_id, dc_channel):
    start_time, end_time = get_eeg_timestamps(raw, subject_id)
    df, patient_id = load_stimulus(config["event_full_path"], start_time, end_time)
    
    patient_trial = df.loc[(df['patient_id'] == patient_id) & (df['trial_type'].isin(config['trial_type']))]

    patient_trial['start_sec'] = patient_trial.apply(lambda row: (row['start_time'] - start_time).total_seconds(), axis=1)
    patient_trial['end_sec'] = patient_trial.apply(lambda row: (row['end_time'] - start_time).total_seconds(), axis=1)

    expanded_rows = []
    for _, row in patient_trial.iterrows():
        trial_start_sec = row['start_sec']
        for i in range(8):
            signal_start_sample, signal_start_time = detect_signal_start(raw, trial_start_sec, dc_channel)
            new_row = row.copy()
            new_row['start_sample'] = signal_start_sample
            new_row['start_sec'] = signal_start_time
            if i % 2 == 0:
                new_row['trial_type'] = f"{row['trial_type']}-keep"
            else:
                new_row['trial_type'] = f"{row['trial_type']}-stop"
            expanded_rows.append(new_row)
            trial_start_sec = signal_start_time + 10

    return pd.DataFrame(expanded_rows)

# Generate epochs based on trial information
def generate_epochs(raw, df):
    previous_values = [0] * len(df)
    instruction_dict = {
        'rcmd-keep': 1,
        'rcmd-stop': 2,
        'lcmd-keep': 1,
        'lcmd-stop': 2
    }    

    # Map trial types to event IDs
    df['event_id'] = df['trial_type'].map(instruction_dict)
    event_ids = df['event_id'].tolist()  # Removed addition of a final event

    # Combine results into the instruction array
    instructions = np.column_stack([df['start_sample'], previous_values, event_ids])
    
    # Each 10s following a given instruction is split into 5 epochs (each of 2s)
    n_epo_segments = 5
    n_samples = raw.info['sfreq'] * 10 / n_epo_segments

    events = list()
    events_info = list()

    # For each instruction
    for instr_id, (onset, _, code) in enumerate(instructions):
        # Generate 5 epochs
        for repeat in range(n_epo_segments):
            event = [onset + repeat * n_samples + 2.3 * config['sfreq'], 0, code]

            # Store into new event array
            events.append(event)
            events_info.append(instr_id)

    events = np.array(events, int)

    metadata = pd.DataFrame(dict(
        time_sample=events[:, 0], # time sample in the EEG file
        id=events[:, 2], # the unique code of the epoch
        move=(events[:, 2] % 2) == 1, # whether the code corresponds to a 'move' trial
        instr=events_info, # the instruction from which the epoch comes
        trial=np.array(events_info)//2 # trial number: there are two instructions per trial
    ))

    metadata['block'] = metadata['trial'] // 8
    return events, metadata

# Plot trial/instruction/epoch structure
def plot_instructions_and_epochs(instructions, events):
    """Plot trial/instruction/epoch structure and save the plot."""
    plt.figure(figsize=(10, 5))
    plt.scatter(instructions[:, 0], instructions[:, 2], color=['g', 'r'] * (len(instructions) // 2), marker='s', label='Instruction offset')
    plt.scatter(events[:, 0], events[:, 2] + 0.1, color=np.ravel([['g'] * 5 + ['r'] * 5] * (len(instructions) // 2)), label='Epoch onset')
    
    plt.ylabel('Instructions')
    plt.yticks([1, 2])
    plt.gca().set_yticklabels(['Keep moving...', 'Stop moving...'])
    plt.legend()
    plt.grid(True)
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    return plt

# Preprocess epochs
def preprocess_epochs(raw, events, metadata, subject_id):
    """Preprocess the epochs and compute the Current Source Density (CSD)."""
    use_ch = config.get(f"{subject_id}_channels", config.get("channels"))
    if "DC7" in use_ch:
        use_ch.remove("DC7")
    
    picks = mne.pick_types(raw.info, include=use_ch)

    if raw.info['dig'] is None:
        print("No digitization data found. Assigning standard montage.")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"))

    epochs = mne.Epochs(
        raw,
        tmin=0, tmax=2,
        picks=picks,
        events=events,
        metadata=metadata,
        preload=True,
        proj=False,
        baseline=None
    )

    epochs.info['description'] = 'standard/1020'
    csd_epochs = mne.preprocessing.compute_current_source_density(epochs)
    return csd_epochs

# Perform Power Spectral Density (PSD) analysis
def compute_psd(epochs, sfreq=512, fmin=1, fmax=30):
    """Compute Power Spectral Density (PSD) for the epochs."""
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    psds_all_epochs = []

    for epoch_data in data:  # Shape: (n_channels, n_times)
        psds, freqs = psd_array_multitaper(epoch_data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
        psds_all_epochs.append(psds)

    return np.array(psds_all_epochs), freqs

# Extract band-specific PSD data
def extract_band_psd(psds_all_epochs, freqs, bands):
    """Extract band-specific PSD data from the full PSD."""
    n_epochs, n_chans, n_freqs = psds_all_epochs.shape
    psd_data = np.zeros((n_epochs, n_chans, len(bands)))

    for ii, (fmin, fmax) in enumerate(bands):
        freq_index = np.where(np.logical_and(freqs >= fmin, freqs <= fmax))[0]
        psd_data[:, :, ii] = psds_all_epochs[:, :, freq_index].mean(2)

    return psd_data.reshape(n_epochs, n_chans * len(bands))

# Define classifier
def define_classifier():
    """Define and return the classifier pipeline."""
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True)
    )
    return clf

# Perform cross-validation
def plot_cross_validation(cv, epochs, psd_data):
    """Plot the cross-validation results."""
    fig, axes = plt.subplots(4, 1, figsize=[14, 9])
    axes = iter(axes)

    for split, (train, test) in enumerate(cv.split(X=psd_data, y=epochs.metadata['move'], groups=epochs.metadata['trial'])):
        if split >= 3 and split != 8:
            continue
        ax = next(axes)

        ax.scatter(epochs.metadata['time_sample'].values[train], 1 - epochs.metadata['move'].values[train], color='k', label='train')
        ax.scatter(epochs.metadata['time_sample'].values[test], 1 - epochs.metadata['move'].values[test], edgecolor='b', color='w', label='test')
        ax.set_title(f'CV Split #{split + 1}')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['"Keep moving..."', '"Stop moving..."'])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    return plt

def decode_performance(clf, psd_data, epochs, cv):
    y_pred = cross_val_predict(
        clf,
        X=psd_data,
        y=epochs.metadata['move'],
        method='predict_proba',
        cv=cv,
        groups=epochs.metadata['trial']
    )

    epochs.metadata['y_pred'] = y_pred[:, 1]
    proba = np.mean([block['y_pred'] for _, block in epochs.metadata.groupby('block')], axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylabel('P ("Keep moving ...")')
    ax.set_xlabel('Epoch number')
    ax.plot(proba, marker='o', linestyle='-', linewidth=3, markersize=6)
    ax.set_ylim([0, 1])

    plt.axhline(0.5, linestyle=':', color='k', label='Chance')

    for x in np.arange(-0.5, 79, 10):
        plt.axvline(x, color='g', label='Keep moving ...' if x < 4 else None)
        plt.axvline(x + 5, color='r', label='Stop moving ...' if x < 4 else None)

    plt.legend(loc='lower right', framealpha=1.)
    plt.title("Average predicted probability of 'keep moving ...' across the three blocks")
    
    return plt

def plot_topo_map(clf, psd_data, epochs, bands):
    clf = make_pipeline(
        StandardScaler(), # z-score to center data
        LinearModel(LinearSVC())) # Linear SVM augmented with an automatic storing of spatial patterns

    clf.fit(X=psd_data, y=epochs.metadata['move'])

    patterns = get_coef(clf, 'patterns_', inverse_transform=True)
    n_elect = int(patterns.shape[0] / len(bands))
    spatial_pattern = patterns.reshape(n_elect, len(bands))

    montage = mne.channels.make_standard_montage('standard_1020')
    pos = np.array([epochs.info['chs'][i]['loc'][:2] for i in range(len(epochs.info['chs'])) if epochs.info['chs'][i]['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH])

    fig, axes = plt.subplots(1, len(bands), figsize=(12, 3))
    fig.suptitle("EEG Spatial Patterns Across Frequency Bands", fontsize=14)
    fig.tight_layout()

    for idx, (band, sp, ax) in enumerate(zip(bands, spatial_pattern.T, axes)):
        scale = np.percentile(np.abs(sp), 99)
        im, _ = mne.viz.plot_topomap(sp, pos, vlim=(-scale, +scale), cmap='RdBu_r', axes=ax, show=False)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Weight (Arbitrary Units)", fontsize=8)
        ax.set_title(f'{band[0]} - {band[1]} Hz')

    return plt

def compute_auc(clf, psd_data, epochs, cv, subject_id, base_dir):
    scores = cross_val_score(
        estimator=clf,
        X=psd_data,
        y=epochs.metadata['move'],
        scoring='roc_auc',
        cv=cv,
        groups=epochs.metadata['trial']
    )

    mean_score = scores.mean(0)
    save_log(subject_id, base_dir, f'Mean scores across split: AUC={mean_score:.3f}')
    print(f'Mean scores across split: AUC={mean_score:.3f}')
    return mean_score, scores

def permutation_test(clf, psd_data, epochs, cv, subject_id, base_dir, n_permutations=500):
    permutation_scores = []
    order = np.arange(len(epochs))

    for _ in tqdm(range(n_permutations)):  
        np.random.shuffle(order)

        permutation_score = cross_val_score(
            estimator=clf,
            X=psd_data,
            y=epochs.metadata['move'].values[order],
            scoring='roc_auc',
            cv=cv,
            groups=epochs.metadata['trial'].values[order],
            n_jobs=-1
        )

        permutation_scores.append(permutation_score.mean(0))

    observed_score = cross_val_score(
        estimator=clf,
        X=psd_data,
        y=epochs.metadata['move'].values,
        scoring='roc_auc',
        cv=cv,
        groups=epochs.metadata['trial'].values,
        n_jobs=-1
    ).mean(0)

    p_value = np.mean(np.array(permutation_scores) >= observed_score)
    save_log(subject_id, base_dir, f"Permutation p-value = {p_value:.4f}")
    print(f"Permutation p-value = {p_value:.4f}")

    plt.hist(permutation_scores, bins=30, alpha=0.7, label='Permutation scores')
    plt.axvline(observed_score, color='red', linestyle='--', label='Observed score')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Permutation Test Null Distribution')

    return p_value, permutation_scores, observed_score, plt


def plot_permutation_test(permutation_scores, scores, observed_score, subject_id, base_dir, n_permutations):
    """
    Plots the permutation test results and computes the p-value.

    Parameters:
    - permutation_scores: array-like, scores obtained from permutations
    - scores: array-like, scores from original data
    - observed_score: float, score from the original model
    - n_permutations: int, number of permutations performed
    """
    # Plot histogram of permutation scores
    plt.figure(figsize=(10, 5))
    plt.hist(permutation_scores, bins=30, alpha=0.7, label='Permutation scores')
    plt.axvline(observed_score, color='red', linestyle='--', label='Observed score')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Permutation Test Null Distribution')
    plt.show()
    
    # Compute p-value
    n_higher = sum(s >= scores.mean(0) for s in permutation_scores)
    pvalue = (n_higher + 1.) / (n_permutations + 1.)

    save_log(subject_id, base_dir, f"Empirical AUC = {scores.mean(0):.2f} +/- {scores.std(0):.2f}")
    save_log(subject_id, base_dir, f"Shuffle AUC = {np.mean(permutation_scores, 0):.2f}")
    save_log(subject_id, base_dir, f"p-value = {pvalue:.4f}")    
    print("Empirical AUC = %.2f +/- %.2f" % (scores.mean(0), scores.std(0)))
    print("Shuffle AUC = %.2f" % np.mean(permutation_scores, 0))
    print("p-value = %.4f" % pvalue)
    
    # Plot permutation and empirical distributions
    plt.figure(figsize=(10, 5))
    sns.kdeplot(permutation_scores, label='Permutation scores', fill=True)
    sns.kdeplot(scores, label='Empirical scores', fill=True)
    plt.axvline(0.5, linestyle='--', label='Theoretical chance')
    plt.axvline(scores.mean(), color='orange', label='Mean score')
    plt.scatter(scores, 6. + np.random.randn(len(scores)) / 10., color='orange', s=5, label='Split score')
    plt.xlim(-0.1, 1.1)
    plt.legend()
    plt.xlabel('AUC Score')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.title('Permutation vs Empirical Distribution')

    return plt

def run_analysis(subject_id, dir, date_str):
    subject_id = subject_id # "CON002" #CON001a, CON001b, CON002, CON003, CON004, CON005
    base_dir = os.path.join(os.getcwd(), dir)
    bands = ((1,3), (4,7), (8,13), (14,30))
    
    # Create a folder for the patient and date (if not exists)
    patient_folder = os.path.join(base_dir, f"{subject_id}_{date_str}")
    
    # Create the directory if it doesn't exist
    os.makedirs(patient_folder, exist_ok=True)

    try:
        raw, dc_channel, subject_id = load_eeg_data(subject_id)
        save_plot(subject_id, patient_folder, raw.plot(), 'raw_eeg_plot')

        df = process_trials(raw, subject_id, dc_channel)
        events, metadata = generate_epochs(raw, df)

        np.set_printoptions(threshold=np.inf)
        previous_values = [0] * len(df)
        event_ids = df['event_id'].tolist()  # Removed addition of a final event
        instructions = np.column_stack([df['start_sample'], previous_values, event_ids])
        epochs_plt = plot_instructions_and_epochs(instructions, events)
        save_plot(subject_id, patient_folder, epochs_plt, 'instructions_epochs')
        
        epochs = preprocess_epochs(raw, events, metadata, subject_id)
        save_plot(subject_id, patient_folder, epochs.plot(scalings='auto', n_epochs=3), 'prerpocess_epochs_plot')
        
        psds_all_epochs, freqs = compute_psd(epochs)
        psd_data = extract_band_psd(psds_all_epochs, freqs, bands)

        # Define cross validation
        cv = LeaveOneGroupOut()
        cv_plt = plot_cross_validation(cv, epochs, psd_data)
        save_plot(subject_id, patient_folder, cv_plt, 'cross_validation')

        clf = define_classifier()
        # Decoding performance over time
        prob_plt = decode_performance(clf, psd_data, epochs, cv)
        save_plot(subject_id, patient_folder, prob_plt, 'average_predicted_probability')

        # Topo Map
        topo_plt = plot_topo_map(clf, psd_data, epochs, bands)
        save_plot(subject_id, patient_folder, topo_plt, 'topo_map')

        # Computing cross-validated AUC scores
        mean_score, scores = compute_auc(clf, psd_data, epochs, cv, subject_id, patient_folder)

        # Performing permutation test
        p_value, permutation_scores, observed_score, dis_plt = permutation_test(clf, psd_data, epochs, cv, subject_id, patient_folder, n_permutations=500)
        save_plot(subject_id, patient_folder, dis_plt, 'permutation_distribution')
        
        perm_plt = plot_permutation_test(permutation_scores, scores, observed_score, subject_id, patient_folder, n_permutations=500)
        save_plot(subject_id, patient_folder, perm_plt, 'permutation_plt')

    except Exception as e:
        save_log(subject_id, patient_folder, f"Error: {str(e)}")


# Main processing flow
def main():
    subject_id = "CON002" #CON001a, CON001b, CON002, CON003, CON004, CON005
    base_dir = os.path.join(os.getcwd(), 'data')
    bands = ((1,3), (4,7), (8,13), (14,30))

    try:
        raw, dc_channel, subject_id = load_eeg_data(subject_id)
        save_plot(subject_id, base_dir, raw.plot(), 'raw_eeg_plot')

        df = process_trials(raw, subject_id, dc_channel)
        events, metadata = generate_epochs(raw, df)

        np.set_printoptions(threshold=np.inf)
        previous_values = [0] * len(df)
        event_ids = df['event_id'].tolist()  # Removed addition of a final event
        instructions = np.column_stack([df['start_sample'], previous_values, event_ids])
        epochs_plt = plot_instructions_and_epochs(instructions, events)
        save_plot(subject_id, base_dir, epochs_plt, 'instructions_epochs')
        
        epochs = preprocess_epochs(raw, events, metadata, subject_id)
        save_plot(subject_id, base_dir, epochs.plot(scalings='auto', n_epochs=3), 'prerpocess_epochs_plot')
        
        psds_all_epochs, freqs = compute_psd(epochs)
        psd_data = extract_band_psd(psds_all_epochs, freqs, bands)

        # Define cross validation
        cv = LeaveOneGroupOut()
        cv_plt = plot_cross_validation(cv, epochs, psd_data)
        save_plot(subject_id, base_dir, cv_plt, 'cross_validation')

        clf = define_classifier()
        # Decoding performance over time
        prob_plt = decode_performance(clf, psd_data, epochs, cv)
        save_plot(subject_id, base_dir, prob_plt, 'average_predicted_probability')

        # Topo Map
        topo_plt = plot_topo_map(clf, psd_data, epochs, bands)
        save_plot(subject_id, base_dir, topo_plt, 'topo_map')

        # Computing cross-validated AUC scores
        mean_score, scores = compute_auc(clf, psd_data, epochs, cv, subject_id, base_dir)

        # Performing permutation test
        p_value, permutation_scores, observed_score, dis_plt = permutation_test(clf, psd_data, epochs, cv, subject_id, base_dir, n_permutations=500)
        save_plot(subject_id, base_dir, dis_plt, 'permutation_distribution')
        
        perm_plt = plot_permutation_test(permutation_scores, scores, observed_score, subject_id, base_dir, n_permutations=500)
        save_plot(subject_id, base_dir, perm_plt, 'permutation_plt')

    except Exception as e:
        save_log(subject_id, base_dir, f"Error: {str(e)}")

if __name__ == "__main__":
    main()
