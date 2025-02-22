# Cognitive-Motor Command Response Analysis (Adapted from Claassen 2019)

# %pip install matplotlib
# %pip install PyQt5
# %pip install pyyaml
# %pip install mne
# %pip install numpy
# %pip install pandas
# %pip install scikit-learn
# %pip install seaborn
# %pip install -U ipywidgets
# %pip install jupyter
# %pip install --upgrade ipywidgets
# %pip install git+https://github.com/nice-tools/pycsd.git

import matplotlib.pyplot as plt
import matplotlib
import yaml
import sys
import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import pycsd

from tqdm import tqdm_notebook
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneGroupOut
from mne.time_frequency import psd_array_multitaper
from mne.decoding import LinearModel, get_coef
from data.eeg_loader import load_eeg, get_eeg_timestamps, load_stimulus, detect_signal_start

# Use matplotlib Qt5Agg backend - best choice for MNE-Python interactive plotting functions
matplotlib.use('Qt5Agg')

# Add the parent directory of 'data' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

config_file_path = '../configs/claassen_cfg.yml'  # Replace with the actual path to your config file
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# ### 1. Preprocessing Raw Data
# %%
# eeg_path = r"/Users/joobeejung/EEG_DATA/X~ X_ef7f7805-3781-4f1e-8134-8d5b57750330.EDF"
# eeg_path = r"/Users/joobeejung/EEG_DATA/X~ X_86b44d5d-5036-4748-b666-ceff710a1e8d.EDF" #CON001
# eeg_path = r"/Users/joobeejung/EEG_DATA/CON002_20240924.EDF"
# eeg_path = r"/Users/joobeejung/EEG_DATA/CON003_clipped.EDF"

subject = "CON003" #CON001a, CON001b, CON002, CON003, CON004
eeg_path = config.get(f"{subject}_path", "")  # Remove extra quotes

if eeg_path is None:
    print(f"Error: No path found for subject {subject}")
else:
    print(f"EEG Path for {subject}: {eeg_path}")

# Check if file exists
if not os.path.exists(eeg_path):
    raise FileNotFoundError(f"File not found: {eeg_path}")

#---Read raw without removing channels---
# # Read raw data
raw = mne.io.read_raw_edf(eeg_path, preload=True)

# # Band-pass filter between 1-30 Hz over the readings
fmin, fmax = 1, 30 # Hz
raw.filter(l_freq=fmin, h_freq=fmax)

#---Read raw & removing channels---
fname, raw, dc_channel = load_eeg(eeg_path, config, subject)
# raw.set_eeg_reference('average')
raw.plot()

# ### 3. Reading events and segmenting trials into epochs

# %%
start_time, end_time = get_eeg_timestamps(raw, subject)
df, patient_id = load_stimulus(config["event_full_path"], start_time, end_time)
patient_trial = df.loc[(df['patient_id'] == patient_id) & (df['trial_type'].isin(config['trial_type']))]

def trial_start_sec(row):
    return (row['start_time']-start_time).total_seconds()
def trial_end_sec(row):
    return (row['end_time']-start_time).total_seconds()

patient_trial['start_sec'] = patient_trial.apply(trial_start_sec, axis=1)
patient_trial['end_sec'] = patient_trial.apply(trial_end_sec, axis=1)
df = patient_trial
expanded_rows = []

# %%
for _, row in df.iterrows():
    trial_start_sec = row['start_sec']
    signal_start_time = trial_start_sec

    for i in range(8):
        signal_start_sample, signal_start_time = detect_signal_start(raw, dc_channel, trial_start_sec)

        new_row = row.copy()
        new_row['start_sample'] = signal_start_sample
        new_row['start_sec'] = signal_start_time
        if i % 2 == 0:
            new_row['trial_type'] = f"{row['trial_type']}-keep"
        else:
            new_row['trial_type'] = f"{row['trial_type']}-stop"
        expanded_rows.append(new_row)
        trial_start_sec = signal_start_time + 10

new_df = pd.DataFrame(expanded_rows)
previous_values = [0] * len(new_df)

instruction_dict = {
    'rcmd-keep': 1,
    'rcmd-stop': 2,
    'lcmd-keep': 1,
    'lcmd-stop': 2
}

# Map trial types to event IDs
new_df['event_id'] = new_df['trial_type'].map(instruction_dict)
event_ids = new_df['event_id'].tolist()  # Removed addition of a final event

# # Combine results into the instruction array
instructions = np.column_stack([new_df['start_sample'], previous_values, event_ids])

# Each 10s following a given instruction is split into 5 epochs (each of 2s)
n_epo_segments = 5
n_samples = raw.info['sfreq']*10/n_epo_segments

events = list()
events_info = list()

# For each instruction
for instr_id, (onset, _, code) in enumerate(instructions):
    # Generate 5 epochs
    for repeat in range(n_epo_segments):
        event = [onset + repeat * n_samples + 2.3*config['sfreq'], 0, code]

        # Store into new event array
        events.append(event)
        events_info.append(instr_id)
events = np.array(events, int)

# Add information
metadata = pd.DataFrame(dict(
    time_sample=events[:, 0], # time sample in the EEG file
    id=events[:, 2], # the unique code of the epoch
    move=(events[:, 2] % 2) == 1, # whether the code corresponds to a 'move' trial
    instr=events_info, # the instruction from which the epoch comes
    trial=np.array(events_info)//2, # trial number: there are two instructions per trial
))

# There are 8 trials per block
metadata['block'] = metadata['trial'] // 8
np.set_printoptions(threshold=np.inf)
# %%
# Plot trial/instruction/epoch structure for clarity
plt.scatter(instructions[:, 0], instructions[:, 2],
            color=['g', 'r'] * (len(instructions) // 2),  # Ensure the size matches
            marker='s',
            label='Instruction offset')
plt.scatter(events[:,0], events[:,2]+.1,
            color=np.ravel([['g'] * 5 + ['r'] * 5] * (len(instructions)//2)),
            label='Epoch onset')

plt.ylabel('Instructions')
plt.yticks([1,2])
plt.gca().set_yticklabels(['Keep moving...', 'Stop moving...'])
plt.legend()
plt.grid(True)
plt.show()

# %%
# use_ch = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'Pz', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
# use_ch = ['Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'Pz', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2'] # patient 2 after filtering
use_ch = config.get(f"{subject}_channels")  # Append "_channels" to match config keys
use_ch = [ch for ch in use_ch if ch != "DC7"]  # Exclude "DC7"

if use_ch is None:
    use_ch = config.get("channels")
    print(f"Error: No channels found for subject {subject}")
else:
    print(f"EEG channels for {subject}: {use_ch}")
    
picks = mne.pick_types(raw.info, include=use_ch)
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

# %%
epochs.info['description'] = 'standard/1020'
epochs = pycsd.epochs_compute_csd(epochs)

epochs.plot(scalings='auto', n_epochs=3)

# ## 3. Performing Power Spectral Density (PSD) Analysis
data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
n_epochs, n_channels, n_times = data.shape

# Initialize a container for storing PSDs
psds_all_epochs = []

# Loop through each epoch
for epoch_data in data:  # Shape: (n_channels, n_times)
    psds, freqs = psd_array_multitaper(
        epoch_data, sfreq=512, fmin=1, fmax=30, verbose=False
    )
    psds_all_epochs.append(psds)  # Append PSD for this epoch

# Convert to NumPy array for easier manipulation
psds_all_epochs = np.array(psds_all_epochs)  # Shape: (n_epochs, n_channels, n_freqs)

# %%
n_epochs, n_chans, n_freqs = psds_all_epochs.shape

#Frequency bands of interest in Hz
bands = ((1,3), (4,7), (8,13), (14,30))

# Setup X array average PSD within a given frequency band
psd_data = np.zeros((n_epochs, n_chans, len(bands)))
for ii, (fmin, fmax) in enumerate(bands):
    #Find frequencies
    freq_index = np.where(np.logical_and(freqs >= fmin, freqs <= fmax))[0]

    psd_data[:, :, ii] = psds_all_epochs[:, :, freq_index].mean(2)

# Vectorize PSD
psd_data = psd_data.reshape(n_epochs, n_chans * len(bands))

# ## 4. Defining Cross Validation 

# Define cross validation
cv = LeaveOneGroupOut()

fig, axes = plt.subplots(4, 1, figsize=[14,9])
axes = iter(axes)

for split, (train, test) in enumerate(cv.split(
    X = psd_data,
    y = epochs.metadata['move'],
    groups = epochs.metadata['trial'],
)):
    if split >= 3 and split != 8:
        continue
    ax = next(axes)

    ax.scatter(epochs.metadata['time_sample'].values[train], 1 - epochs.metadata['move'].values[train], color='k', label='train')
    ax.scatter(epochs.metadata['time_sample'].values[test], 1 - epochs.metadata['move'].values[test], edgecolor='b', color='w', label='test')
    ax.set_title('CV Split #%i' % (split+1))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['"Keep moving..."', '"Stop moving..."'])
    ax.set_ylabel('Instruction')
    ax.set_xlabel('Time')
    ax.set_xticks([])
    # ax.set_xlim(0, 100000) # zoom for clarity
    ax.legend()

fig.tight_layout()
fig.show()


# ## 5. Defining classifier (SVM)
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', probability=True)
)

# ## 6. Decoding performance over time

y_pred = cross_val_predict(
    clf,
    X = psd_data,
    y = epochs.metadata['move'],
    method = 'predict_proba',
    cv = cv,
    groups = epochs.metadata['trial']
)

epochs.metadata['y_pred'] = y_pred[:, 1]

# Average the proba over the 3 blocks to obtain temporal pattern
proba = np.mean([block['y_pred'] for _, block in epochs.metadata.groupby('block')], axis=0)

# Plot the proba
fig, ax = plt.subplots(figsize=(12,4))
ax.set_ylabel('P ("Keep moving ...")')
ax.set_xlabel('Epoch number')
ax.plot(proba, marker='o', linestyle='-', linewidth=3, markersize=6)
ax.set_ylim([0,1])
plt.axhline(0.5, linestyle=':', color='k', label='Chance')

for x in np.arange(-0.5, 79, 10):
    plt.axvline(x, color='g', label='Keep moving ...' if x < 4 else None)
    plt.axvline(x + 5, color='r', label='Stop moving ...' if x < 4 else None)

plt.legend(loc='lower right', framealpha=1.)
plt.title("Average predicted probability of 'keep moving ...' across the three blocks") 

plt.show()


# %%
# use_ch = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'Pz', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
# use_ch = ['Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'Pz', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
# use_ch = ["Fz", "F3", "F4", "F7", "F8", "Cz", "C3", "C4", "T3", "T4", "Pz", "P3", "P4", "T5", "T6", "O1", "O2"]
# use_ch = config.get(f"{subject}_channels")  # Append "_channels" to match config keys
# use_ch = config.get("channels")  # Append "_channels" to match config keys
use_ch = config.get(f"{subject}_channels")  # Append "_channels" to match config keys
# use_ch = ["Fz", "F3", "F4", "Cz", "C3", "C4", "T3", "T4", "Pz", "P3", "P4", "T5", "T6", "O1", "O2"]
use_ch = [ch for ch in use_ch if ch != "DC7"]  # Exclude "DC7"

if use_ch is None:
    print(f"Error: No channels found for subject {subject}")
else:
    print(f"EEG channels for {subject}: {use_ch}")
raw.pick(use_ch)


# %% [markdown]
# ## 5. Topo Map

# %%
# Define the classifier and stode spatial patterns
# To plot the SVM patterns, it is necessary to compute the data
# covariance (Haufe et al Neuroimage 2014).
# Spatial patterns are automatically stored by MNE LinearModel.
clf = make_pipeline(
StandardScaler(), # z-score to center data
LinearModel(LinearSVC())) # Linear SVM augmented with an automatic storing of spatial patterns
# fit classifier
clf.fit(X=psd_data,
y=epochs.metadata['move'])
# Unscale the spatial patterns before plotting
patterns = get_coef(clf, 'patterns_', inverse_transform=True)
# In our study, the SVM is trained on all frequencies simultanouesly
# we thus pull the corresponding spatial topographies apart
n_elect = int(patterns.shape[0] / len(bands))
spatial_pattern = patterns.reshape(n_elect, len(bands))
# Plot
# montage = mne.channels.read_montage('standard/1020')
montage = mne.channels.make_standard_montage('standard_1020')

# The paper used this code but this may not match actual EEG data
# raw.set_montage(montage)
# pos = mne.channels.layout.find_layout(raw.info['chs']).pos

# Get electrode positions directly from montage - This way is more reliable for real data, ensures correct channel order and locations
pos = np.array([raw.info['chs'][i]['loc'][:2] for i in range(len(raw.info['chs'])) if raw.info['chs'][i]['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH])

fig, axes = plt.subplots(1, len(bands),figsize=(12,3))
fig.suptitle("EEG Spatial Patterns Across Frequency Bands", fontsize=14)
fig.tight_layout()

for idx, (band, sp, ax) in enumerate(zip(bands, spatial_pattern.T, axes)):
    scale = np.percentile(np.abs(sp), 99)
    im, _ = mne.viz.plot_topomap(sp, pos, vlim=(-scale,+scale), cmap='RdBu_r', axes=ax, show=False)
    # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight (Arbitrary Units)", fontsize=8)  # Update with correct units if known

    ax.set_title('%i - %i Hz' % band)

plt.show()

# %%
raw.info['chs']

# %% [markdown]
# ## 6. Computing cross-validated AUC scores

# %%
clf = make_pipeline(
    StandardScaler(),
    LinearSVC()
)

scores = cross_val_score(
    estimator=clf,
    X=psd_data,
    y=epochs.metadata['move'],
    scoring='roc_auc',
    cv=cv,
    groups=epochs.metadata['trial']
)

mean_score = scores.mean(0)
print('Mean scores across split: AUC=%.3f' % mean_score)

# %% [markdown]
# ## 7. Diagnosis of cognitive motor dissociation (CMD)

# %% [markdown]
# ### 7.1 Performing permutation test

# %%
permutation_scores = []
n_permutations = 500
order = np.arange(len(epochs))

for _ in tqdm_notebook(range(n_permutations)):  
    # Shuffle order
    np.random.shuffle(order)

    # Compute score with similar parameters
    permutation_score = cross_val_score(
        estimator=clf,
        X=psd_data,
        y=epochs.metadata['move'].values[order],
        scoring='roc_auc',
        cv=cv,
        groups=epochs.metadata['trial'].values[order],
        n_jobs=-1,
    )

    # Store results
    permutation_scores.append(permutation_score.mean(0))

# %%
observed_score = cross_val_score(
    estimator=clf,
    X=psd_data,
    y=epochs.metadata['move'].values,
    scoring='roc_auc',
    cv=cv,
    groups=epochs.metadata['trial'].values,
    n_jobs=-1,
).mean(0)

p_value = np.mean(np.array(permutation_scores) >= observed_score)

plt.hist(permutation_scores, bins=30, alpha=0.7, label='Permutation scores')
plt.axvline(observed_score, color='red', linestyle='--', label='Observed score')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.legend()
plt.title('Permutation Test Null Distribution')
plt.show()

# The p-value is computed from the number of permutations which
# leads to a higher score than the one obtained without permutation
# p = n_higher + 1 / (n_permutation + 1)
#
# (Ojala M GG. Journal of Machine Learning Research. 2010).
n_higher = sum([s >= scores.mean(0) for s in permutation_scores])
pvalue = (n_higher + 1.) / (n_permutations + 1.)
print("Empirical AUC = %.2f +/-%.2f" % (scores.mean(0), scores.std(0)))
print("Shuffle AUC = %.2f" % np.mean(permutation_scores, 0))
print("p-value = %.4f" % pvalue)
# plot permutation and empirical distributions
sns.kdeplot(permutation_scores, label='permutation scores')
sns.kdeplot(scores)
plt.axvline(.5, linestyle='--', label='theoretical chance')
plt.axvline(scores.mean(), color='orange', label='mean score')
plt.scatter(scores, 6. + np.random.randn(len(scores))/10., color='orange', s=5, label='split score')
plt.xlim(-.1, 1.1)
plt.legend()
plt.xlabel('AUC Score')
plt.ylabel('Probability')
plt.yticks([])
plt.show()