#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import FastICA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

# Import Classifiers (5 matching TABLE II: SVM, RF, DT, KNN, NB)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone

# =====================================================================
# CONFIGURATION -- EDIT THIS SECTION
# =====================================================================

# Base directory where CSV files are located
BASE_DIR = r'C:\Users\Admin\Downloads\MI BCI'

# List of EEG CSV files to process.
# Add your 15 files here. Each entry is (Label, Filename).
# Label = short name for the subject (e.g. "S1", "S2", etc.)
# Filename = CSV filename (will be looked up in BASE_DIR)
EEG_FILES = [
    ("S1",  "S16"),
    ("S2",  "S23"),
    ("S3",  "S41"),
    ("S4",  "S78"),
    ("S5",  "S62"),
    ("S6",  "S65"),
    ("S7",  "S48"),
    ("S8",  "S06"),
    ("S9",  "S46"),
    ("S10", "S15"),
    ("S11", "S52"),
    ("S12", "S15_1"),
    ("S13", "S03"),
    ("S14", "P1943"),
    ("S15", "S40"),
]

# Channel configurations (matching TABLE II columns)
all_channels   = ['Fp1','Fp2','F3','Fz','F4','T7','C3','Cz','C4','T8',
                  'P3','Pz','P4','PO7','Oz','PO8']
strict_motor   = ['C3','Cz','C4']
extended_motor = ['F3','Fz','F4','C3','Cz','C4','P3','Pz','P4']

CHANNEL_CONFIGS = {
    "All Channels (Fixation Off)":              (all_channels, 6),
    "Strict Motor Cortex (Fixation Off)":       (strict_motor, 2),
    "Extended Motor Cortex (Fixation Off)":      (extended_motor, 6),
}

# Classifiers (matching TABLE II: SVM, RF, DT, KNN, NB)
CLASSIFIERS = {
    "SVM": SVC(kernel="rbf", random_state=42),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
    "DT":  DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NB":  GaussianNB(),
}

FS             = 500        # Sampling frequency (Hz)
DECISION_START = int(0.5 * FS)   # start after cue
DECISION_END   = int(2.5 * FS)   # MI window      # 3-second epoch @ 500 Hz
DECISION_SAMPLES = DECISION_END - DECISION_START
N_FOLDS        = 5          # Stratified K-Fold

CLF_NAMES = list(CLASSIFIERS.keys())  # ['SVM', 'RF', 'DT', 'KNN', 'NB']


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def apply_bandpass(data, lowcut, highcut, fs, order=4):
    """Bandpass filter (8-30 Hz -- Mu + Beta rhythms)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)
# ================= NEW: ERD + LATERALIZATION =================

def compute_erd(epoch):
    baseline = epoch[:, :int(0.5 * FS)]
    active   = epoch[:, int(0.5 * FS):]

    bp = np.mean(baseline**2, axis=1)
    ap = np.mean(active**2, axis=1)

    bp = np.where(bp < 1e-10, 1e-10, bp)
    return (ap - bp) / bp * 100


def check_lateralization(epoch, label):
    erd = compute_erd(epoch)

    c3 = all_channels.index('C3')
    c4 = all_channels.index('C4')

    erd_c3 = erd[c3]
    erd_c4 = erd[c4]

    # LEFT MI → C4 ERD
    if label == 0:
        return (erd_c4 < -5) and (abs(erd_c4) > abs(erd_c3))

    # RIGHT MI → C3 ERD
    else:
        return (erd_c3 < -5) and (abs(erd_c3) > abs(erd_c4))

class LengthAgnosticCSP:
    """Common Spatial Patterns for variable-length trials."""
    def __init__(self, n_components=6):
        self.n_components = n_components
        self.W = None

    def _avg_cov(self, trials):
        n_ch = trials[0].shape[0]
        if not trials:
            return np.eye(n_ch)
        covs = []
        for trial in trials:
            trial_c = trial - trial.mean(axis=1, keepdims=True)
            cov     = trial_c @ trial_c.T
            cov    += np.eye(n_ch) * 1e-6
            covs.append(cov / np.trace(cov))
        return np.mean(covs, axis=0)

    def fit(self, X_train, y_train):
        X0 = [X_train[i] for i in range(len(y_train)) if y_train[i] == 0]
        X1 = [X_train[i] for i in range(len(y_train)) if y_train[i] == 1]
        C0 = self._avg_cov(X0)
        C1 = self._avg_cov(X1)
        eigenvalues, eigenvectors = eigh(C1, C0 + C1)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        half = self.n_components // 2
        self.W = np.vstack([eigenvectors[:, :half].T,
                            eigenvectors[:, -half:].T])
        return self

    def transform(self, X_list):
        features = []
        for trial in X_list:
            proj    = self.W @ trial
            var     = np.var(proj, axis=1)
            log_var = np.log(var / var.sum() + 1e-10)
            features.append(log_var)
        return np.array(features)


def classify_with_kfold(X_all_data, y, channel_subset, n_comp, n_splits=N_FOLDS):
    """
    Run Stratified K-Fold CV for all 5 classifiers on a given channel config.

    Returns
    -------
    mean_results : dict  {clf_name: mean_accuracy}        -- TABLE II output (unchanged)
    cv_info      : dict  -- CV test-fold data for visualisation ONLY:
        'y_true'    : np.ndarray  true labels of every test-fold sample
        'y_pred'    : dict {clf_name -> np.ndarray}  predicted labels per classifier
        'epoch_idx' : np.ndarray  original indices into X_all_data that were test epochs
        'X_csp'     : np.ndarray  (n_epochs, n_comp) CSP features from test folds
        'csp_last'  : LengthAgnosticCSP fitted on last fold's training data
        'ch_subset' : list of channel names used
    All arrays are sorted back into original epoch order.
    """
    indices  = [all_channels.index(ch) for ch in channel_subset]
    X_subset = [trial[indices, :] for trial in X_all_data]
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {name: [] for name in CLASSIFIERS}

    # Accumulators for CV test predictions (visualisation only -- TABLE II unaffected)
    _cv_epoch_idx = []
    _cv_y_true    = []
    _cv_X_csp     = []
    _cv_y_pred    = {name: [] for name in CLASSIFIERS}
    _csp_last     = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subset, y)):
        X_train = [X_subset[i] for i in train_idx]
        X_test  = [X_subset[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # Fit CSP on training fold ONLY
        csp = LengthAgnosticCSP(n_components=n_comp)
        csp.fit(X_train, y_train)
        X_train_csp = csp.transform(X_train)
        X_test_csp  = csp.transform(X_test)

        # Accumulate test-fold data for later visualisation
        _cv_epoch_idx.extend(test_idx.tolist())
        _cv_y_true.extend(y_test.tolist())
        _cv_X_csp.append(X_test_csp)
        _csp_last = csp                        # last fold's CSP for pattern plots

        for name, clf_template in CLASSIFIERS.items():
            clf = clone(clf_template)
            clf.fit(X_train_csp, y_train)
            y_pred = clf.predict(X_test_csp)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            _cv_y_pred[name].extend(y_pred.tolist())

    # Mean accuracy per classifier  (TABLE II -- unchanged)
    mean_results = {name: np.mean(accs) for name, accs in results.items()}

    # Sort everything back into original epoch order so epoch indices are monotonic
    _sort = np.argsort(_cv_epoch_idx)
    cv_info = {
        'y_true'    : np.array(_cv_y_true)[_sort],
        'y_pred'    : {n: np.array(_cv_y_pred[n])[_sort] for n in CLASSIFIERS},
        'epoch_idx' : np.array(_cv_epoch_idx)[_sort],
        'X_csp'     : np.vstack(_cv_X_csp)[_sort],
        'csp_last'  : _csp_last,
        'ch_subset' : channel_subset,
    }

    return mean_results, cv_info


# =====================================================================
# MAIN PROCESSING FUNCTION (per subject)
# =====================================================================

def process_single_file(file_path, subject_label=None):
    """
    Process a single EEG CSV file through the full pipeline.

    Returns
    -------
    subject_results : dict
        {config_name: {clf_name: mean_accuracy, ...}, ...}
    rejected_epochs : int
    total_epochs_raw : int
    clean_epochs : int
    """
    print(f"\n  Loading: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df['direction'] = df['direction'].astype(str).str.lower().str.strip()

    # --- Bandpass Filter ---
    filtered_data = apply_bandpass(df[all_channels].values, 8, 30, FS)
    df_filtered = df.copy()
    df_filtered[all_channels] = filtered_data

    # --- Epoch Extraction ---
    df_filtered['block_id'] = (
        df_filtered['direction'] != df_filtered['direction'].shift()
    ).cumsum()

    left_epochs_raw  = []
    right_epochs_raw = []

    for _, group in df_filtered.groupby('block_id'):
        direction = group['direction'].iloc[0]
        if direction == 'left' and len(group) >= DECISION_END:
            left_epochs_raw.append(group[all_channels].iloc[DECISION_START:DECISION_END].values)
        elif direction == 'right' and len(group) >= DECISION_END:
            right_epochs_raw.append(group[all_channels].iloc[DECISION_START:DECISION_END].values)

    all_epochs_raw = left_epochs_raw + right_epochs_raw
    all_labels_raw = [0] * len(left_epochs_raw) + [1] * len(right_epochs_raw)
    total_epochs_raw = len(all_epochs_raw)

    if total_epochs_raw < 2:
        print(f"    [!] Only {total_epochs_raw} epochs extracted. Skipping.")
        return None, 0, total_epochs_raw, 0

    # --- FastICA Artifact Removal ---
    concat_epochs = np.vstack(all_epochs_raw)
    ica = FastICA(n_components=len(all_channels), random_state=42, max_iter=1000)
    S_components = ica.fit_transform(concat_epochs)

    fp1_signal = concat_epochs[:, all_channels.index('Fp1')]
    fp2_signal = concat_epochs[:, all_channels.index('Fp2')]

    corrs_fp1 = np.abs([np.corrcoef(S_components[:, i], fp1_signal)[0, 1]
                        for i in range(S_components.shape[1])])
    corrs_fp2 = np.abs([np.corrcoef(S_components[:, i], fp2_signal)[0, 1]
                        for i in range(S_components.shape[1])])

    bad_components = np.where((corrs_fp1 > 0.5) | (corrs_fp2 > 0.5))[0]
    S_clean = S_components.copy()
    S_clean[:, bad_components] = 0
    clean_concat = ica.inverse_transform(S_clean)

    epochs_clean = []
    start_idx = 0
    for _ in range(len(all_epochs_raw)):
        epochs_clean.append(clean_concat[start_idx : start_idx + DECISION_SAMPLES, :])
        start_idx += DECISION_SAMPLES

    # --- Artifact Rejection ---
    ptp_threshold = np.percentile(np.ptp(filtered_data, axis=1), 99) * 2

    X_all_data = []
    y = []
    rejected_epochs = 0

    # for idx, (epoch, label) in enumerate(zip(epochs_clean, all_labels_raw)):
    #     if np.max(np.ptp(epoch, axis=0)) < ptp_threshold:
    #         X_all_data.append(epoch.T)  # CSP expects (Channels x Samples)
    #         y.append(label)
    #     else:
    #         rejected_epochs += 1


    for idx, (epoch, label) in enumerate(zip(epochs_clean, all_labels_raw)):

        # Convert to (channels x samples)
        epoch_t = epoch.T

        # 1. Amplitude artifact check (existing)
        if np.max(np.ptp(epoch_t, axis=1)) >= ptp_threshold:
            rejected_epochs += 1
            continue

        # 2. NEW: ERD + lateralization check
        if not check_lateralization(epoch_t, label):
            rejected_epochs += 1
            continue

        X_all_data.append(epoch_t)
        y.append(label)

    y = np.array(y)
    clean_epoch_count = len(y)

    if clean_epoch_count < 4 or len(np.unique(y)) < 2:
        print(f"    [!] Not enough clean epochs ({clean_epoch_count}) or classes. Skipping.")
        return None, rejected_epochs, total_epochs_raw, clean_epoch_count

    print(f"    Epochs: {total_epochs_raw} total -> {clean_epoch_count} clean, "
          f"{rejected_epochs} rejected | ICA bad: {bad_components}")

    # --- Classification across all 3 channel configs ---
    subject_results     = {}
    cv_test_per_config  = {}   # collects CV test-fold data for topoplots
    for config_name, (ch_subset, n_comp) in CHANNEL_CONFIGS.items():
        mean_accs, cv_info = classify_with_kfold(X_all_data, y, ch_subset, n_comp)
        subject_results[config_name]    = mean_accs
        cv_test_per_config[config_name] = cv_info          # store test-fold results
        best_clf = max(mean_accs, key=mean_accs.get)
        print(f"    {config_name}: best={best_clf} ({mean_accs[best_clf]:.2f})")

    # ----------------------------------------------------------------
    # ADDED: Generate topoplots using CV TEST data only
    # (pure visualization -- does not affect any classification result)
    # ----------------------------------------------------------------
    _subj_lbl = subject_label if subject_label else os.path.splitext(os.path.basename(file_path))[0]
    generate_topoplots_for_subject(
        _subj_lbl, X_all_data, y,
        cv_test_per_config=cv_test_per_config
    )
    # ----------------------------------------------------------------

    return subject_results, rejected_epochs, total_epochs_raw, clean_epoch_count


# =====================================================================
# BATCH RUNNER & TABLE GENERATION
# =====================================================================

def run_batch():
    """Process all files and generate TABLE II."""

    print("=" * 80)
    print("  EEG MOTOR IMAGERY -- BATCH PROCESSING PIPELINE")
    print("  Generating TABLE II: Classification Accuracies")
    print("=" * 80)
    print(f"  Files to process: {len(EEG_FILES)}")
    print(f"  Channel configs:  {list(CHANNEL_CONFIGS.keys())}")
    print(f"  Classifiers:      {CLF_NAMES}")
    print(f"  K-Folds:          {N_FOLDS}")
    print("=" * 80)

    # Storage for all results
    all_results = []       # list of dicts per subject
    subject_labels = []
    rejected_list = []
    total_epoch_list = []
    clean_epoch_list = []

    start_time = time.time()

    for i, (label, filename) in enumerate(EEG_FILES):
        file_path = os.path.join(BASE_DIR, filename)

        if not os.path.exists(file_path):
            # Also check in MI BCI subdirectory
            alt_path = os.path.join(BASE_DIR, "MI BCI", filename)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                print(f"\n  [ERROR] File not found: {filename}")
                print(f"          Looked in: {file_path}")
                print(f"          Also tried: {alt_path}")
                continue

        print(f"\n{'-' * 60}")
        print(f"  [{i+1}/{len(EEG_FILES)}] Processing {label}: {filename}")
        print(f"{'-' * 60}")

        try:
            results, rejected, total_raw, clean_count = process_single_file(file_path, label)

            if results is not None:
                all_results.append(results)
                subject_labels.append(label)
                rejected_list.append(rejected)
                total_epoch_list.append(total_raw)
                clean_epoch_list.append(clean_count)
            else:
                print(f"  [SKIP] {label} -- insufficient data")

        except Exception as e:
            print(f"  [ERROR] {label} failed: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time

    if not all_results:
        print("\n[!] No files were successfully processed.")
        return

    # =================================================================
    # BUILD TABLE II
    # =================================================================
    print("\n\n")
    print("=" * 100)
    print("  TABLE II")
    print("  CLASSIFICATION TEST ACCURACIES ACROSS CHANNEL CONFIGURATIONS")
    print("=" * 100)

    config_names = list(CHANNEL_CONFIGS.keys())
    n_configs = len(config_names)
    n_clfs = len(CLF_NAMES)

    # -- Build DataFrame --
    rows = []
    for i, label in enumerate(subject_labels):
        row = {
            'Subject': label,
            'Total Epochs': total_epoch_list[i],
            'Clean Epochs': clean_epoch_list[i],
            'Rejected Epochs': rejected_list[i],
        }
        for config in config_names:
            for clf in CLF_NAMES:
                col_name = f"{config}|{clf}"
                row[col_name] = all_results[i][config][clf]
        rows.append(row)

    df_results = pd.DataFrame(rows)

    # -- Compute Mean and SD rows --
    accuracy_cols = [c for c in df_results.columns if '|' in c]
    mean_row = {'Subject': 'Mean', 'Total Epochs': '', 'Clean Epochs': '', 'Rejected Epochs': ''}
    sd_row   = {'Subject': 'SD',   'Total Epochs': '', 'Clean Epochs': '', 'Rejected Epochs': ''}

    for col in accuracy_cols:
        mean_row[col] = df_results[col].mean()
        sd_row[col]   = df_results[col].std()

    df_table = pd.concat([df_results, pd.DataFrame([mean_row, sd_row])],
                          ignore_index=True)

    # -- Print Formatted Console Table --
    # Header line 1: Config group names
    header1 = f"  {'Subject':<10} {'Rej.':<5}"
    for config in config_names:
        config_short = config.replace(" (Fixation Off)", "")
        w = n_clfs * 7
        header1 += f" | {config_short:^{w}}"
    print(header1)

    # Header line 2: Classifier names
    header2 = f"  {'':10} {'':5}"
    for config in config_names:
        header2 += " |"
        for clf in CLF_NAMES:
            header2 += f" {clf:>6}"
    print(header2)

    # Separator
    total_w = 15 + n_configs * (1 + n_clfs * 7)
    print(f"  {'-' * total_w}")

    # Data rows
    for _, row in df_table.iterrows():
        subj = row['Subject']
        rej  = row['Rejected Epochs']
        if isinstance(rej, (int, float)) and rej != '':
            rej_str = str(int(rej))
        else:
            rej_str = ''

        line = f"  {str(subj):<10} {rej_str:<5}"
        for config in config_names:
            line += " |"
            for clf in CLF_NAMES:
                col = f"{config}|{clf}"
                val = row[col]
                if isinstance(val, (int, float)):
                    line += f" {val:>5.2f}"
                else:
                    line += f" {str(val):>5}"
        print(line)

        # Print separator before Mean row
        if subj == subject_labels[-1]:
            print(f"  {'-' * total_w}")

    # -- Save to CSV --
    csv_rows = []
    for _, row in df_table.iterrows():
        csv_row = {
            'Subject': row['Subject'],
            'Total Epochs': row['Total Epochs'],
            'Clean Epochs': row['Clean Epochs'],
            'Rejected Epochs': row['Rejected Epochs'],
        }
        for config in config_names:
            for clf in CLF_NAMES:
                col_in  = f"{config}|{clf}"
                config_short = config.replace(" (Fixation Off)", "")
                col_out = f"{config_short} - {clf}"
                val = row[col_in]
                if isinstance(val, (int, float)):
                    csv_row[col_out] = round(val, 2)
                else:
                    csv_row[col_out] = val
        csv_rows.append(csv_row)

    df_csv = pd.DataFrame(csv_rows)
    output_csv = os.path.join(os.getcwd(), 'table_ii_results.csv')  # FIXED: replaced __file__
    df_csv.to_csv(output_csv, index=False)
    print(f"\n  [SAVED] {output_csv}")

    # -- Save as publication-quality table image --
    save_table_image(df_table, config_names, CLF_NAMES, subject_labels)

    print(f"\n  Total processing time: {elapsed:.1f}s")
    print("=" * 100)

    return df_table


def save_table_image(df_table, config_names, clf_names, subject_labels):
    """Generate a publication-quality TABLE II image."""
    n_configs = len(config_names)
    n_clfs = len(clf_names)
    n_subjects = len(subject_labels)
    n_rows = len(df_table)  # subjects + Mean + SD

    # Figure sizing
    col_width = 0.65
    row_height = 0.4
    extra_cols = 2  # Subject + Rejected
    total_cols = extra_cols + n_configs * n_clfs
    fig_w = total_cols * col_width + 1.5
    fig_h = (n_rows + 3) * row_height + 1.0  # +3 for header rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    ax.set_xlim(0, total_cols)
    ax.set_ylim(0, n_rows + 3)

    # Colors
    header_bg     = '#1a237e'
    header_text   = 'white'
    subheader_bg  = '#283593'
    row_bg_even   = '#f5f5f5'
    row_bg_odd    = '#ffffff'
    mean_bg       = '#e8eaf6'
    best_color    = '#1b5e20'
    border_color  = '#9e9e9e'

    y_top = n_rows + 3

    # -- Title --
    ax.text(total_cols / 2, y_top - 0.1, 'TABLE II', ha='center', va='top',
            fontsize=14, fontweight='bold', fontfamily='serif')
    ax.text(total_cols / 2, y_top - 0.5,
            'Classification Test Accuracies Across Channel Configurations',
            ha='center', va='top', fontsize=10, fontfamily='serif', style='italic')

    # -- Row 1: Config group headers --
    y_h1 = y_top - 1.2
    # Subject + Rejected header
    ax.add_patch(plt.Rectangle((0, y_h1 - 0.4), 1, 0.8,
                                facecolor=header_bg, edgecolor=border_color, lw=0.5))
    ax.text(0.5, y_h1, 'Subject', ha='center', va='center',
            fontsize=8, fontweight='bold', color=header_text, fontfamily='serif')

    ax.add_patch(plt.Rectangle((1, y_h1 - 0.4), 1, 0.8,
                                facecolor=header_bg, edgecolor=border_color, lw=0.5))
    ax.text(1.5, y_h1, 'Rejected\nEpochs', ha='center', va='center',
            fontsize=7, fontweight='bold', color=header_text, fontfamily='serif')

    x_start = extra_cols
    for ci, config in enumerate(config_names):
        w = n_clfs
        config_short = config.replace(" (Fixation Off)", "")
        ax.add_patch(plt.Rectangle((x_start, y_h1 - 0.4), w, 0.8,
                                    facecolor=header_bg, edgecolor=border_color, lw=0.5))
        ax.text(x_start + w / 2, y_h1, config_short, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=header_text, fontfamily='serif')
        x_start += w

    # -- Row 2: Classifier sub-headers --
    y_h2 = y_top - 2.0
    ax.add_patch(plt.Rectangle((0, y_h2 - 0.4), extra_cols, 0.8,
                                facecolor=subheader_bg, edgecolor=border_color, lw=0.5))

    x_start = extra_cols
    for ci, config in enumerate(config_names):
        for cli, clf in enumerate(clf_names):
            ax.add_patch(plt.Rectangle((x_start, y_h2 - 0.4), 1, 0.8,
                                        facecolor=subheader_bg, edgecolor=border_color, lw=0.5))
            ax.text(x_start + 0.5, y_h2, clf, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=header_text, fontfamily='serif')
            x_start += 1

    # -- Data Rows --
    # Find best accuracy per config for highlighting
    best_per_config = {}
    for config in config_names:
        for clf in clf_names:
            col = f"{config}|{clf}"
            vals = df_table[col].iloc[:n_subjects]
            if all(isinstance(v, (int, float)) for v in vals):
                best_per_config[(config, clf)] = vals.max()

    for ri in range(n_rows):
        y_row = y_top - 2.8 - ri * 0.8
        row = df_table.iloc[ri]
        subj = str(row['Subject'])

        # Row background
        if subj in ('Mean', 'SD'):
            bg = mean_bg
            fw = 'bold'
        elif ri % 2 == 0:
            bg = row_bg_even
            fw = 'normal'
        else:
            bg = row_bg_odd
            fw = 'normal'

        # Subject cell
        ax.add_patch(plt.Rectangle((0, y_row - 0.4), 1, 0.8,
                                    facecolor=bg, edgecolor=border_color, lw=0.3))
        ax.text(0.5, y_row, subj, ha='center', va='center',
                fontsize=8, fontweight='bold', fontfamily='serif')

        # Rejected epochs cell
        rej = row['Rejected Epochs']
        rej_str = str(int(rej)) if isinstance(rej, (int, float)) and rej != '' else ''
        ax.add_patch(plt.Rectangle((1, y_row - 0.4), 1, 0.8,
                                    facecolor=bg, edgecolor=border_color, lw=0.3))
        ax.text(1.5, y_row, rej_str, ha='center', va='center',
                fontsize=8, fontfamily='serif')

        # Accuracy cells
        x_start = extra_cols
        for config in config_names:
            for clf in clf_names:
                col = f"{config}|{clf}"
                val = row[col]

                cell_bg = bg
                text_color = 'black'
                fw_cell = fw

                # Highlight best values for subject rows
                if isinstance(val, (int, float)) and subj not in ('Mean', 'SD'):
                    if (config, clf) in best_per_config:
                        if abs(val - best_per_config[(config, clf)]) < 1e-6:
                            text_color = best_color
                            fw_cell = 'bold'

                ax.add_patch(plt.Rectangle((x_start, y_row - 0.4), 1, 0.8,
                                            facecolor=cell_bg, edgecolor=border_color, lw=0.3))
                if isinstance(val, (int, float)):
                    ax.text(x_start + 0.5, y_row, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, fontweight=fw_cell, color=text_color, fontfamily='serif')
                x_start += 1

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(os.getcwd(), 'table_ii_results.png')  # FIXED: replaced __file__
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  [SAVED] {out_path}")


# =====================================================================
# >>>  ADDED SECTION: ERD/ERS + CSP TOPOPLOT VISUALIZATIONS  <<<
# =====================================================================
# All functions below are purely additive. No existing code is altered.
# Called via generate_topoplots_for_subject() at end of process_single_file().
#
# Per subject + per channel config, two figures are produced:
#
#  FIGURE A — Per-Epoch ERD Topomap Grid
#    Every clean epoch rendered as its own ERD/ERS topomap.
#    Sorted: Left MI epochs first, then Right MI epochs.
#    Each cell title shows:  "Ep N  |  True: Left  |  Pred: Left  ✓"
#    Cell border: green = correctly classified,  red = misclassified.
#
#  FIGURE B — ERD + CSP by True/Predicted Label Category
#    4 row × (1 + n_comp) column grid:
#      Rows  → classification outcome:
#              True:Left  → Pred:Left  ✓   (True Positive Left)
#              True:Left  → Pred:Right ✗   (False Negative Left)
#              True:Right → Pred:Right ✓   (True Positive Right)
#              True:Right → Pred:Left  ✗   (False Negative Right)
#      Col 0 → averaged ERD/ERS topomap for all epochs in that row category
#      Cols 1…n → CSP activation-pattern topomaps (Haufe et al. 2014)
#    Row labels coloured green (correct) / red (wrong).
# =====================================================================


def _plot_topomap(data, info, ax, vmin, vmax, **kwargs):
    """
    Version-safe wrapper around mne.viz.plot_topomap.
    MNE < 1.2  : accepts vmin / vmax as separate keyword arguments.
    MNE >= 1.2 : requires vlim=(vmin, vmax); separate args raise TypeError.
    """
    try:
        return mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            vlim=(vmin, vmax), **kwargs
        )
    except TypeError:
        return mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            vmin=vmin, vmax=vmax, **kwargs
        )


def _topo_info(channel_names):
    """MNE Info with standard 10-20 montage; missing channels silently ignored."""
    montage = mne.channels.make_standard_montage('standard_1020')
    info    = mne.create_info(ch_names=channel_names, sfreq=FS, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')
    return info


def _epoch_erd(epoch, baseline_samples=250):
    """
    Compute ERD/ERS (%) for a single epoch.
    epoch : np.ndarray (n_channels, n_samples)
    Returns np.ndarray (n_channels,)
    """
    bp = np.mean(epoch[:, :baseline_samples] ** 2, axis=1)
    ap = np.mean(epoch[:, baseline_samples:]  ** 2, axis=1)
    bp = np.where(bp < 1e-14, 1e-14, bp)
    return (ap - bp) / bp * 100.0


def _config_tag(config_name):
    """Filesystem-safe short tag from a config name."""
    return (config_name
            .replace(' (Fixation Off)', '')
            .replace(' ', '_')
            .replace('(', '').replace(')', ''))


def _csp_patterns(csp_obj):
    """
    Return CSP activation patterns using the Haufe et al. (2014) inversion.
    Falls back to raw filter rows if pseudo-inverse fails.
    Shape: (n_comp, n_channels)
    """
    try:
        return np.linalg.pinv(csp_obj.W).T
    except np.linalg.LinAlgError:
        return csp_obj.W.copy()


# ─────────────────────────────────────────────────────────────────────
# FIGURE A: Per-Epoch ERD Topomap Grid
# ─────────────────────────────────────────────────────────────────────

def plot_epoch_erd_topogrid(X_sub, y_true, y_pred, channel_subset,
                             subject_label, config_name, out_dir=None):
    """
    One topomap cell per epoch, arranged in a grid.

    Layout
    ------
    • Epochs sorted: all Left-MI epochs first, then all Right-MI epochs.
    • Up to 8 columns per row.
    • Cell title  : "Ep N | True: Left | Pred: Left ✓"
    • Cell border : green = correct,  red = misclassified.
    • Shared symmetric colour scale (95th-percentile of |ERD| across epochs).

    Saved as: <subject_label>_A_epoch_erd_grid_<config_tag>.png
    """
    info     = _topo_info(channel_subset)
    n_epochs = len(X_sub)
    correct  = (y_true == y_pred)
    lbl      = {0: 'Left', 1: 'Right'}

    # Sort: left first, then right
    left_idx  = np.where(y_true == 0)[0]
    right_idx = np.where(y_true == 1)[0]
    order     = np.concatenate([left_idx, right_idx])

    # Pre-compute ERD for each epoch in display order
    all_erd = [_epoch_erd(X_sub[i]) for i in order]
    flat    = np.concatenate(all_erd)
    vmax    = max(float(np.percentile(np.abs(flat), 95)), 1.0)

    ncols   = min(8, n_epochs)
    nrows   = int(np.ceil(n_epochs / ncols))
    cell_w, cell_h = 2.5, 3.1
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * cell_w, nrows * cell_h + 1.4))
    axes = np.array(axes).reshape(nrows, ncols)

    config_short = config_name.replace(' (Fixation Off)', '')

    for pi, ep_i in enumerate(order):
        r, c  = divmod(pi, ncols)
        ax    = axes[r, c]
        im, _ = _plot_topomap(all_erd[pi], info, ax,
                               vmin=-vmax, vmax=vmax,
                               cmap='RdBu_r', sensors=True, contours=4)

        t_lbl = lbl[int(y_true[ep_i])]
        p_lbl = lbl[int(y_pred[ep_i])]
        ok    = bool(correct[ep_i])
        bc    = '#1b5e20' if ok else '#b71c1c'   # green / red
        sym   = '✓' if ok else '✗'

        ax.set_title(
            f'Ep {ep_i + 1}\nTrue: {t_lbl} | Pred: {p_lbl} {sym}',
            fontsize=7.5, pad=3, color=bc, fontweight='bold'
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2.5)

        # Subtle background tint
        ax.set_facecolor('#f1faf1' if ok else '#fff1f1')

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm      = plt.cm.ScalarMappable(cmap='RdBu_r',
                                     norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('ERD / ERS (%)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Hide unused axes
    for pi in range(n_epochs, nrows * ncols):
        r, c = divmod(pi, ncols)
        axes[r, c].axis('off')

    # Left-group / Right-group annotation
    n_left = len(left_idx)
    if n_left > 0 and ncols > 0:
        sep_col = (n_left - 1) % ncols
        sep_row = (n_left - 1) // ncols
        # Grey divider line after last left epoch
        axes[sep_row, sep_col].spines['right'].set_edgecolor('#757575')
        axes[sep_row, sep_col].spines['right'].set_linewidth(3.0)
        axes[sep_row, sep_col].spines['right'].set_linestyle('--')

    acc_pct = correct.mean() * 100
    n_wrong = int((~correct).sum())
    fig.suptitle(
        f'{subject_label}  —  Per-Epoch ERD Topomaps  |  {config_short}\n'
        f'Total: {n_epochs} epochs  |  '
        f'Correct: {n_epochs - n_wrong}  |  '
        f'Misclassified: {n_wrong}  |  '
        f'Acc: {acc_pct:.1f}%     '
        f'[green border = correct  |  red border = wrong]',
        fontsize=10, fontweight='bold', fontfamily='serif'
    )

    plt.subplots_adjust(left=0.02, right=0.91, top=0.88,
                        bottom=0.02, wspace=0.05, hspace=0.55)
    fname = os.path.join(
        out_dir or os.getcwd(),
        f'{subject_label}_A_epoch_erd_grid_{_config_tag(config_name)}.png'
    )
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"    [TOPO-SAVE] {fname}")


# ─────────────────────────────────────────────────────────────────────
# FIGURE B: ERD + CSP Topomaps by True/Predicted Label Category
# ─────────────────────────────────────────────────────────────────────

def plot_csp_erd_by_label_category(X_sub, y_true, y_pred, csp_obj,
                                    channel_subset, subject_label,
                                    config_name, out_dir=None):
    """
    4-row × (1 + n_comp) column summary figure.

    Row definitions
    ---------------
    Row 0  True:Left  → Pred:Left  ✓   (correctly predicted Left MI)
    Row 1  True:Left  → Pred:Right ✗   (Left MI misclassified as Right)
    Row 2  True:Right → Pred:Right ✓   (correctly predicted Right MI)
    Row 3  True:Right → Pred:Left  ✗   (Right MI misclassified as Left)

    Column definitions
    ------------------
    Col 0        Averaged ERD/ERS topomap for epochs in this row category.
                 Title shows epoch count.  Empty if no epochs in category.
    Cols 1…n_comp  CSP activation-pattern topomaps (Haufe et al. 2014).
                   Same spatial patterns across rows; shown for spatial reference.
                   Column header (row 0 only): "CSP 1 (Right↑)", etc.

    Row labels on the y-axis: green for correct rows, red for error rows.

    Saved as: <subject_label>_B_csp_erd_by_label_<config_tag>.png
    """
    info     = _topo_info(channel_subset)
    n_comp   = csp_obj.W.shape[0]
    half     = n_comp // 2
    patterns = _csp_patterns(csp_obj)          # (n_comp, n_channels)
    vmax_csp = max(float(np.abs(patterns).max()), 1e-6)

    # CSP component column headers
    comp_headers = (
        [f'CSP {i+1}\n(Right ↑)' for i in range(half)] +
        [f'CSP {half + i+1}\n(Left ↑)' for i in range(n_comp - half)]
    )

    # 4 row categories: (true_cls, pred_cls, label_text, text_color, bg_color)
    categories = [
        (0, 0, 'True: Left   →   Pred: Left  ✓',  '#1b5e20', '#f1faf1'),
        (0, 1, 'True: Left   →   Pred: Right ✗',  '#b71c1c', '#fff1f1'),
        (1, 1, 'True: Right  →   Pred: Right ✓',  '#1b5e20', '#f1faf1'),
        (1, 0, 'True: Right  →   Pred: Left  ✗',  '#b71c1c', '#fff1f1'),
    ]

    n_rows = len(categories)
    n_cols = 1 + n_comp
    cell_w, cell_h = 2.9, 3.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w + 1.8, n_rows * cell_h + 1.6)
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    config_short = config_name.replace(' (Fixation Off)', '')
    fig.suptitle(
        f'{subject_label}  —  ERD / ERS  &  CSP Patterns by True / Predicted Label\n'
        f'{config_short}',
        fontsize=12, fontweight='bold', fontfamily='serif', y=1.02
    )

    for ri, (true_cls, pred_cls, row_lbl, lbl_color, row_bg) in enumerate(categories):
        mask    = (y_true == true_cls) & (y_pred == pred_cls)
        n_in    = int(mask.sum())
        ep_idxs = np.where(mask)[0]

        # ── Column 0: averaged ERD topomap ──────────────────────────
        ax0 = axes[ri, 0]
        ax0.set_facecolor(row_bg)

        if n_in > 0:
            erds    = [_epoch_erd(X_sub[i]) for i in ep_idxs]
            avg_erd = np.mean(erds, axis=0)
            vmax_e  = max(float(np.abs(avg_erd).max()), 1.0)
            im, _   = _plot_topomap(avg_erd, info, ax0,
                                     vmin=-vmax_e, vmax=vmax_e,
                                     cmap='RdBu_r', sensors=True, contours=4)
            cb = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.08)
            cb.set_label('ERD/ERS (%)', fontsize=7)
            cb.ax.tick_params(labelsize=6)
            ax0.set_title(f'Avg ERD / ERS\n(n = {n_in} epochs)',
                          fontsize=8.5, pad=4, fontfamily='serif')
        else:
            ax0.axis('off')
            ax0.set_facecolor(row_bg)
            ax0.text(0.5, 0.5, f'No epochs\n(n = 0)',
                     ha='center', va='center',
                     fontsize=10, color='#9e9e9e',
                     transform=ax0.transAxes)
            if ri == 0:
                ax0.set_title('Avg ERD / ERS', fontsize=8.5,
                              pad=4, fontfamily='serif')

        # Row label as y-axis label (left-hand side)
        ax0.set_ylabel(row_lbl, fontsize=9.5, color=lbl_color,
                       fontweight='bold', labelpad=10)

        # Thick coloured left border to reinforce row identity
        for spine in ax0.spines.values():
            spine.set_edgecolor(lbl_color)
            spine.set_linewidth(2.0)

        # ── Columns 1…n_comp: CSP activation patterns ───────────────
        for ci in range(n_comp):
            ax  = axes[ri, ci + 1]
            ax.set_facecolor(row_bg)
            im, _ = _plot_topomap(patterns[ci], info, ax,
                                   vmin=-vmax_csp, vmax=vmax_csp,
                                   cmap='RdBu_r', sensors=True, contours=3)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
            cb.ax.tick_params(labelsize=6)

            # Column headers on the first row only
            if ri == 0:
                ax.set_title(comp_headers[ci], fontsize=8.5,
                             pad=4, fontfamily='serif')

            # Subtle border matching row colour
            for spine in ax.spines.values():
                spine.set_edgecolor(lbl_color)
                spine.set_linewidth(1.2)

    plt.subplots_adjust(left=0.14, right=0.97, top=0.91,
                        bottom=0.03, wspace=0.35, hspace=0.55)
    fname = os.path.join(
        out_dir or os.getcwd(),
        f'{subject_label}_B_csp_erd_by_label_{_config_tag(config_name)}.png'
    )
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"    [TOPO-SAVE] {fname}")


# ─────────────────────────────────────────────────────────────────────
# MASTER DRIVER
# ─────────────────────────────────────────────────────────────────────

def generate_topoplots_for_subject(subject_label, X_all_data, y,
                                    cv_test_per_config=None, out_dir=None):
    """
    Called once per subject after classification.

    Uses **CV test-fold predictions only** (never training data) so every
    epoch's predicted label came from a classifier that never saw that epoch.

    For each of the 3 channel configurations, produces 2 PNG files:
      A  <subj>_A_epoch_erd_grid_<config>.png   — per-epoch ERD topomap grid
      B  <subj>_B_csp_erd_by_label_<config>.png — ERD + CSP by true/predicted category

    Output folder:  <cwd>/topoplots/<subject_label>/
    Total: 6 PNGs per subject.

    Parameters
    ----------
    subject_label      : str  e.g. "S1"
    X_all_data         : list of np.ndarray (16, n_samples) — all 16 channels, all epochs
    y                  : np.ndarray int,  0=Left MI, 1=Right MI  (all clean epochs)
    cv_test_per_config : dict  {config_name -> cv_info} as returned by classify_with_kfold
    out_dir            : override output directory (default: <cwd>/topoplots/<subject_label>)
    """
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'topoplots', subject_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"    ── Topoplot generation for {subject_label}  →  {out_dir} ──")

    for config_name, (ch_subset, n_comp) in CHANNEL_CONFIGS.items():
        try:
            # ── Pull CV test-fold data for this config ───────────────
            if cv_test_per_config is None or config_name not in cv_test_per_config:
                print(f"    [WARN] No CV test data for '{config_name}', skipping.")
                continue

            cv_info    = cv_test_per_config[config_name]
            epoch_idx  = cv_info['epoch_idx']   # original indices into X_all_data
            y_true_test = cv_info['y_true']      # shape (n_test_epochs,)
            csp_last   = cv_info['csp_last']     # CSP from last fold

            # Pick the best classifier by highest test accuracy among all CV epochs
            best_clf_name = max(
                cv_info['y_pred'].keys(),
                key=lambda n: accuracy_score(y_true_test, cv_info['y_pred'][n])
            )
            y_pred_test = cv_info['y_pred'][best_clf_name]
            test_acc    = accuracy_score(y_true_test, y_pred_test) * 100

            print(f"    {config_name}: best clf = {best_clf_name}  "
                  f"(CV test acc = {test_acc:.1f}%,  n_test_epochs = {len(epoch_idx)})")

            # ── Extract the channel-subset raw signals for TEST epochs only ─
            ch_idx   = [all_channels.index(ch) for ch in ch_subset]
            X_sub_test = [X_all_data[i][ch_idx, :] for i in epoch_idx]

            # ── FIGURE A: per-epoch ERD topomap grid (test epochs only) ─
            plot_epoch_erd_topogrid(
                X_sub_test, y_true_test, y_pred_test,
                ch_subset, subject_label, config_name, out_dir
            )

            # ── FIGURE B: ERD + CSP by true/predicted label category ────
            plot_csp_erd_by_label_category(
                X_sub_test, y_true_test, y_pred_test, csp_last,
                ch_subset, subject_label, config_name, out_dir
            )

        except Exception as exc:
            import traceback
            print(f"    [WARN] Topoplot for '{config_name}' skipped: {exc}")
            traceback.print_exc()


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    df_table = run_batch()

