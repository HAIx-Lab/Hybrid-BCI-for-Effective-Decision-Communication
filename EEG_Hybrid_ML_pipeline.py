#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.base import clone
import warnings
import os
import time

warnings.filterwarnings('ignore')

# =====================================================================
# Classifiers: SVM, RF, DT, KNN, NB
# =====================================================================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# =====================================================================
# CONFIGURATION -- EDIT THIS SECTION
# =====================================================================

# Base directory where CSV files are located
BASE_DIR = r'C:\Users\Admin\Downloads\MI BCI'

# List of EEG CSV files to process (hybrid_on_calibration).
# Each entry is (Label, Filename).
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

# Channel configurations (same as FILE 1 TABLE II)
all_channels   = ['Fp1','Fp2','F3','Fz','F4','T7','C3','Cz','C4','T8',
                  'P3','Pz','P4','PO7','Oz','PO8']
strict_motor   = ['C3','Cz','C4']
extended_motor = ['F3','Fz','F4','C3','Cz','C4','P3','Pz','P4']

CHANNEL_CONFIGS = {
    "All Channels (Hybrid Off)":              (all_channels, 6),
    "Strict Motor Cortex (Hybrid Off)":       (strict_motor, 2),
    "Extended Motor Cortex (Hybrid Off)":     (extended_motor, 6),
}

# =====================================================================
# Classifiers -- 10 from File 2 logic (unchanged)
# =====================================================================
CLASSIFIERS = {
    "SVM": SVC(kernel="rbf", random_state=42),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
    "DT":  DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NB":  GaussianNB(),
}

# =====================================================================
# Epoch constraints -- from File 2 logic (unchanged)
# =====================================================================
DECISION_START   = 300
DECISION_END     = 1300
DECISION_SAMPLES = DECISION_END - DECISION_START   # 1000 samples
REST_SAMPLES     = 1000
FS               = 500
N_FOLDS          = 5

CLF_NAMES = list(CLASSIFIERS.keys())


# =====================================================================
# HELPER FUNCTIONS -- from File 2 logic (unchanged)
# =====================================================================

def apply_bandpass(data, lowcut, highcut, fs, order=4):
    """Bandpass filter (8-30 Hz -- Mu + Beta rhythms)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


class LengthAgnosticCSP:
    """Length-Agnostic CSP -- from File 2 logic (unchanged)."""
    def __init__(self, n_components=6):
        self.n_components = n_components

    def fit(self, X_train, y_train):
        X0 = [X_train[i] for i in range(len(y_train)) if y_train[i] == 0]
        X1 = [X_train[i] for i in range(len(y_train)) if y_train[i] == 1]

        def get_avg_cov(trials):
            if not trials:
                return np.eye(self.n_components)
            covs = []
            for trial in trials:
                trial_c = trial - np.mean(trial, axis=1, keepdims=True)
                cov = np.dot(trial_c, trial_c.T) + np.eye(trial_c.shape[0]) * 1e-6
                covs.append(cov / np.trace(cov))
            return np.mean(covs, axis=0)

        eigenvalues, eigenvectors = eigh(get_avg_cov(X1), get_avg_cov(X0) + get_avg_cov(X1))
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        half = self.n_components // 2
        self.W = np.hstack([eigenvectors[:, :half], eigenvectors[:, -half:]]).T
        return self

    def transform(self, X_list):
        return np.array([
            np.log(np.var(np.dot(self.W, t), axis=1) / np.sum(np.var(np.dot(self.W, t), axis=1)))
            for t in X_list
        ])


def classify_with_kfold(X_all_data, y, channel_subset, n_comp, n_splits=N_FOLDS):
    """
    Run Stratified K-Fold CV for all 10 classifiers on a given channel config.
    Returns dict: {clf_name: mean_accuracy}
    """
    indices  = [all_channels.index(ch) for ch in channel_subset]
    X_subset = [trial[indices, :] for trial in X_all_data]
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {name: [] for name in CLASSIFIERS}

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

        for name, clf_template in CLASSIFIERS.items():
            clf = clone(clf_template)
            clf.fit(X_train_csp, y_train)
            y_pred = clf.predict(X_test_csp)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)

    mean_results = {name: np.mean(accs) for name, accs in results.items()}
    return mean_results


# =====================================================================
# MAIN PROCESSING FUNCTION (per subject)
# Uses File 2 logic EXACTLY -- decision vs rest, ICA on decision only
# =====================================================================

def process_single_file(file_path):
    """
    Process a single EEG hybrid_on CSV file through the full File 2 pipeline.

    Returns
    -------
    subject_results : dict  {config_name: {clf_name: mean_accuracy}}
    rejected_epochs : int
    total_epochs_raw : int   (decision + rest before rejection)
    clean_epochs : int
    """
    print(f"\n  Loading: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)

    # ----------------------------------------------------------------
    # Step 2: Bandpass Filter (File 2 logic -- unchanged)
    # ----------------------------------------------------------------
    filtered_data = apply_bandpass(df[all_channels].values, 8, 30, FS)
    df_filtered = df.copy()
    df_filtered[all_channels] = filtered_data

    # ----------------------------------------------------------------
    # Step 3: Initial Epoch Extraction (File 2 logic -- unchanged)
    # ----------------------------------------------------------------
    df_filtered['segment_id'] = (
        df_filtered['event'] != df_filtered['event'].shift()
    ).cumsum()

    decision_epochs_raw = []
    rest_epochs_raw     = []

    for _, group in df_filtered.groupby('segment_id'):
        event_type = group['event'].iloc[0]

        if event_type == 'decision' and len(group) >= DECISION_END:
            decision_epochs_raw.append(
                group[all_channels].iloc[DECISION_START:DECISION_END].values
            )
        elif event_type == 'rest' and len(group) >= REST_SAMPLES:
            rest_epochs_raw.append(
                group[all_channels].iloc[:REST_SAMPLES].values
            )

    total_epochs_raw = len(decision_epochs_raw) + len(rest_epochs_raw)
    print(f"    Extracted {len(decision_epochs_raw)} Decision + "
          f"{len(rest_epochs_raw)} Rest epochs ({total_epochs_raw} total)")

    if total_epochs_raw < 2 or len(decision_epochs_raw) == 0 or len(rest_epochs_raw) == 0:
        print(f"    [!] Insufficient epochs. Skipping.")
        return None, 0, total_epochs_raw, 0

    # ----------------------------------------------------------------
    # Step 4: Apply ICA ONLY to Decision Phase (File 2 logic -- unchanged)
    # ----------------------------------------------------------------
    concat_decision = np.vstack(decision_epochs_raw)

    ica = FastICA(n_components=len(all_channels), random_state=42, max_iter=1000)
    S_components = ica.fit_transform(concat_decision)

    fp1_signal = concat_decision[:, all_channels.index('Fp1')]
    fp2_signal = concat_decision[:, all_channels.index('Fp2')]

    corrs_fp1 = np.abs([np.corrcoef(S_components[:, i], fp1_signal)[0, 1]
                        for i in range(S_components.shape[1])])
    corrs_fp2 = np.abs([np.corrcoef(S_components[:, i], fp2_signal)[0, 1]
                        for i in range(S_components.shape[1])])

    bad_components = np.where((corrs_fp1 > 0.5) | (corrs_fp2 > 0.5))[0]

    S_clean = S_components.copy()
    S_clean[:, bad_components] = 0
    clean_decision_concat = ica.inverse_transform(S_clean)

    decision_epochs_clean = []
    start_idx = 0
    for _ in range(len(decision_epochs_raw)):
        decision_epochs_clean.append(
            clean_decision_concat[start_idx: start_idx + DECISION_SAMPLES, :]
        )
        start_idx += DECISION_SAMPLES

    # ----------------------------------------------------------------
    # Step 5: Threshold-based Artifact Rejection (File 2 logic -- unchanged)
    # ----------------------------------------------------------------
    ptp_threshold = np.percentile(np.ptp(filtered_data, axis=1), 99) * 2

    X_all_data = []
    y = []
    rejected_epochs = 0

    for epoch in decision_epochs_clean:
        if np.max(np.ptp(epoch, axis=0)) < ptp_threshold:
            X_all_data.append(epoch.T)   # (Channels x Samples)
            y.append(1)                  # decision = class 1
        else:
            rejected_epochs += 1

    for epoch in rest_epochs_raw:
        if np.max(np.ptp(epoch, axis=0)) < ptp_threshold:
            X_all_data.append(epoch.T)   # (Channels x Samples)
            y.append(0)                  # rest = class 0
        else:
            rejected_epochs += 1

    y = np.array(y)
    clean_epoch_count = len(y)

    if clean_epoch_count < 4 or len(np.unique(y)) < 2:
        print(f"    [!] Not enough clean epochs ({clean_epoch_count}) or classes. Skipping.")
        return None, rejected_epochs, total_epochs_raw, clean_epoch_count

    print(f"    Epochs: {total_epochs_raw} total -> {clean_epoch_count} clean, "
          f"{rejected_epochs} rejected | ICA bad: {bad_components}")

    # ----------------------------------------------------------------
    # Classification across all 3 channel configs
    # ----------------------------------------------------------------
    subject_results = {}
    for config_name, (ch_subset, n_comp) in CHANNEL_CONFIGS.items():
        mean_accs = classify_with_kfold(X_all_data, y, ch_subset, n_comp)
        subject_results[config_name] = mean_accs
        best_clf = max(mean_accs, key=mean_accs.get)
        print(f"    {config_name}: best={best_clf} ({mean_accs[best_clf]:.2f})")

    return subject_results, rejected_epochs, total_epochs_raw, clean_epoch_count


# =====================================================================
# BATCH RUNNER & TABLE GENERATION (from File 1 -- unchanged structure)
# =====================================================================

def run_batch():
    """Process all files and generate TABLE II."""

    print("=" * 80)
    print("  EEG MOTOR IMAGERY (HYBRID OFF) -- BATCH PROCESSING PIPELINE")
    print("  Generating TABLE II: Classification Accuracies")
    print("=" * 80)
    print(f"  Files to process: {len(EEG_FILES)}")
    print(f"  Channel configs:  {list(CHANNEL_CONFIGS.keys())}")
    print(f"  Classifiers:      {CLF_NAMES}")
    print(f"  K-Folds:          {N_FOLDS}")
    print("=" * 80)

    all_results      = []
    subject_labels   = []
    rejected_list    = []
    total_epoch_list = []
    clean_epoch_list = []

    start_time = time.time()

    for i, (label, filename) in enumerate(EEG_FILES):
        file_path = os.path.join(BASE_DIR, filename)

        if not os.path.exists(file_path):
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
            results, rejected, total_raw, clean_count = process_single_file(file_path)

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
    print("=" * 120)
    print("  TABLE II")
    print("  CLASSIFICATION TEST ACCURACIES ACROSS CHANNEL CONFIGURATIONS (HYBRID OFF)")
    print("=" * 120)

    config_names = list(CHANNEL_CONFIGS.keys())
    n_configs = len(config_names)
    n_clfs = len(CLF_NAMES)

    # -- Build DataFrame --
    rows = []
    for i, label in enumerate(subject_labels):
        row = {
            'Subject':         label,
            'Total Epochs':    total_epoch_list[i],
            'Clean Epochs':    clean_epoch_list[i],
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
    header1 = f"  {'Subject':<10} {'Rej.':<5}"
    for config in config_names:
        config_short = config.replace(" (Hybrid Off)", "")
        w = n_clfs * 7
        header1 += f" | {config_short:^{w}}"
    print(header1)

    header2 = f"  {'':10} {'':5}"
    for config in config_names:
        header2 += " |"
        for clf in CLF_NAMES:
            header2 += f" {clf:>6}"
    print(header2)

    total_w = 15 + n_configs * (1 + n_clfs * 7)
    print(f"  {'-' * total_w}")

    for _, row in df_table.iterrows():
        subj = row['Subject']
        rej  = row['Rejected Epochs']
        rej_str = str(int(rej)) if isinstance(rej, (int, float)) and rej != '' else ''

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

        if subj == subject_labels[-1]:
            print(f"  {'-' * total_w}")

    # -- Save to CSV --
    csv_rows = []
    for _, row in df_table.iterrows():
        csv_row = {
            'Subject':         row['Subject'],
            'Total Epochs':    row['Total Epochs'],
            'Clean Epochs':    row['Clean Epochs'],
            'Rejected Epochs': row['Rejected Epochs'],
        }
        for config in config_names:
            for clf in CLF_NAMES:
                col_in      = f"{config}|{clf}"
                config_short = config.replace(" (Hybrid On)", "")
                col_out     = f"{config_short} - {clf}"
                val = row[col_in]
                csv_row[col_out] = round(val, 2) if isinstance(val, (int, float)) else val
        csv_rows.append(csv_row)

    df_csv = pd.DataFrame(csv_rows)
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'table_ii_hybrid_off_results.csv')
    df_csv.to_csv(output_csv, index=False)
    print(f"\n  [SAVED] {output_csv}")

    # -- Save publication-quality table image --
    save_table_image(df_table, config_names, CLF_NAMES, subject_labels)

    print(f"\n  Total processing time: {elapsed:.1f}s")
    print("=" * 120)

    return df_table


def save_table_image(df_table, config_names, clf_names, subject_labels):
    """Generate a publication-quality TABLE II image."""
    n_configs  = len(config_names)
    n_clfs     = len(clf_names)
    n_subjects = len(subject_labels)
    n_rows     = len(df_table)   # subjects + Mean + SD

    col_width  = 0.65
    row_height = 0.4
    extra_cols = 2               # Subject + Rejected
    total_cols = extra_cols + n_configs * n_clfs
    fig_w = total_cols * col_width + 1.5
    fig_h = (n_rows + 3) * row_height + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    ax.set_xlim(0, total_cols)
    ax.set_ylim(0, n_rows + 3)

    header_bg    = '#1a237e'
    header_text  = 'white'
    subheader_bg = '#283593'
    row_bg_even  = '#f5f5f5'
    row_bg_odd   = '#ffffff'
    mean_bg      = '#e8eaf6'
    best_color   = '#1b5e20'
    border_color = '#9e9e9e'

    y_top = n_rows + 3

    # Title
    ax.text(total_cols / 2, y_top - 0.1, 'TABLE II', ha='center', va='top',
            fontsize=14, fontweight='bold', fontfamily='serif')
    ax.text(total_cols / 2, y_top - 0.5,
            'Classification Test Accuracies Across Channel Configurations (Hybrid Off)',
            ha='center', va='top', fontsize=10, fontfamily='serif', style='italic')

    # Row 1: Config group headers
    y_h1 = y_top - 1.2
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
        config_short = config.replace(" (Hybrid On)", "")
        ax.add_patch(plt.Rectangle((x_start, y_h1 - 0.4), w, 0.8,
                                    facecolor=header_bg, edgecolor=border_color, lw=0.5))
        ax.text(x_start + w / 2, y_h1, config_short, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=header_text, fontfamily='serif')
        x_start += w

    # Row 2: Classifier sub-headers
    y_h2 = y_top - 2.0
    ax.add_patch(plt.Rectangle((0, y_h2 - 0.4), extra_cols, 0.8,
                                facecolor=subheader_bg, edgecolor=border_color, lw=0.5))

    x_start = extra_cols
    for ci, config in enumerate(config_names):
        for cli, clf in enumerate(clf_names):
            ax.add_patch(plt.Rectangle((x_start, y_h2 - 0.4), 1, 0.8,
                                        facecolor=subheader_bg, edgecolor=border_color, lw=0.5))
            ax.text(x_start + 0.5, y_h2, clf, ha='center', va='center',
                    fontsize=7, fontweight='bold', color=header_text, fontfamily='serif')
            x_start += 1

    # Find best accuracy per (config, clf) for highlight
    best_per_config = {}
    for config in config_names:
        for clf in clf_names:
            col  = f"{config}|{clf}"
            vals = df_table[col].iloc[:n_subjects]
            if all(isinstance(v, (int, float)) for v in vals):
                best_per_config[(config, clf)] = vals.max()

    # Data rows
    for ri in range(n_rows):
        y_row = y_top - 2.8 - ri * 0.8
        row   = df_table.iloc[ri]
        subj  = str(row['Subject'])

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
        rej     = row['Rejected Epochs']
        rej_str = str(int(rej)) if isinstance(rej, (int, float)) and rej != '' else ''
        ax.add_patch(plt.Rectangle((1, y_row - 0.4), 1, 0.8,
                                    facecolor=bg, edgecolor=border_color, lw=0.3))
        ax.text(1.5, y_row, rej_str, ha='center', va='center',
                fontsize=8, fontfamily='serif')

        # Accuracy cells
        x_start = extra_cols
        for config in config_names:
            for clf in clf_names:
                col        = f"{config}|{clf}"
                val        = row[col]
                cell_bg    = bg
                text_color = 'black'
                fw_cell    = fw

                if isinstance(val, (int, float)) and subj not in ('Mean', 'SD'):
                    if (config, clf) in best_per_config:
                        if abs(val - best_per_config[(config, clf)]) < 1e-6:
                            text_color = best_color
                            fw_cell    = 'bold'

                ax.add_patch(plt.Rectangle((x_start, y_row - 0.4), 1, 0.8,
                                            facecolor=cell_bg, edgecolor=border_color, lw=0.3))
                if isinstance(val, (int, float)):
                    ax.text(x_start + 0.5, y_row, f'{val:.2f}', ha='center', va='center',
                            fontsize=7.5, fontweight=fw_cell, color=text_color,
                            fontfamily='serif')
                x_start += 1

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'table_ii_hybrid_off_results.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  [SAVED] {out_path}")


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    df_table = run_batch()

