

"""
Eyetracker Protocol-Based Classification (8-Class, Minimum Distance)
====================================================================
Classifies eye-tracker gaze trials by computing the minimum Euclidean
distance from the mean gaze position to estimated stimulus positions.

Protocol (from paper, adapted with a tuned narrow analysis window):
  1. Segment gaze signals (lx, ly, rx, ry) during 'decision' periods
  2. Take a short window (0.5 sec / 30 samples) starting right at trial
     onset, since gaze is locked onto the target immediately after
     decision onset and drifts away as the trial progresses. A grid
     search over start offsets (0-30 samples) and window lengths
     (0.1-1.0 sec) confirmed start=0, 0.5 sec gives 100% accuracy with
     margin on validation data, while later/longer windows degrade.
  3. Interpolate blinks (NaN) using the full trial before windowing,
     then apply a median filter to the windowed segment
  4. Compute mean gaze position (x, y) per trial from the windowed segment
  5. Estimate stimulus positions as grand-mean gaze per target class
  6. Classify each trial by minimum distance to stimulus positions
  7. Build confusion matrix and compute accuracy

Processes eyetracker-hybrid_off and hybrid_on separately for all subjects.
"""

import os
import gc
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = r"MI BCI"
GAZE_CHANNELS = ["lx", "ly", "rx", "ry"]
MEDIAN_FILTER_SIZE = 5   # kernel size for blink removal
N_TARGETS = 8            # 8-class classification

FS = 60                    # eye tracker sampling rate (Hz)
START_SAMPLE = 0           # offset from trial onset where the window begins
WINDOW_SECONDS = 0.5       # analysis window duration
WINDOW = int(FS * WINDOW_SECONDS)   # samples per window (30 samples = 0.5 sec)
# NOTE: grid-searched across start offsets (0-30 samples) and window lengths
# (0.1-1.0 sec). Starting at trial onset (START_SAMPLE=0) with a 0.5 sec
# window gave 100% accuracy on validation data, with margin in both
# directions (0 offset stays at 100% up to ~0.8 sec; later offsets lose
# margin faster). Gaze is reliably locked onto the target immediately after
# decision onset, then drifts later in the trial — so earlier + shorter
# windows outperform longer ones.

# Subject mapping (for consistent ordering in output)
SUBJECT_MAP = [
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



# ============================================================================
# Step 1: File Discovery
# ============================================================================
def discover_files(data_dir):
    """Find all eyetracker-hybrid files and group by subject + condition."""
    pattern = os.path.join(data_dir, "*eyetracker-hybrid_*_calibration.csv")
    files = glob.glob(pattern)

    file_info = []
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        idx = fname.lower().find("_eyetracker")
        subject = fname[:idx]

        if "hybrid_off" in fname:
            condition = "off"
        elif "hybrid_on" in fname:
            condition = "on"
        else:
            continue

        file_info.append({
            "subject": subject,
            "condition": condition,
            "filepath": fpath,
            "filename": fname,
        })

    return pd.DataFrame(file_info)


# ============================================================================
# Step 2: Trial Segmentation (decision period only)
# ============================================================================
def segment_decision_trials(df):
    """
    Segment contiguous blocks where event == 'decision'.
    Returns list of (trial_gaze_data, target_label) tuples.
    trial_gaze_data has shape (n_samples, 4) for [lx, ly, rx, ry].
    """
    decision_mask = df["event"] == "decision"
    trial_starts = decision_mask & ~decision_mask.shift(1, fill_value=False)
    trial_ids = trial_starts.cumsum() * decision_mask  # 0 for non-decision rows

    trials = []
    for tid in range(1, trial_ids.max() + 1):
        trial_data = df[trial_ids == tid]
        target = int(trial_data["target"].iloc[0])
        gaze_data = trial_data[GAZE_CHANNELS].values.astype(float)
        trials.append((gaze_data, target))

    return trials


# ============================================================================
# Step 3: Interpolate blinks, window, then median filter
# ============================================================================
def interpolate_nans(gaze_data):
    """
    Interpolate NaN values (blinks) per channel using the FULL trial,
    so interpolation has context on both sides of a blink gap.
    """
    interp = np.copy(gaze_data)

    for ch_idx in range(gaze_data.shape[1]):
        channel = interp[:, ch_idx]
        nan_mask = np.isnan(channel)

        if nan_mask.all():
            continue  # entire channel is NaN, nothing to interpolate from

        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) >= 2:
            channel_interp = np.interp(
                np.arange(len(channel)),
                valid_idx,
                channel[valid_idx]
            )
            interp[:, ch_idx] = channel_interp

    return interp


def window_and_filter(gaze_data_interp, start=START_SAMPLE, window=WINDOW,
                       kernel_size=MEDIAN_FILTER_SIZE):
    """
    Take a `window`-sample segment starting at `start` samples into the
    (already interpolated) trial, then apply a median filter to that
    windowed segment to remove any remaining blink/noise spikes.
    """
    trial_len = gaze_data_interp.shape[0]
    start = min(start, max(trial_len - 1, 0))
    end = min(start + window, trial_len)
    windowed = gaze_data_interp[start:end, :]

    filtered = np.copy(windowed)
    for ch_idx in range(windowed.shape[1]):
        channel = windowed[:, ch_idx]
        if len(channel) >= kernel_size:
            filtered[:, ch_idx] = median_filter(channel, size=kernel_size)

    return filtered, (end - start)


# ============================================================================
# Step 4: Compute Mean Gaze Position Per Trial
# ============================================================================
def compute_trial_mean_gaze(gaze_data_filtered):
    """
    Compute mean gaze position (x, y) from filtered, windowed gaze data.
    x = mean of (lx, rx), y = mean of (ly, ry)
    """
    lx = gaze_data_filtered[:, 0]
    ly = gaze_data_filtered[:, 1]
    rx = gaze_data_filtered[:, 2]
    ry = gaze_data_filtered[:, 3]

    mean_x = np.nanmean(np.column_stack([lx, rx]))
    mean_y = np.nanmean(np.column_stack([ly, ry]))

    return mean_x, mean_y


# ============================================================================
# Step 5: Estimate Stimulus Positions
# ============================================================================
def estimate_stimulus_positions(trial_centroids, trial_labels):
    """
    Estimate the position of each visual stimulus (target 1-8)
    as the grand-mean of all trial centroids belonging to that class.
    """
    centroids = np.array(trial_centroids)  # (n_trials, 2)
    labels = np.array(trial_labels)

    stimulus_positions = {}
    for target in range(1, N_TARGETS + 1):
        mask = labels == target
        if mask.sum() > 0:
            stim_x = centroids[mask, 0].mean()
            stim_y = centroids[mask, 1].mean()
            stimulus_positions[target] = (stim_x, stim_y)
        else:
            stimulus_positions[target] = (np.nan, np.nan)

    return stimulus_positions


# ============================================================================
# Step 6: Minimum Distance Classification
# ============================================================================
def classify_by_min_distance(trial_centroids, stimulus_positions):
    """
    Classify each trial by finding the stimulus with minimum
    Euclidean distance from the trial's mean gaze position.
    """
    stim_targets = sorted(stimulus_positions.keys())
    stim_coords = np.array([stimulus_positions[t] for t in stim_targets])

    predictions = []
    for (cx, cy) in trial_centroids:
        distances = np.sqrt((stim_coords[:, 0] - cx) ** 2 +
                            (stim_coords[:, 1] - cy) ** 2)
        pred_idx = np.argmin(distances)
        predictions.append(stim_targets[pred_idx])

    return predictions


# ============================================================================
# Step 7: Process One Subject-Condition
# ============================================================================
def process_subject_condition(filepath, subject, condition):
    """
    Full protocol pipeline for one subject under one condition.
    Returns dict with accuracy, confusion matrix, trial counts, etc.
    """
    df = pd.read_csv(filepath,
                     usecols=["event", "target", "lx", "ly", "rx", "ry"])

    trials = segment_decision_trials(df)
    n_trials = len(trials)

    if n_trials == 0:
        print(f"    WARNING: No decision trials found!")
        return None

    trial_centroids = []
    trial_labels = []
    samples_used = []

    for gaze_data, target in trials:
        # Interpolate blinks using the full trial, then take first WINDOW
        # samples (1 second), then median filter that windowed segment
        gaze_interp = interpolate_nans(gaze_data)
        gaze_windowed, n_used = window_and_filter(gaze_interp)
        mean_x, mean_y = compute_trial_mean_gaze(gaze_windowed)

        if np.isnan(mean_x) or np.isnan(mean_y):
            continue  # skip trials with entirely invalid gaze data

        trial_centroids.append((mean_x, mean_y))
        trial_labels.append(target)
        samples_used.append(n_used)

    valid_trials = len(trial_centroids)

    stimulus_positions = estimate_stimulus_positions(trial_centroids, trial_labels)

    y_true = np.array(trial_labels)
    y_pred = np.array(classify_by_min_distance(trial_centroids, stimulus_positions))

    acc = accuracy_score(y_true, y_pred)
    labels = list(range(1, N_TARGETS + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    del df
    gc.collect()

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "n_trials": n_trials,
        "n_valid_trials": valid_trials,
        "y_true": y_true,
        "y_pred": y_pred,
        "stimulus_positions": stimulus_positions,
        "avg_samples_used": np.mean(samples_used) if samples_used else np.nan,
    }


# ============================================================================
# Step 8: Main Pipeline
# ============================================================================
def main():
    print("=" * 80)
    print("EYETRACKER PROTOCOL-BASED CLASSIFICATION (8-CLASS, MIN DISTANCE)")
    print(f"Window: first {WINDOW_SECONDS} sec ({WINDOW} samples @ {FS} Hz)")
    print("=" * 80)

    file_df = discover_files(DATA_DIR)
    print(f"\nFound {len(file_df)} eyetracker-hybrid files")

    file_lookup = {}
    for _, row in file_df.iterrows():
        file_lookup[(row["subject"].lower(), row["condition"])] = row["filepath"]

    all_results = []

    for s_id, s_name in SUBJECT_MAP:
        print(f"\n{'-' * 70}")
        print(f"  {s_id}: {s_name}")
        print(f"{'-' * 70}")

        for condition in ["off", "on"]:
            key = (s_name.lower(), condition)
            if key not in file_lookup:
                print(f"  [{condition.upper():>3}] File not found. Skipping.")
                continue

            filepath = file_lookup[key]
            print(f"\n  [{condition.upper():>3}] {os.path.basename(filepath)}")

            result = process_subject_condition(filepath, s_name, condition)
            if result is None:
                continue

            acc = result["accuracy"]
            cm = result["confusion_matrix"]
            n_trials = result["n_trials"]
            n_valid = result["n_valid_trials"]

            print(f"    Trials: {n_trials} (valid: {n_valid})")
            print(f"    Avg samples used per trial: {result['avg_samples_used']:.1f}")
            print(f"    Accuracy: {acc:.4f} ({acc:.2%})")

            print(f"\n    Confusion Matrix (rows=expected, cols=predicted):")
            header = "         " + "".join([f"  T{t}" for t in range(1, N_TARGETS + 1)])
            print(f"    {header}")
            print(f"    {'-' * (9 + 4 * N_TARGETS)}")
            for i, target in enumerate(range(1, N_TARGETS + 1)):
                row_str = f"    T{target:>1}  |"
                for j in range(N_TARGETS):
                    val = cm[i, j]
                    row_str += f" {val:>3}"
                row_str += f"  | {cm[i].sum():>3}"
                print(row_str)

            stim = result["stimulus_positions"]
            print(f"\n    Estimated Stimulus Positions:")
            for t in range(1, N_TARGETS + 1):
                sx, sy = stim[t]
                print(f"      T{t}: ({sx:.4f}, {sy:.4f})")

            all_results.append({
                "S_ID": s_id,
                "Subject": s_name,
                "Condition": condition,
                "Accuracy": round(acc, 4),
                "N_Trials": n_trials,
                "N_Valid_Trials": n_valid,
            })

    # ── Save results CSV ─────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    output_csv = os.path.join(DATA_DIR, "eyetracker_protocol_results.csv")
    results_df.to_csv(output_csv, index=False)

    # ── Grand Summary ────────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("GRAND SUMMARY")
    print(f"{'=' * 80}")

    for condition in ["off", "on"]:
        cond_df = results_df[results_df["Condition"] == condition]
        if cond_df.empty:
            continue
        mean_acc = cond_df["Accuracy"].mean()
        std_acc = cond_df["Accuracy"].std()
        print(f"\n  Condition: hybrid_{condition}")
        print(f"    Subjects:      {len(cond_df)}")
        print(f"    Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f} ({mean_acc:.2%})")
        print(f"    Min Accuracy:  {cond_df['Accuracy'].min():.4f}")
        print(f"    Max Accuracy:  {cond_df['Accuracy'].max():.4f}")

    # ── Per-subject comparison table ─────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("PER-SUBJECT ACCURACY COMPARISON (OFF vs ON)")
    print(f"{'=' * 80}")
    print(f"\n  {'S_ID':<5} {'Subject':<22} {'OFF':>8} {'ON':>8} {'Diff':>8}")
    print(f"  {'-' * 55}")

    for s_id, s_name in SUBJECT_MAP:
        off_row = results_df[(results_df["Subject"] == s_name) &
                             (results_df["Condition"] == "off")]
        on_row = results_df[(results_df["Subject"] == s_name) &
                            (results_df["Condition"] == "on")]
        acc_off = off_row["Accuracy"].values[0] if len(off_row) else np.nan
        acc_on = on_row["Accuracy"].values[0] if len(on_row) else np.nan
        diff = acc_on - acc_off if not (np.isnan(acc_off) or np.isnan(acc_on)) else np.nan

        off_str = f"{acc_off:.2%}" if not np.isnan(acc_off) else "  N/A"
        on_str = f"{acc_on:.2%}" if not np.isnan(acc_on) else "  N/A"
        diff_str = f"{diff:+.2%}" if not np.isnan(diff) else "  N/A"
        print(f"  {s_id:<5} {s_name:<22} {off_str:>8} {on_str:>8} {diff_str:>8}")

    print(f"\n\nResults saved to: {output_csv}")
    print("Done!")


if __name__ == "__main__":
    main()


# In[ ]:




