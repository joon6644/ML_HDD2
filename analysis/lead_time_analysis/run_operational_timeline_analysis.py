import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

try:
    import torch
except ImportError:
    torch = None

import config
from data_loader import load_dataset
from evaluator import RollingEvaluator
from analysis_data_loader import load_threshold_map, load_analysis_model

MODEL_TITLES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU"
}

# Standard academic blue for prediction curve
STANDARD_LINE_COLOR = "#1f77b4"

# Panels draw the last year of each disk's history.
MAX_DAYS_PLOT = 365

# Each panel illustrates one category, so the representative disk is chosen for
# how legibly it shows that category rather than for the highest peak score.
#   On-time  : a lead time clearly inside H -- a case sitting on the boundary
#              would flip to Early if H moved by a single day.
#   Early    : an alarm that persists and keeps crossing the threshold on its way
#              to the failure, showing repeated alarms after the first one.
#   Cens.    : a probability that rises over the threshold and then falls back
#              under it, showing an alarm the later observations do not sustain.
ONTIME_TARGET_LEAD_TIME = 20
ONTIME_MIN_PEAK_RATIO = 2.0
EARLY_MIN_SUSTAIN = 0.6
CENSORED_MIN_PEAK_RATIO = 1.5


def _shape_stats(disk, threshold: float) -> dict:
    """Curve shape over the span the panel actually draws (the last
    MAX_DAYS_PLOT observations), so selection matches what the reader sees."""
    preds = np.asarray(disk['preds'])
    window = preds[-MAX_DAYS_PLOT:]
    alarms_in_window = np.flatnonzero(window >= threshold)
    all_alarms = np.flatnonzero(preds >= threshold)

    if len(alarms_in_window) > 0:
        after_first = window[alarms_in_window[0]:]
        above = (after_first >= threshold).astype(int)
        sustain = float(above.mean())
        volatility = float(np.std(after_first))
        # Times the curve drops under the threshold and comes back up: the
        # visual signature of an alarm that keeps re-firing.
        recrossings = int(np.sum(np.diff(above) == 1))
    else:
        sustain = volatility = 0.0
        recrossings = 0

    return {
        'peak': float(preds.max()),
        'last': float(preds[-1]),
        'sustain': sustain,
        'volatility': volatility,
        'recrossings': recrossings,
        # A disk observed for less than the window makes its panel span a
        # shorter period than the others, which invites reading the four
        # timelines on different time scales.
        'full_history': len(preds) >= MAX_DAYS_PLOT,
        # The panel annotates the first alarm it can see. If the disk's true
        # first alarm predates the window, that annotation would name a later
        # alarm instead, so such disks are only a fallback.
        'first_alarm_in_window': bool(len(all_alarms) > 0
                                      and all_alarms[0] >= max(0, len(preds) - MAX_DAYS_PLOT)),
    }


def _pick(candidates, key, condition=None):
    """Best candidate by `key`, preferring ones that satisfy `condition` and
    whose first alarm is visible. Both preferences fall back rather than
    returning nothing, so a panel is never dropped for lack of an ideal case."""
    if not candidates:
        return None
    pool = [c for c in candidates if c['first_alarm_in_window']] or candidates
    pool = [c for c in pool if c['full_history']] or pool
    if condition is not None:
        pool = [c for c in pool if condition(c)] or pool
    return max(pool, key=key)['disk']


def select_best_operational_samples(raw_preds, threshold: float, lead_time: int = None):
    if lead_time is None:
        lead_time = config.TARGET_LEAD_TIME
    ontime_candidates = []
    early_candidates = []
    cens_early_candidates = []
    missed_candidates = []

    for disk in raw_preds:
        has_failed = disk['has_failed']
        failure_date = pd.to_datetime(disk['failure_date']) if (has_failed and disk['failure_date'] is not None) else None
        dates = pd.to_datetime(disk['dates'])
        preds = disk['preds']

        alarm_mask = (preds >= threshold)
        alarm_indices = np.where(alarm_mask)[0]
        record = {'disk': disk, 'lead_time': None, **_shape_stats(disk, threshold)}

        if len(alarm_indices) > 0:
            first_alarm_idx = alarm_indices[0]
            first_alarm_date = dates[first_alarm_idx]
            if has_failed and failure_date is not None:
                days_to_fail = (failure_date - first_alarm_date).days
                record['lead_time'] = days_to_fail
                if 0 <= days_to_fail <= lead_time:
                    ontime_candidates.append(record)
                elif days_to_fail > lead_time:
                    early_candidates.append(record)
            else:
                cens_early_candidates.append(record)
        else:
            if has_failed:
                missed_candidates.append(record)

    return {
        # Nearest the target lead time, among cases whose signal clears the
        # threshold by a visible margin; ties go to the stronger peak.
        'On-time': _pick(
            ontime_candidates,
            key=lambda c: (-abs(c['lead_time'] - ONTIME_TARGET_LEAD_TIME), c['peak']),
            condition=lambda c: c['peak'] >= ONTIME_MIN_PEAK_RATIO * threshold,
        ),
        # Among alarms that persist, the one re-crossing the threshold most
        # often, so the panel shows an early alarm that keeps re-firing rather
        # than a single step up.
        'Early': _pick(
            early_candidates,
            key=lambda c: (c['recrossings'], c['volatility']),
            condition=lambda c: c['sustain'] >= EARLY_MIN_SUSTAIN,
        ),
        # Largest fall from peak back under the threshold.
        'Censored Early': _pick(
            cens_early_candidates,
            key=lambda c: c['peak'] - c['last'],
            condition=lambda c: c['peak'] >= CENSORED_MIN_PEAK_RATIO * threshold and c['last'] < threshold,
        ),
        # No alarm to shape the curve; the highest peak shows how close the
        # missed failure came to being detected.
        'Missed': _pick(missed_candidates, key=lambda c: c['peak']),
    }


def plot_operational_timelines(samples: dict, model_key: str, threshold: float, hdd_name: str, output_paths: list):
    m_title = MODEL_TITLES.get(model_key, model_key.upper())

    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#111111'
    plt.rcParams['axes.linewidth'] = 1.1

    sns.set_theme(style="ticks", palette="muted")
    # Height is set by the tallest curve, not by the probability range: the peaks
    # are 0.67 / 0.30 / 0.26 / 0.15, so an axis to 1.0 would leave the bottom
    # third of the page carrying every line. No suptitle -- the paper's caption
    # already names the figure, and repeating it here costs a full text line.
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.6), dpi=300, sharey=True)

    categories = [
        ('On-time', (0, 0), "(a) On-time Alarm"),
        ('Early', (0, 1), "(b) Early Alarm"),
        ('Censored Early', (1, 0), "(c) Censored Early Alarm"),
        ('Missed', (1, 1), "(d) Missed Failure")
    ]

    max_days_plot = MAX_DAYS_PLOT

    for key, (r, c), title in categories:
        ax = axes[r, c]
        disk_data = samples.get(key)

        if disk_data is None:
            ax.text(0.5, 0.5, f"No sample found for category: {key}", ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=12.5, fontweight='bold', pad=9, loc='left', color='#111111')
            ax.set_ylim(-0.02, 0.74)
            ax.set_ylabel("Prediction Probability" if c == 0 else "")
            sns.despine(ax=ax, top=True, right=True)
            continue

        dates = pd.to_datetime(disk_data['dates'])
        preds = disk_data['preds']
        has_failed = disk_data['has_failed']
        failure_date = pd.to_datetime(disk_data['failure_date']) if (has_failed and disk_data['failure_date'] is not None) else None

        # Truncate view to last 365 days for clean, standardized visualization
        if len(dates) > max_days_plot:
            dates = dates[-max_days_plot:]
            preds = preds[-max_days_plot:]

        # 1. Prediction Probability Curve P(t)
        ax.plot(
            dates, preds,
            color=STANDARD_LINE_COLOR, linewidth=1.8,
            label=r"Prediction Probability $P(t)$"
        )

        # 2. Shading above threshold for alarm regions
        alarm_mask = (preds >= threshold)
        alarm_indices = np.where(alarm_mask)[0]
        if len(alarm_indices) > 0:
            ax.fill_between(
                dates, threshold, preds,
                where=(preds >= threshold),
                color="#e41a1c", alpha=0.20, interpolate=True
            )
            # Mark First Alarm Event
            first_alarm_idx = alarm_indices[0]
            first_alarm_date = dates[first_alarm_idx]
            ax.axvline(
                first_alarm_date, color="#e41a1c", linestyle=":", linewidth=1.5,
                label="First Alarm"
            )

        # 3. Decision Threshold Line
        ax.axhline(
            threshold, color='#c00000', linestyle='--', linewidth=1.4,
            label=f"Decision Threshold ({threshold:.3f})"
        )

        # 4. Actual Failure Event Line (For Failed Disks: On-time, Early, Missed)
        if has_failed and failure_date is not None:
            ax.axvline(
                failure_date, color='#000000', linestyle='--', linewidth=1.5,
                label="Actual Failure"
            )

        # Subplot Title
        ax.set_title(
            title,
            fontsize=12.5, fontweight='bold', pad=9, loc='left', color='#111111'
        )

        ax.set_ylim(-0.02, 0.74)

        # Set 5 evenly spaced date ticks across the time range
        tick_indices = np.linspace(0, len(dates) - 1, 5, dtype=int)
        tick_dates = [dates[i] for i in tick_indices]
        ax.set_xticks(tick_dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=9.5)
        ax.set_xlabel("")

        if c == 0:
            ax.set_ylabel("Prediction Probability", fontsize=11, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

        ax.grid(True, axis="y", linestyle=":", alpha=0.25, color="#666666")
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    # One shared legend under the grid. Per-panel dates are dropped from the
    # labels: they differ by panel, and the body text already gives each
    # interval, so keeping them forced a box into panel (a)'s headroom.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03),
               ncol=4, fontsize=10.5, frameon=False, handlelength=1.8,
               columnspacing=2.0)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.32)

    for out_path in output_paths:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved updated operational category timeline image -> {output_paths[0]}")


def main():
    hdd_name = "HGST_20HUH721212ALN604"
    hdd_path = os.path.join(PROJECT_ROOT, "data", "splitted", hdd_name)
    threshold_map = load_threshold_map(seed=config.SEED)

    target_models = ["gru", "lgbm", "xgb", "lstm"]

    results_dir = os.path.join(PROJECT_ROOT, "results", "lead_time_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f" UPDATING OPERATIONAL LIFE-CYCLE TIMELINES (4 OPERATIONAL CATEGORIES) ")
    print("=" * 80)

    for m in target_models:
        model_upper = MODEL_TITLES[m]
        lookup_key = "LGBM" if m == "lgbm" else m.upper()
        thresh = threshold_map[(hdd_name, lookup_key)]

        train_df, val_df, test_df, features = load_dataset(hdd_path, model=m)
        is_sequence_model = (m in ['lstm', 'gru'])

        model = load_analysis_model(
            dataset=hdd_name,
            model_name=m,
            seed=config.SEED,
            features=features
        )

        model_type = 'pytorch_class' if is_sequence_model else m

        evaluator = RollingEvaluator(
            model=model,
            features=features,
            window_size=config.WINDOW_SIZE if is_sequence_model else 1,
            device='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu',
            model_type=model_type,
            seed=config.SEED
        )

        print(f"\n[Inference & Plotting] Model: {model_upper} | Proposed Threshold: {thresh:.4f}")
        raw_preds = evaluator.get_raw_predictions(test_df, lead_time=config.TARGET_LEAD_TIME)

        samples = select_best_operational_samples(raw_preds, thresh, lead_time=config.TARGET_LEAD_TIME)

        out1 = os.path.join(results_dir, f"operational_timeline_4quadrants_{model_upper}.png")
        plot_operational_timelines(samples, m, thresh, hdd_name, [out1])


if __name__ == "__main__":
    main()
