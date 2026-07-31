import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AlarmLifetimeVisualizer:
    """
    Visualizer for disk evaluation post-analysis:
    1. Lead Time Distribution Plot
    2. Lifetime Total Alarm Counts per Disk (Failed Disks vs. Healthy Disks)
    """

    def __init__(self, raw_preds, threshold=0.5, lead_time=30):
        """
        raw_preds: List of dicts returned by RollingEvaluator.get_raw_predictions()
        """
        self.raw_preds = raw_preds
        self.threshold = threshold
        self.lead_time = lead_time
        self.stats_df = self._compute_disk_alarm_statistics()

    def _compute_disk_alarm_statistics(self):
        """
        Extracts per-disk lifetime statistics from rolling predictions.
        """
        records = []
        for disk in self.raw_preds:
            serial = disk['serial_number']
            has_failed = disk['has_failed']
            failure_date = disk['failure_date']
            dates = disk['dates']
            preds = disk['preds']

            total_days = len(preds)
            alarm_mask = (preds >= self.threshold)
            alarm_indices = np.where(alarm_mask)[0]
            total_alarms = len(alarm_indices)
            alarm_ratio = float(total_alarms / total_days) if total_days > 0 else 0.0

            first_alarm_date = None
            days_to_failure = None
            is_hit = False

            if total_alarms > 0:
                first_alarm_idx = alarm_indices[0]
                first_alarm_date = pd.to_datetime(dates[first_alarm_idx])
                if has_failed and failure_date is not None:
                    days_to_failure = (failure_date - first_alarm_date).days
                    if 0 <= days_to_failure <= self.lead_time:
                        is_hit = True

            records.append({
                'serial_number': serial,
                'has_failed': 'Failed' if has_failed else 'Healthy',
                'is_failed_bool': has_failed,
                'total_observation_days': total_days,
                'total_alarm_count': total_alarms,
                'alarm_ratio_pct': alarm_ratio * 100,
                'first_alarm_date': first_alarm_date,
                'failure_date': failure_date,
                'days_to_failure_at_alarm': days_to_failure,
                'is_hit': is_hit
            })
        return pd.DataFrame(records)

    def plot_lead_time_distribution(self, save_dir, filename_prefix="evaluation"):
        """
        Generates and saves Lead Time Distribution Plot for Hit Alarms.
        """
        os.makedirs(save_dir, exist_ok=True)
        hit_disks = self.stats_df[self.stats_df['is_hit'] == True].copy()
        lead_times = hit_disks['days_to_failure_at_alarm'].dropna().values

        # 1. Save Lead Time Distribution Data CSV
        lt_csv_path = os.path.join(save_dir, f"{filename_prefix}_lead_time_distribution.csv")
        cols_to_save = ['serial_number', 'failure_date', 'first_alarm_date', 'days_to_failure_at_alarm', 'total_alarm_count']
        hit_disks[cols_to_save].to_csv(lt_csv_path, index=False, encoding='utf-8-sig')

        # 2. Plot Lead Time Distribution
        plot_path = os.path.join(save_dir, f"{filename_prefix}_lead_time_plot.png")
        plt.figure(figsize=(10, 5))

        if len(lead_times) > 0:
            sns.histplot(lead_times, bins=min(30, max(5, len(np.unique(lead_times)))), kde=True, color='#2b5c8f', edgecolor='black')
            median_lt = float(np.median(lead_times))
            mean_lt = float(np.mean(lead_times))
            plt.axvline(median_lt, color='red', linestyle='--', linewidth=2, label=f'Median Lead Time: {median_lt:.1f} days')
            plt.axvline(mean_lt, color='orange', linestyle=':', linewidth=2, label=f'Mean Lead Time: {mean_lt:.1f} days')
            plt.legend(fontsize=11)
        else:
            plt.text(0.5, 0.5, 'No Hit Alarms Triggered', horizontalalignment='center', verticalalignment='center', fontsize=14)

        plt.title(f"Lead Time Distribution ({filename_prefix})", fontsize=13, fontweight='bold')
        plt.xlabel("Lead Time (Days to Failure at First Alarm)", fontsize=11)
        plt.ylabel("Disk Count", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"[Visualizer] Saved Lead Time CSV : {lt_csv_path}")
        print(f"[Visualizer] Saved Lead Time Plot: {plot_path}")
        return lt_csv_path, plot_path

    def plot_lifetime_alarm_counts(self, save_dir, filename_prefix="evaluation"):
        """
        Generates and saves lifetime alarm count distribution per disk (Failed vs Healthy).
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save Alarm Count Stats CSV
        csv_path = os.path.join(save_dir, f"{filename_prefix}_lifetime_alarm_counts.csv")
        self.stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 2. Plot Lifetime Alarm Count Comparison (Histogram + Boxplot grid)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        palette = {'Failed': '#d9534f', 'Healthy': '#5cb85c'}

        # Subplot 1: Distribution Histogram
        sns.histplot(
            data=self.stats_df,
            x='total_alarm_count',
            hue='has_failed',
            element='step',
            stat='count',
            common_norm=False,
            palette=palette,
            ax=axes[0],
            bins=30
        )
        axes[0].set_title("Lifetime Total Alarm Count per Disk Distribution", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Total Lifetime Alarm Count (Days)", fontsize=11)
        axes[0].set_ylabel("Disk Count", fontsize=11)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Subplot 2: Boxplot for Comparison
        sns.boxplot(
            data=self.stats_df,
            x='has_failed',
            y='total_alarm_count',
            palette=palette,
            ax=axes[1],
            width=0.4
        )
        axes[1].set_title("Lifetime Alarm Count Comparison (Failed vs Healthy)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Disk Category", fontsize=11)
        axes[1].set_ylabel("Total Lifetime Alarm Count (Days)", fontsize=11)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # Add Mean values as annotations on boxplot
        means = self.stats_df.groupby('has_failed')['total_alarm_count'].mean()
        medians = self.stats_df.groupby('has_failed')['total_alarm_count'].median()
        for idx, cat in enumerate(['Failed', 'Healthy']):
            if cat in means:
                axes[1].text(idx, medians[cat] + 0.5, f"Med: {medians[cat]:.0f}\nMean: {means[cat]:.1f}", 
                            horizontalalignment='center', size=10, color='black', weight='semibold')

        plt.suptitle(f"Disk Lifetime Alarm Frequency Analysis (Threshold = {self.threshold:.4f})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"{filename_prefix}_lifetime_alarm_counts_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Print summary of lifetime alarm counts
        print("\n" + "=" * 60)
        print("     DISK LIFETIME ALARM FREQUENCY SUMMARY      ")
        print("=" * 60)
        for cat in ['Failed', 'Healthy']:
            sub = self.stats_df[self.stats_df['has_failed'] == cat]
            if len(sub) > 0:
                print(f"[{cat} Disks] Total: {len(sub)} | Mean Alarms: {sub['total_alarm_count'].mean():.2f} days | Median Alarms: {sub['total_alarm_count'].median():.1f} days | Max: {sub['total_alarm_count'].max()} days")
        print("=" * 60)

        print(f"[Visualizer] Saved Lifetime Alarm CSV : {csv_path}")
        print(f"[Visualizer] Saved Lifetime Alarm Plot: {plot_path}")

        return csv_path, plot_path
