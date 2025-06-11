import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import matplotlib


# Mapping dictionaries for categorical variables
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    data = pd.read_csv(file_path)

    # Map categorical columns to numeric
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)

    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)

    if display:
        print("Raw data sample:")
        print(data.head())

    if prepare_for == 'sdt':
        # Group by participant, condition, and signal
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']

        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]

                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]

                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        sdt_df = pd.DataFrame(sdt_data)

        if display:
            print("\nSDT summary:")
            print(sdt_df.head())

        return sdt_df

    elif prepare_for == 'delta plots':
        # Prepare delta plot percentiles
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', *[f'p{p}' for p in PERCENTILES]])
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]

                if c_data.empty:
                    continue

                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])

                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                if len(accurate_rt) > 0:
                    dp_data = pd.concat([dp_data, pd.DataFrame({
                        'pnum': [pnum],
                        'condition': [condition],
                        'mode': ['accurate'],
                        **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                    })])

                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                if len(error_rt) > 0:
                    dp_data = pd.concat([dp_data, pd.DataFrame({
                        'pnum': [pnum],
                        'condition': [condition],
                        'mode': ['error'],
                        **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                    })])
        dp_data = dp_data.reset_index(drop=True)

        if display:
            print("\nDelta plot data sample:")
            print(dp_data.head())

        return dp_data


def apply_hierarchical_sdt_model(data):
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())

    with pm.Model() as sdt_model:
        mean_d_prime = pm.Normal('mean_d_prime', mu=0., sigma=1., shape=C)
        sd_d_prime = pm.HalfNormal('sd_d_prime', sigma=1.)

        mean_criterion = pm.Normal('mean_criterion', mu=0., sigma=1., shape=C)
        sd_criterion = pm.HalfNormal('sd_criterion', sigma=1.)

        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=sd_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=sd_criterion, shape=(P, C))

        # Probabilities
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)

        # Observed data likelihood
        pm.Binomial('hit_obs',
                    n=data['nSignal'].values,
                    p=hit_rate[data['pnum'].values - 1, data['condition'].values],
                    observed=data['hits'].values)

        pm.Binomial('fa_obs',
                    n=data['nNoise'].values,
                    p=false_alarm_rate[data['pnum'].values - 1, data['condition'].values],
                    observed=data['false_alarms'].values)

    return sdt_model


def plot_posteriors(trace, data):
    C = len(data['condition'].unique())
    fig, axs = plt.subplots(2, C, figsize=(4 * C, 8))
    for c in range(C):
        az.plot_posterior(trace.posterior['mean_d_prime'].sel(mean_d_prime_dim_0=c), ax=axs[0, c])
        axs[0, c].set_title(f"mean_d_prime: {CONDITION_NAMES[c]}")
        az.plot_posterior(trace.posterior['mean_criterion'].sel(mean_criterion_dim_0=c), ax=axs[1, c])
        axs[1, c].set_title(f"mean_criterion: {CONDITION_NAMES[c]}")

    plt.tight_layout()
    plt.savefig("posterior_distributions.png")
    plt.close(fig) 


import matplotlib.pyplot as plt

#used chatgpt to help refine and correct some errors in the draw delta plots code

def draw_delta_plots(data, pnum):
    print(f"\n[INFO] Drawing delta plots for participant {pnum}...")
    
    if 'mode' not in data.columns:
        print("[ERROR] 'mode' column missing in delta plot data.")
        return

    if 'condition' not in data.columns or 'pnum' not in data.columns:
        print("[ERROR] Missing required columns in delta plot data.")
        return

    data = data[data['pnum'] == pnum]

    if data.empty:
        print(f"[WARNING] No data found for participant {pnum}")
        return

    conditions = sorted(data['condition'].unique())
    n = len(conditions)

    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if i == j:
                axes[i, j].axis('off')
                continue

            overall_mask1 = (data['condition'] == c1) & (data['mode'] == 'overall')
            overall_mask2 = (data['condition'] == c2) & (data['mode'] == 'overall')

            if overall_mask1.sum() == 0 or overall_mask2.sum() == 0:
                print(f"[WARNING] No 'overall' RTs for conditions {c1} or {c2}")
                axes[i, j].axis('off')
                continue

            try:
                quantiles_c1 = data.loc[overall_mask1, [f'p{p}' for p in PERCENTILES]].values.flatten()
                quantiles_c2 = data.loc[overall_mask2, [f'p{p}' for p in PERCENTILES]].values.flatten()

                delta = quantiles_c2 - quantiles_c1

                print(f"[DEBUG] Delta ({CONDITION_NAMES.get(c2, c2)} - {CONDITION_NAMES.get(c1, c1)}): {delta}")

                axes[i, j].plot(PERCENTILES, delta, marker='o', linestyle='-')
                axes[i, j].axhline(0, color='gray', linestyle='--')
                axes[i, j].set_title(f"{CONDITION_NAMES.get(c2, c2)} - {CONDITION_NAMES.get(c1, c1)}")
                axes[i, j].set_xlabel("Percentile")
                axes[i, j].set_ylabel("RT difference (s)")
            except Exception as e:
                print(f"[ERROR] Failed to plot for c1={c1}, c2={c2}: {e}")
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Delta Plots for Participant {pnum}", fontsize=16, y=1.02)
    plt.savefig("delta_plot.png")
    plt.close(fig)




def main():
    file_path = 'data1.csv'

    # Step 1: Read and prepare data for SDT analysis
    sdt_data = read_data(file_path, prepare_for='sdt', display=True)

    # Step 2: Build and sample from hierarchical SDT model
    sdt_model = apply_hierarchical_sdt_model(sdt_data)

    print("Sampling the model... (this may take a few minutes)")
    with sdt_model:
        trace = pm.sample(1000, tune=1000, chains= 4, target_accept=0.95, return_inferencedata=True)

    # Step 3: Check convergence
    print("\nConvergence diagnostics:")
    print(az.summary(trace, var_names=['mean_d_prime', 'mean_criterion', 'sd_d_prime', 'sd_criterion']))

    # Step 4: Plot posterior distributions of group-level parameters
    plot_posteriors(trace, sdt_data)

    # Step 5: Prepare data for delta plots
    delta_data = read_data(file_path, prepare_for='delta plots', display=True)
    print("\n[DEBUG] Delta plot data shape:", delta_data.shape)
    print(delta_data.head())


    # Step 6: Draw delta plots for one participant (e.g., participant 1)
    participant_to_plot = sdt_data['pnum'].iloc[0]
    print(f"\nDrawing delta plots for participant {participant_to_plot} ...")
    draw_delta_plots(delta_data, participant_to_plot)


if __name__ == "__main__":
    main()
