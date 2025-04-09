import neptune.new as neptune 
import os
import numpy as np
import pandas as pd
bounds = [2, 4, 3, 1]
def get_param(project, param):
    if project[0].exists(param):
        return project[0][param].fetch()
    return np.nan

def bootstrap_ci(data, num_samples=10000, ci=95):
    """
    Estimates the confidence interval of the mean using bootstrapping.
    
    Parameters:
    - data (array-like): The input data.
    - num_samples (int): Number of bootstrap samples (default: 10000).
    - ci (float): The confidence level (default: 95).
    
    Returns:
    - (float, float): The lower and upper bounds of the confidence interval.
    """
    boot_means = np.random.choice(data, size=(num_samples, len(data)), replace=True).mean(axis=1)
    lower_bound = np.percentile(boot_means, (100 - ci) / 2)
    upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound

def get_full(projects, stat):
    try:
        stats = []
        for i, project in enumerate(projects):
            stats.append(project[stat].fetch_values()['value'].to_numpy()[:bounds[i]])
        stats = (np.concatenate(stats) == 1)
        stat_avg =  stats.mean()
        stats_low, stats_up = bootstrap_ci(stats)
    except:
        return np.nan, np.nan, np.nan
    
    return stat_avg, stats_up, stats_low

def get_stat_mean_std(projects, stat = False):
    try:
        stats = []
        for i, project in enumerate(projects):
            stats.append(project[stat].fetch_values()['value'].to_numpy()[:bounds[i]])
        stats = np.concatenate(stats)
        stat_avg = stats.mean()
        stats_low, stats_up = bootstrap_ci(stats)
    except:
        return np.nan, np.nan, np.nan
    
    return stat_avg, stats_up, stats_low

ids = [tuple([f'GRAD-{i}' for i in [1149, 1196, 1225, 1241]])]
for id in ids:
    print(id, isinstance(id, tuple), type(id))
    if isinstance(id, tuple):
        project = []
        for run_id in id:
            project.append(neptune.init_run(project="INSAIT/gradient-invesrion-attacks-gnn", mode="read-only", with_id=f"{run_id}", api_token=os.environ['NEPTUNE_API_KEY']))
    else:
        project = [neptune.init_run(project="INSAIT/gradient-invesrion-attacks-gnn", mode="read-only", with_id=f"{id}", api_token=os.environ['NEPTUNE_API_KEY'])]

    gsm0, gsm0_top, gsm0_bot = get_stat_mean_std(project, 'logs/deg_0_f')
    gsm1, gsm1_top, gsm1_bot  = get_stat_mean_std(project, 'logs/deg_1_r2')
    gsm2, gsm2_top, gsm2_bot  = get_stat_mean_std(project, 'logs/deg_2_r2')
    full, full_top, _ = get_full(project, 'logs/deg_1_r2')
    elapsed_time = 0
    for proj in project:
        elapsed_time += proj['sys/running_time'].fetch()

    print(f'{gsm0*100:.1f}^{{+{(gsm0_top-gsm0)*100:.1f}}}_{{-{(gsm0-gsm0_bot)*100:.1f}}} & {gsm1*100:.1f}^{{+{(gsm1_top-gsm1)*100:.1f}}}_{{-{(gsm1-gsm1_bot)*100:.1f}}} & {gsm2*100:.1f}^{{+{(gsm2_top-gsm2)*100:.1f}}}_{{-{(gsm2-gsm2_bot)*100:.1f}}} & {100*full}\pm{{{100*(full_top-full):.1f}}} & {elapsed_time/3600:.1f}')
    
    
