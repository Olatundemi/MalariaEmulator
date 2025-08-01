import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D


#Function to plot simulated data (Exploratory Data Analysis)
def plot_subplots_for_runs(df, num_runs=20):
    # Convert time column to years
    df = df.copy()
    df['t_years'] = df['t'] / 365  # Convert days to years
    
    # Select the first num_runs unique runs
    unique_runs = df['run'].unique()[:num_runs]
    columns_to_plot = ["prev_true", "incall", "EIR_true"]
    
    # Define distinct colors for the three columns
    color_map = {
        'prev_true': 'tab:blue',
        'incall': 'tab:orange',
        'EIR_true': 'tab:green'
    }
    
    # Define grid size for subplots (4 subplots per row)
    rows = (num_runs - 1) // 4 + 1
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()  # Flatten for easy iteration
    
    for i, run in enumerate(unique_runs):
        ax = axes[i]
        # Extract and sort data for the run by time
        run_data = df[df['run'] == run].sort_values('t_years')
        
        # Separate pre-observation (annual averages) and monthly observation data
        pre_obs = run_data[run_data['t_years'] < 10].sort_values('t_years')  # 10 years
        monthly = run_data[run_data['t_years'] >= 10].sort_values('t_years')
        
        # Shade the pre-observation region with a soft color
        ax.axvspan(run_data['t_years'].min(), 10, facecolor='lightblue', alpha=0.2,
                   label='Pre-observation period' if i == 0 else None)
        
        # Plot monthly observation data as lines for each column
        for col in columns_to_plot:
            # Plot the monthly data as a solid line
            ax.plot(monthly['t_years'], monthly[col],
                    color=color_map[col],
                    lw=2,
                    label=f'{col} (Monthly)' if i == 0 else None)
            
            # Overlay the pre-observation points with diamond markers.
            if not pre_obs.empty:
                ax.scatter(pre_obs['t_years'], pre_obs[col],
                           color=color_map[col],
                           marker='D', s=100,
                           edgecolor='k',
                           zorder=5,
                           label='Pre-observation (Annual)' if i == 0 else None)
            
            # Connect all pre-observation points with dashed lines
            if len(pre_obs) > 1:
                pre_obs_times = pre_obs['t_years'].values
                pre_obs_vals = pre_obs[col].values
                for j in range(1, len(pre_obs_times)):
                    ax.plot([pre_obs_times[j-1], pre_obs_times[j]],
                            [pre_obs_vals[j-1], pre_obs_vals[j]],
                            color=color_map[col],
                            linestyle='--', lw=1.5, zorder=4)
            
            # Connect last pre-observation point to first monthly point if both exist
            if (not pre_obs.empty) and (not monthly.empty):
                last_pre = pre_obs.iloc[-1]
                first_month = monthly.iloc[0]
                ax.plot([last_pre['t_years'], first_month['t_years']],
                        [last_pre[col], first_month[col]],
                        color=color_map[col],
                        linestyle='--', lw=1.5, zorder=4)
        
        # Draw a vertical dashed line to clearly mark the boundary at t=10 years
        ax.axvline(x=10, color='grey', linestyle='--', lw=1)
        
        # Set the y-axis to a logarithmic scale for better comparison
        ax.set_yscale('log')
        
        # Set subplot title and labels
        ax.set_title(f'Run {run}')
        ax.set_xlabel('Time (years)')  # Updated label
        ax.set_ylabel('Values')
        ax.grid(True, which="both", ls="--", linewidth=0.5)
    
    # Hide any unused subplots
    for j in range(len(unique_runs), len(axes)):
        axes[j].axis('off')
    
    # Create a custom legend.
    custom_lines = [
        Line2D([0], [0], color=color_map['prev_true'], lw=2),
        Line2D([0], [0], color=color_map['incall'], lw=2),
        Line2D([0], [0], color=color_map['EIR_true'], lw=2),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='grey', markeredgecolor='k', markersize=10),
        Line2D([0], [0], color='grey', linestyle='--', lw=1.5)
    ]
    custom_labels = [
        'ANC_prevalence (Monthly)',
        'ANC_incidence (Monthly)',
        'EIR (Monthly)',
        'Pre-observation (Annual)',
        'Ages_2-10'
    ]
    
    # Place the universal legend at the top center
    fig.legend(custom_lines, custom_labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to provide space for the legend
    plt.savefig('plots/data_exploration/ANC_Params_Plot_Unobserved_Observed_Phase.png', dpi=300, bbox_inches='tight')
    plt.show()


#Function for plotting/checking cross cross correlation between inputs and targets
def plot_cross_correlation(df, input_col, target_cols, lags=range(-50, 50), single_target=None): 
    """
    Computes and plots cross-correlation between an input feature and one or multiple target features.
    
    Parameters:
    df : pandas.DataFrame
        The DataFrame containing input and target features.
    input_col : str
        Column name of the input feature.
    target_cols : list
        List of column names for target features.
    lags : range, optional
        Range of lag values for correlation computation (default is -50 to 50).
    single_target : str, optional
        If specified, plots only the cross-correlation with this single target feature.
    """
   
    input_feature = df[input_col]
    targets = {col: df[col] for col in target_cols}
    
    results = {}
    for name, target in targets.items():
        correlations = []
        for lag in lags:
            shifted_input = input_feature.shift(-lag) if lag < 0 else input_feature.shift(lag)
            corr = target.corr(shifted_input)
            correlations.append(corr)
        results[name] = correlations
    
    # What to plot
    plot_targets = {single_target: results[single_target]} if single_target else results
    
    # Plotting
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("deep", len(plot_targets))
    
    for (name, correlations), color in zip(plot_targets.items(), colors):
        plt.plot(lags, correlations, marker='o', label=f'Cross-Correlation with {name}', color=color, markersize=5)
    
    plt.axvline(0, color='red', linestyle='--', linewidth=1, label='Current Time (t)')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    plt.xlabel('Lag (Timesteps)', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.title('Cross-Correlation between Input Feature and Target(s)', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    #plt.savefig('plots/data_exploration/Cross-Correlation_plots_between_prevalence_incidence_and_EIR.png')
    plt.show()


#Function for plotting model losses
def plot_training_metrics(results):
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result['loss_history'], label=f"{result['name']} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss Over Epochs: ANC Data at Equilibrium")
    plt.legend()
    plt.savefig("plots/model_performance/training_loss_25000runs_at_ Equilibrium.png")
    plt.show()
    

#Function for comparing model parameters and training timelines 
def plot_model_comparison(results):
    model_names = [result['name'] for result in results]
    parameters = [sum(p.numel() for p in result['model'].parameters()) for result in results]
    durations = [result['duration'] for result in results]

    max_params = max(parameters)
    max_durations = max(durations) if max(durations) > 0 else 1

    normalized_params = [p / max_params for p in parameters]
    normalized_durations = [d / max_durations for d in durations]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(model_names))
    bar_width = 0.4

    bar1 = ax.bar(x - bar_width / 2, normalized_params, bar_width, label="Normalized Parameters", color="skyblue")
    bar2 = ax.bar(x + bar_width / 2, normalized_durations, bar_width, label="Normalized Training Duration (sec)", color="darkorange")

    for bar, param in zip(bar1, parameters):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{param}', ha='center', va='bottom', fontsize=12, color="blue")
    for bar, duration in zip(bar2, durations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{duration:.1f}', ha='center', va='bottom', fontsize=12, color="orange")

    ax.set_xlabel("Model")
    ax.set_ylabel("Normalized Values")
    ax.set_title("Comparison of Model Parameters and Training Durations: ANC at Equilibrium")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend()

    plt.savefig("plots/model_performance/model_comparison_25000runs_at_Equilibrium.png")
    plt.show()


def plot_subplots_for_run(df, num_runs=2):
    # Convert time column to years
    df = df.copy()
    df['t_years'] = df['t'] / 365  # Convert days to years

    # Randomly sample 2 unique runs
    unique_runs = np.random.choice(df['run'].unique(), size=num_runs, replace=False)
    columns_to_plot = ["prev_true", "incall", "EIR_true"]

    # Define colors
    color_map = {
        'prev_true': 'tab:blue',
        'incall': 'tab:orange',
        'EIR_true': 'tab:green'
    }

    # Set up 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Adjust width to reduce padding

    for i, (run, ax) in enumerate(zip(unique_runs, axes)):
        run_data = df[(df['run'] == run) & (df['t_years'] >= 0)]
        run_data = run_data.sort_values('t_years')

        for col in columns_to_plot:
            ax.plot(run_data['t_years'], run_data[col],
                    color=color_map[col],
                    lw=2,
                    label=col if i == 0 else None)

        ax.set_yscale('log')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Values')
        ax.grid(True, which="both", ls="--", linewidth=0.5)

    # Create a shared legend above both plots
    custom_lines = [
        Line2D([0], [0], color=color_map['prev_true'], lw=2),
        Line2D([0], [0], color=color_map['incall'], lw=2),
        Line2D([0], [0], color=color_map['EIR_true'], lw=2),
    ]
    custom_labels = [
        'ANC_prevalence (Monthly)',
        'ANC_incidence (Monthly)',
        'EIR (Monthly)',
    ]

    fig.legend(custom_lines, custom_labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for legend
    plt.savefig('plots/data_exploration/ANC_Params_TwoRuns_Observed_Only.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_normalized_eir_with_prevalence(df, num_runs=15):
    df = df.copy()
    df['t_years'] = df['t'] / 365  # Convert days to years

    unique_runs = df['run'].unique()[:num_runs]
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, run in enumerate(unique_runs):
        ax = axes[i]
        run_data = df[df['run'] == run].sort_values('t_years')

        # Normalize EIR
        eir_max = run_data['EIR_true'].max() * 1.1
        run_data['EIR_scaled'] = run_data['EIR_true'] / eir_max if eir_max != 0 else 0

        # Plot prevalence as line (primary y-axis)
        ax.plot(run_data['t_years'], run_data['prev_true'],
                color='tab:blue', lw=2, label='Prevalence')

        # Plot scaled EIR as a solid line (secondary y-axis)
        ax2 = ax.twinx()
        ax2.plot(run_data['t_years'], run_data['EIR_scaled'],
                 color='tab:orange', lw=2, label='EIR (scaled)')
        ax2.set_ylim(0, 1)
        ax2.set_yticks(np.linspace(0, 1, 5))
        ax2.set_yticklabels([f"{int(round(v))}" for v in np.linspace(0, eir_max, 5)])
        ax2.tick_params(axis='y', labelsize=8)

        # Set prevalence axis
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelsize=8)

        if i >= cols * (rows - 1):
            ax.set_xlabel('Time (years)', fontsize=9)
        else:
            ax.set_xticklabels([])

        ax.set_title(f'Run {run}', fontsize=10)
        ax.grid(False)

    # Turn off unused subplots
    for j in range(len(unique_runs), len(axes)):
        axes[j].axis('off')

    # Shared legend
    custom_lines = [
        Line2D([0], [0], color='tab:blue', lw=2),
        Line2D([0], [0], color='tab:orange', lw=2)
    ]
    fig.legend(custom_lines, ['Prevalence', 'EIR'],
               loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.04))

    # Universal y-axis labels
    fig.text(-0.02, 0.5, 'Prevalence', va='center', rotation='vertical', fontsize=20)
    fig.text(1, 0.5, 'EIR', va='center', rotation='vertical', fontsize=20)

    plt.subplots_adjust(top=0.91, right=0.91)
    plt.savefig('plots/data_exploration/Final_EIR_Prevalence_LineStyle_Clean.png',
                dpi=300, bbox_inches='tight')
    plt.show()



def plot_normalized_eir_with_prev(df, num_runs=2):
    df = df.copy()
    df['t_years'] = df['t'] / 365  # Convert days to years

    # Randomly select 2 unique runs
    unique_runs = np.random.choice(df['run'].unique(), size=num_runs, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for i, (run, ax) in enumerate(zip(unique_runs, axes)):
        run_data = df[df['run'] == run].sort_values('t_years')

        # Normalize EIR
        eir_max = run_data['EIR_true'].max() * 1.1
        run_data['EIR_scaled'] = run_data['EIR_true'] / eir_max if eir_max != 0 else 0

        # Plot prevalence (primary y-axis)
        ax.plot(run_data['t_years'], run_data['prev_true'],
                color='tab:blue', lw=2, label='Prevalence')

        # Plot incidence on same primary y-axis
        ax.plot(run_data['t_years'], run_data['incall'],
                color='tab:green', lw=2, label='Incidence')

        ax.set_ylim(0, 1)
        ax.set_ylabel('Prevalence / Incidence', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

        # Plot scaled EIR (secondary y-axis)
        ax2 = ax.twinx()
        ax2.plot(run_data['t_years'], run_data['EIR_scaled'],
                 color='tab:orange', lw=2, label='EIR (scaled)')
        ax2.set_ylim(0, 1)
        ax2.set_yticks(np.linspace(0, 1, 5))
        ax2.set_yticklabels([f"{int(round(v))}" for v in np.linspace(0, eir_max, 5)])
        ax2.set_ylabel('EIR', fontsize=10)
        ax2.tick_params(axis='y', labelsize=8)

        ax.set_xlabel('Time (years)', fontsize=9)
        #ax.set_title(f'Run {run}', fontsize=10)
        ax.grid(False)

    # Shared legend
    custom_lines = [
        Line2D([0], [0], color='tab:blue', lw=2),
        Line2D([0], [0], color='tab:green', lw=2),
        Line2D([0], [0], color='tab:orange', lw=2)
    ]
    fig.legend(custom_lines, ['ANC Prevalence', 'Clinical Incidence', 'EIR (scaled)'],
               loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.08))

    plt.savefig('plots/data_exploration/EIR_Prev_Inc_2Runs_SameScale.png',
                dpi=300, bbox_inches='tight')
    plt.show()
