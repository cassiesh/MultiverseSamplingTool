# =============================================================================
# 1. Libraries and Setup
# =============================================================================
import numpy as np
import pandas as pd
import pickle
import warnings
import openpyxl
from pathlib import Path
from scipy import io
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor

# Import helper functions from your uploaded script
from helper import (
    OUTPUT_PATH,
    objective_func_reg,
    objective_func_reg_IVF,
    initialize_bo,
    run_bo,
    average_nearest_neighbor_distance,
)

warnings.filterwarnings("ignore")
np.random.seed()

# =============================================================================
# 2. Data Loading and Preparation
# =============================================================================

def load_and_prepare_data(data_path):
    """
    Loads all necessary data files and prepares them for analysis.

    Args:
        data_path (Path): The path to the data directory.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The main feature data.
            - np.ndarray: The 2D embedding of the feature space.
            - np.ndarray: The target scores (e.g., extraversion) for all participants.
    """
    print("Loading and preparing data...")
    features_sp = io.loadmat(data_path / 'mats.mat')['mats']
    embed_features = io.loadmat(data_path / 'tSNEembeddingFeatures5.mat')['tSNEembedding']
    features_sp_dict = io.loadmat(data_path / 'LPP_sep.mat')['LPP_sep']
    features_sp_df = pd.DataFrame(features_sp_dict, columns=['subject', 'emotion', 'path', 'LPP'])
    subject_ls_str = features_sp_df['subject'].apply(lambda x: str(x)).unique()
    subject_ls = [item.strip("['']").strip() for item in subject_ls_str]
    subject_df = pd.DataFrame(np.array(subject_ls, dtype=int), columns=['subj'])
    scores = pd.read_csv(data_path / 'Extraversion.dat', sep="\t", skiprows=1,
                         names=['subj', 'Extrav', 'NEOE_W', 'NEOE_G', 'NEOE_A', 'NEOE_AC', 'NEOE_ES', 'NEOE_PE'])
    extrav_sc = pd.merge(subject_df, scores, on='subj')
    y_scores = extrav_sc['Extrav'].values
    print("Data loading complete.")
    return features_sp, embed_features, y_scores

def create_pipeline_configurations():
    """
    Creates a DataFrame of all possible pipeline configurations from predefined components.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame where each row is a unique pipeline configuration.
            - dict: A dictionary version of the pipeline configurations for easy lookup.
    """
    print("Creating pipeline configurations...")
    bl = ['b-200', 'b-100']
    rf = ['rAvg', 'rMas', 'rCSD']
    tw = ['500200', '500300', '600200', '600300', '600600', '700200', '700300', '700600', '450100', 'GAV400', 'SAV400']
    el = ['CP1CP2PzP3P4', 'P3P4CP1CP2', 'P3PzP4', 'FzCzPz', 'CP1CP2', 'Cz', 'Pz', 'around']
    config = {}
    count = 0
    for bl_temp in bl:
        for rf_temp in rf:
            for tw_temp in tw:
                for el_temp in el:
                    config.setdefault('bl_run', {})[count] = bl_temp
                    config.setdefault('rf_run', {})[count] = rf_temp
                    config.setdefault('tw_run', {})[count] = tw_temp
                    config.setdefault('el_run', {})[count] = el_temp
                    count += 1
    pipelines_df = pd.DataFrame(config)
    pipelines_df.to_csv(OUTPUT_PATH / "Pipelines.csv", index=False)
    print(f"{len(pipelines_df)} pipeline configurations created.")
    return pipelines_df, config

def export_all_data_to_excel(num_items, data_dict):
    """
    Exports all performance data from both datasets to a single Excel file with 8 sheets.

    Args:
        num_items (int): The sample size, used for file naming.
        data_dict (dict): A dictionary containing the 8 DataFrames to export.
    """
    output_filename = OUTPUT_PATH / f'All_Results_Combined_{num_items}.xlsx'
    print(f"\nExporting all data to a single Excel file: {output_filename}")

    with pd.ExcelWriter(output_filename) as writer:
        # Prediction Sheets
        data_dict['All_Prediction'].to_excel(writer, sheet_name='All_Pipelines_Prediction', index=False)
        data_dict['Stra_Prediction'].to_excel(writer, sheet_name='Stratified_Prediction', index=False)
        data_dict['Rand_Prediction'].to_excel(writer, sheet_name='Random_Prediction', index=False)
        data_dict['AL_Prediction'].to_excel(writer, sheet_name='AL_Prediction', index=False)
        
        # Lockbox Sheets
        data_dict['All_Lockbox'].to_excel(writer, sheet_name='All_Pipelines_Lockbox', index=False)
        data_dict['Stra_Lockbox'].to_excel(writer, sheet_name='Stratified_Lockbox', index=False)
        data_dict['Rand_Lockbox'].to_excel(writer, sheet_name='Random_Lockbox', index=False)
        data_dict['AL_Lockbox'].to_excel(writer, sheet_name='AL_Lockbox', index=False)
    
    print("Export complete.")


# =============================================================================
# 3. Analysis Functions
# =============================================================================

def run_exhaustive_analysis(dataset_name, feature_data, y_data, model_config):
    """
    Runs the analysis for every pipeline to establish a ground truth for a given dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "Prediction", "Lockbox") for file naming.
        feature_data (np.ndarray): The feature data for the dataset.
        y_data (np.ndarray): The target scores for the dataset.
        model_config (dict): The dictionary of all pipeline configurations.

    Returns:
        np.ndarray: An array of performance scores (R-squared) for every pipeline.
    """
    print(f"Starting exhaustive analysis for '{dataset_name}' dataset...")
    num_pipelines = len(model_config['bl_run'])
    predicted_acc = np.zeros(num_pipelines)
    for i in range(num_pipelines):
        temp_pred_acc, _ = objective_func_reg(i, y_data, model_config, feature_data)
        predicted_acc[i] = temp_pred_acc
    filepath = OUTPUT_PATH / f"PredictedAcc_Full_{dataset_name}.p"
    with open(filepath, "wb") as f: pickle.dump(predicted_acc, f)
    print(f"Exhaustive analysis for '{dataset_name}' complete. Results saved.")
    return predicted_acc

def evaluate_pipelines(pipeline_indices, feature_data, y_data, model_config):
    """
    Evaluates a pre-selected list of pipelines on a given dataset.

    Args:
        pipeline_indices (list or np.ndarray): Indices of the pipelines to evaluate.
        feature_data (np.ndarray): The feature data for the dataset.
        y_data (np.ndarray): The target scores for the dataset.
        model_config (dict): The dictionary of all pipeline configurations.

    Returns:
        np.ndarray: An array of performance scores for the evaluated pipelines.
    """
    performance = np.zeros(len(pipeline_indices))
    for idx, pipeline_idx in enumerate(pipeline_indices):
        perf, _ = objective_func_reg(pipeline_idx, y_data, model_config, feature_data)
        performance[idx] = perf
    return performance

def create_unique_balanced_stratified_sample(df, sample_size):
    """
    Creates a single balanced, stratified sample without replacement.
    """
    categories = ["bl_run", "rf_run", "tw_run", "el_run"]
    selected_indices = []
    remaining_size = sample_size

    while remaining_size > 0 and len(selected_indices) < sample_size:
        len_before = len(selected_indices)
        for category in categories:
            available_df = df.loc[~df.index.isin(selected_indices)]
            if available_df.empty: continue
            
            category_groups = available_df.groupby(category)
            if not category_groups: continue
            
            # Sample one from each group if possible
            sampled = category_groups.apply(lambda x: x.sample(1)).index.get_level_values(1)
            
            # Add unique indices up to the remaining sample size
            new_indices = [idx for idx in sampled if idx not in selected_indices]
            num_to_add = min(remaining_size, len(new_indices))
            selected_indices.extend(new_indices[:num_to_add])
            remaining_size -= num_to_add
            if remaining_size <= 0: break
        
        # Break if no new pipelines are added in a full loop
        if len(selected_indices) == len_before:
            break
            
    # If still not enough, fill randomly from remaining
    if len(selected_indices) < sample_size:
        remaining_indices = df.index[~df.index.isin(selected_indices)]
        fill_count = sample_size - len(selected_indices)
        if len(remaining_indices) >= fill_count:
            selected_indices.extend(np.random.choice(remaining_indices, fill_count, replace=False))

    return df.loc[selected_indices]

def create_stratified_samples(df, sample_sizes):
    """
    Creates multiple stratified samples of different sizes.
    """
    return {
        f"Sample_{size}": create_unique_balanced_stratified_sample(df, size)
        for size in sample_sizes
    }

def identify_best_pipelines(dataset_name, num_items, pipelines_df, full_perf, al_df, rand_df, stra_df):
    """
    Identifies and saves the top N performing pipelines for each sampling method.

    Args:
        dataset_name (str): The name of the dataset being analyzed (e.g., "Prediction", "Lockbox"),
                            used for naming the output file.
        num_items (int): The number of samples used in the sampling methods, for file naming.
        pipelines_df (pd.DataFrame): The master DataFrame containing all possible pipeline
                                     configurations (specs).
        full_perf (np.ndarray): An array of performance scores for the full multiverse.
        al_df (pd.DataFrame): A DataFrame containing the specs and performance of pipelines
                              selected by Active Learning. Must include a 'perf_pipelines' column.
        rand_df (pd.DataFrame): A DataFrame for pipelines from Random sampling, including 'perf_pipelines'.
        stra_df (pd.DataFrame): A DataFrame for pipelines from Stratified sampling, including 'perf_pipelines'.
    """
    print(f"Identifying best pipelines for '{dataset_name}'...")
    N = 10
    
    # Helper function to format the output strings
    def get_best_string(df):
        sorted_df = df.sort_values(by='perf_pipelines', ascending=False).head(N)
        # Separate specs from performance
        specs = sorted_df.drop(columns=['perf_pipelines'])
        perf = sorted_df['perf_pipelines']
        # Create string
        best_strings = specs.apply(lambda row: ', '.join(row.astype(str)), axis=1)
        best_strings = best_strings + ', ' + perf.astype(str)
        return best_strings.values

    # For Full Sample
    full_indices_sorted = np.argsort(full_perf)[-N:][::-1]
    full_specs_sorted = pipelines_df.iloc[full_indices_sorted]
    full_perf_sorted = full_perf[full_indices_sorted]
    full_sample_best_series = full_specs_sorted.apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    full_sample_best = (full_sample_best_series + ', ' + pd.Series(full_perf_sorted, index=full_sample_best_series.index).astype(str)).values

    best_pipeline = pd.DataFrame({
        'Best Full Sample': full_sample_best,
        'Best Active Learning': get_best_string(al_df),
        'Best Random Sampling': get_best_string(rand_df),
        'Best Stratified Sampling': get_best_string(stra_df)
    })
    best_pipeline.to_csv(OUTPUT_PATH / f'bestPipelines_{dataset_name}_{num_items}.csv', index=False)


# =============================================================================
# 4. Plotting Functions
# =============================================================================
def plot_raincloud(dataset_name, num_items, full_perf, al_estimated_perf, rand_df, stra_df, al_burnin_indices, al_intelligent_indices):
    """
    Generates and saves a raincloud plot with special coloring for Active Learning.

    Args:
        dataset_name (str): The name of the dataset for the plot title and filename.
        num_items (int): The number of samples for the plot title.
        full_perf (np.ndarray): An array of performance scores for the full multiverse.
        al_estimated_perf (np.ndarray): The GP's estimated performance for ALL pipelines. This is
                                        used for the Active Learning violin and box plot.
        rand_df (pd.DataFrame): DataFrame for Random sampling, must contain a 'perf_pipelines' column.
        stra_df (pd.DataFrame): DataFrame for Stratified sampling, must contain a 'perf_pipelines' column.
        al_burnin_indices (list): A list of pipeline indices that were sampled by Active Learning
                                  during the initial random "burn-in" phase.
        al_intelligent_indices (list): A list of pipeline indices that were sampled by Active
                                       Learning during the intelligent optimization phase.
    """
    print(f"Generating raincloud plot for '{dataset_name}'...")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue','green', 'orange', 'red']
    
    data_to_plot = [full_perf, al_estimated_perf, stra_df['perf_pipelines'], rand_df['perf_pipelines']]
    bp = ax.boxplot(data_to_plot, patch_artist=True, vert=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    
    vp = ax.violinplot(data_to_plot, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False)
    for idx, b in enumerate(vp['bodies']):
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx + 1, idx + 2)
        b.set_color(colors[idx])
        
    for idx, data in enumerate(data_to_plot):
        y = np.full(len(data), idx + .8)
        y = y.astype(float)
        y += np.random.uniform(low=-.05, high=.05, size=len(y))
        if idx == 1: # Active Learning
            colors_dots = ['lightgreen'] * len(data)
            for i in al_burnin_indices: colors_dots[i] = 'lightgreen'
            for i in al_intelligent_indices: colors_dots[i] = 'darkgreen'
            ax.scatter(data, y, s=10, c=colors_dots, alpha=0.6)
        else:
            ax.scatter(data, y, s=10, c=colors[idx])

    plt.yticks(np.arange(1, 5, 1), ['Full Sample', 'Active learning', 'Stratified Sampling', 'Random Sampling'])
    plt.xlabel('R-square')
    plt.title(f"R-square values of the sampled pipelines")
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    plt.savefig(OUTPUT_PATH / f'raincloud_plot_{dataset_name}_{num_items}samples.svg')
    plt.close(fig)

def plot_sampled_pipelines_in_space(dataset_name, num_items, embed_features, al_indices, rand_indices, stra_indices):
    """
    Plots where the sampled pipelines fall in the 2D embedding space.

    Args:
        dataset_name (str): The name of the dataset for the plot title and filename.
        num_items (int): The number of samples for the plot title.
        embed_features (np.ndarray): A 2D array of coordinates for ALL pipelines in the multiverse.
        al_indices (np.ndarray): An array of indices for pipelines selected by Active Learning.
        rand_indices (np.ndarray): An array of indices for pipelines selected by Random sampling.
        stra_indices (np.ndarray): An array of indices for pipelines selected by Stratified sampling.
    """
    print(f"Plotting sampled pipelines in space for '{dataset_name}'...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    axs[0].scatter(embed_features[:, 0], embed_features[:, 1], color='blue', label='Full Multiverse', alpha=0.1)
    axs[0].scatter(embed_features[rand_indices, 0], embed_features[rand_indices, 1], color='red', label='Random Sampling')
    axs[0].set_title('Random Sampling'); axs[0].legend(); axs[0].grid(True)

    axs[1].scatter(embed_features[:, 0], embed_features[:, 1], color='blue', label='Full Multiverse', alpha=0.1)
    axs[1].scatter(embed_features[stra_indices, 0], embed_features[stra_indices, 1], color='orange', label='Stratified Sampling')
    axs[1].set_title('Stratified Sampling'); axs[1].legend(); axs[1].grid(True)

    axs[2].scatter(embed_features[:, 0], embed_features[:, 1], color='blue', label='Full Multiverse', alpha=0.1)
    axs[2].scatter(embed_features[al_indices, 0], embed_features[al_indices, 1], color='green', label='Active Learning')
    axs[2].set_title('Active Learning'); axs[2].legend(); axs[2].grid(True)

    fig.text(0.5, 0.04, 'Dimension 1', ha='center', va='center', fontsize=14)
    fig.text(0.04, 0.5, 'Dimension 2', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_PATH / f'samplePipelines_{dataset_name}_{num_items}samples.svg')
    plt.close(fig)

def spec_curve(spec_df, dataset_name, sampling_method, num_items, all_sampled_indices=None, burnin_indices=None):
    """
    Generates and saves a specification curve plot. This version ensures correct alignment
    between the performance plot and the decision node heatmap.

    Args:
        spec_df (pd.DataFrame): A DataFrame containing pipeline specifications and a 'perf_pipelines' column.
                                For AL, this should contain all pipelines with estimated performance.
                                For others, it contains only the sampled pipelines.
        dataset_name (str): The name of the dataset for the plot title and filename.
        sampling_method (str): The name of the sampling method ("AL", "Rand", "Stra").
        num_items (int): The number of samples for the plot title.
        all_sampled_indices (list, optional): All indices sampled by AL (burn-in + intelligent).
                                              Required if sampling_method is 'AL'. Defaults to None.
        burnin_indices (list, optional): Indices sampled by AL during the burn-in phase.
                                         Required if sampling_method is 'AL'. Defaults to None.
    """
    print(f"Generating spec curve for {dataset_name} - {sampling_method}...")

    # 1. Sort the data ONCE by performance. This is the master order for the x-axis.
    data_sorted = spec_df.sort_values(by='perf_pipelines', ascending=False).reset_index()
    
    # Extract sorted performance for the top plot
    acc_sort = data_sorted['perf_pipelines']

    # Extract the specification choices, also sorted
    df_forks = data_sorted.drop(columns=['perf_pipelines', 'index']) # Drop original index if it exists
    if 'indices' in df_forks.columns: df_forks = df_forks.drop(columns=['indices'], errors='ignore')

    # 2. Create the list of all possible decision options for the y-axis, grouped by category.
    decision_categories = ['bl_run', 'rf_run', 'tw_run', 'el_run']
    items = []
    for cat in decision_categories:
        # Sort the unique values within each category for consistent plotting
        unique_vals = sorted(spec_df[cat].unique().astype(str))
        items.extend(unique_vals)
    
    # Create a mapping from each decision item to its y-position on the plot
    item_y_map = {item: i for i, item in enumerate(items)}
    
    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    
    # --- Top Plot (Performance Curve) ---
    ax0 = plt.subplot(gs[0])
    
    if sampling_method == 'AL':
        sorted_original_indices = data_sorted['index'].values
        intelligent_indices = list(set(all_sampled_indices) - set(burnin_indices))
        colors_dots = ['darkgreen' if idx in intelligent_indices else 'darkgreen' if idx in burnin_indices else 'lightgreen' for idx in sorted_original_indices]
        ax0.scatter(range(len(acc_sort)), acc_sort, c=colors_dots, s=10, alpha=0.5)
    else:
        ax0.plot(range(len(acc_sort)), acc_sort, marker='o', linestyle='None', markersize=2, color="black")

    ax0.axhline(y=0, color='blue', linestyle='dashed')
    ax0.set_ylabel('R-square', fontsize=14)
    ax0.set_title(f'Specification curve analysis', fontsize=16)
    ax0.grid(True, alpha=0.4)
    ax0.set_xlim(-1, len(acc_sort))

    # --- Bottom Plot (Decision Heatmap) ---
    ax1 = plt.subplot(gs[1])
    
    # Define colors for each category
    category_colors = {'bl_run': '#00008B', 'rf_run': '#DAA520', 'tw_run': '#006400', 'el_run': '#800080'}

    # 3. Iterate through each pipeline (x-axis position) and plot its decisions
    for i, (idx, pipeline) in enumerate(df_forks.iterrows()):
        # For each pipeline, iterate through its decision choices
        for col_name, decision in pipeline.items():
            if col_name in category_colors:
                # Find the y-position for this decision
                y_pos = item_y_map[str(decision)]
                # Get the color for this category
                color = category_colors[col_name]
                # Plot the single dot
                ax1.scatter(i, y_pos, color=color, s=25, edgecolor='none')

    ax1.set_yticks(range(len(items)))
    ax1.set_yticklabels(items, fontsize=12)
    ax1.set_ylabel('Decision Node in Pipeline', fontsize=14)
    ax1.set_xlim(-1, len(acc_sort))
    ax1.set_ylim(-1, len(items))
    ax1.grid(True, alpha=0.5)
    # Invert y-axis to have bl_run at the bottom
    ax1.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f'spec_curve_{dataset_name}_{sampling_method}_{num_items}.svg')
    plt.close(fig)

# =============================================================================
# 5. Main Functions
# =============================================================================

def run_all_analyses(Pipelines, embed_features, model_config, data_partitions, sample_sizes):
    """
    Performs all computationally expensive analyses and saves the results to disk.
    This includes exhaustive analysis, sampling, and performance evaluation on both
    the Prediction and Lockbox datasets. The results are saved in a .pkl file
    for each sample size, which can be loaded later for plotting.

    Args:
        Pipelines (pd.DataFrame): DataFrame with all possible pipeline configurations.
        embed_features (np.ndarray): The 2D embedding of the feature space.
        model_config (dict): Dictionary of pipeline configurations.
        data_partitions (dict): Dictionary containing the 'Prediction' and 'Lockbox' data splits.
        sample_sizes (list): A list of integers representing the different sample sizes to test.
    """
    print("\n" + "="*25 + " RUNNING ALL ANALYSES " + "="*25)
    
    # Unpack data partitions
    FeaturePrediction, YPrediction = data_partitions['Prediction']
    FeatureLockbox, YLockbox = data_partitions['Lockbox']

    # --- Exhaustive Analysis (Ground Truth) ---
    PredictedAcc_Full_Prediction = run_exhaustive_analysis("Prediction", FeaturePrediction, YPrediction, model_config)
    PredictedAcc_Full_Lockbox = run_exhaustive_analysis("Lockbox", FeatureLockbox, YLockbox, model_config)

    for num_items in sample_sizes:
        print(f"\n--- Running analysis for sample size: {num_items} ---")

        # --- Step 1: Select pipelines based on the 'Prediction' dataset ---
        n_burnin, n_bayesopt = 20, num_items
        kernel, optimizer, utility, _, _, pbounds, nbrs, seed = initialize_bo(embed_features, 10, n_burnin, n_bayesopt)
        bad_iter, sel_indices, _ = run_bo(optimizer, utility, n_burnin, n_bayesopt, pbounds, nbrs, seed, embed_features, model_config, YPrediction, FeaturePrediction, verbose=False)
        
        good_indices_all = [idx for i, idx in enumerate(sel_indices) if bad_iter[i] == 0]
        al_burnin_indices = [idx for i, idx in enumerate(sel_indices) if bad_iter[i] == 0 and i < n_burnin]
        al_intelligent_indices = [idx for i, idx in enumerate(sel_indices) if bad_iter[i] == 0 and i >= n_burnin]
        
        random_indices = np.random.choice(range(len(Pipelines)), num_items, replace=False)
        stratified_samples = create_stratified_samples(Pipelines, [num_items])
        stratified_indices = stratified_samples[f'Sample_{num_items}'].index.values

        # --- Step 2: Evaluate performance and get AL estimations for both datasets ---
        
        # Performance on Prediction set
        al_estimated_pred, _ = optimizer._gp.predict(embed_features, return_std=True)
        al_sampled_pred = evaluate_pipelines(good_indices_all, FeaturePrediction, YPrediction, model_config)
        rand_pred = evaluate_pipelines(random_indices, FeaturePrediction, YPrediction, model_config)
        stra_pred = evaluate_pipelines(stratified_indices, FeaturePrediction, YPrediction, model_config)
        
        # Performance on Lockbox set
        al_sampled_lockbox = evaluate_pipelines(good_indices_all, FeatureLockbox, YLockbox, model_config)
        rand_lockbox = evaluate_pipelines(random_indices, FeatureLockbox, YLockbox, model_config)
        stra_lockbox = evaluate_pipelines(stratified_indices, FeatureLockbox, YLockbox, model_config)

        # CORRECTED LOGIC: Re-train GP on Lockbox performance to get a true Lockbox estimation
        print("Re-training Gaussian Process model on Lockbox data...")
        gp_lockbox = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
        gp_lockbox.fit(embed_features[good_indices_all], al_sampled_lockbox)
        al_estimated_lockbox, _ = gp_lockbox.predict(embed_features, return_std=True)

        # --- Step 3: Save all results and indices to disk ---
        results_to_save = {
            'indices': {
                'al': good_indices_all,
                'al_burnin': al_burnin_indices,
                'al_intelligent': al_intelligent_indices,
                'rand': random_indices,
                'stra': stratified_indices
            },
            'performance': {
                'Prediction': {
                    'Full': PredictedAcc_Full_Prediction, 'AL_estimated': al_estimated_pred,
                    'AL_sampled': al_sampled_pred, 'Rand': rand_pred, 'Stra': stra_pred
                },
                'Lockbox': {
                    'Full': PredictedAcc_Full_Lockbox, 'AL_estimated': al_estimated_lockbox,
                    'AL_sampled': al_sampled_lockbox, 'Rand': rand_lockbox, 'Stra': stra_lockbox
                }
            }
        }
        
        with open(OUTPUT_PATH / f'analysis_results_{num_items}.pkl', 'wb') as f:
            pickle.dump(results_to_save, f)
        print(f"All analysis results for sample size {num_items} saved.")


def generate_all_visuals(Pipelines, embed_features, sample_sizes):
    """
    Loads pre-computed analysis results from .pkl files and generates all plots 
    and statistical comparisons for each dataset and sample size.

    Args:
        Pipelines (pd.DataFrame): DataFrame with all possible pipeline configurations.
        embed_features (np.ndarray): The 2D embedding of the feature space.
        sample_sizes (list): A list of integers representing the different sample sizes to generate visuals for.
    """
    print("\n" + "="*20 + " GENERATING ALL VISUALS & COMPARISONS " + "="*20)

    for num_items in sample_sizes:
        print(f"\n--- Generating visuals for sample size: {num_items} ---")
        
        # Load the saved results
        try:
            with open(OUTPUT_PATH / f'analysis_results_{num_items}.pkl', 'rb') as f:
                results = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find analysis_results_{num_items}.pkl. Please run the analysis first.")
            continue

        indices = results['indices']
        performance = results['performance']
        
        data_for_export = {}

        for dataset_name in ['Prediction', 'Lockbox']:
            print(f"\n--- Processing visuals for '{dataset_name}' dataset ---")
            
            # Unpack performance data for the current dataset
            full_perf = performance[dataset_name]['Full']
            al_estimated_perf = performance[dataset_name]['AL_estimated']
            al_sampled_perf = performance[dataset_name]['AL_sampled']
            rand_perf = performance[dataset_name]['Rand']
            stra_perf = performance[dataset_name]['Stra']

            # --- Create full DataFrames for functions that need them ---
            al_df = Pipelines.iloc[indices['al']].copy()
            al_df['perf_pipelines'] = al_sampled_perf
            
            rand_df = Pipelines.iloc[indices['rand']].copy()
            rand_df['perf_pipelines'] = rand_perf

            stra_df = Pipelines.iloc[indices['stra']].copy()
            stra_df['perf_pipelines'] = stra_perf
            
            full_df = Pipelines.copy()
            full_df['perf_pipelines'] = full_perf

            al_spec_df = Pipelines.copy()
            al_spec_df['perf_pipelines'] = al_estimated_perf
            
            # --- Store data for final export ---
            data_for_export[f'All_{dataset_name}'] = full_df
            data_for_export[f'AL_{dataset_name}'] = al_spec_df
            data_for_export[f'Rand_{dataset_name}'] = rand_df
            data_for_export[f'Stra_{dataset_name}'] = stra_df

            # --- KS Statistics ---
            al_combined_perf = al_estimated_perf.copy()
            al_combined_perf[indices['al']] = al_sampled_perf

            ks_results = {
                'Full': full_perf,
                'Active Learning': al_combined_perf,
                'Random': rand_perf,
                'Stratified': stra_perf
            }
            ks_stats = {name: ks_2samp(full_perf, data) for name, data in ks_results.items()}
            medians_df = pd.DataFrame({
                'Sampling Method': list(ks_results.keys()),
                'Median': [pd.Series(data).median() for data in ks_results.values()],
                'KS Statistic': [s.statistic for s in ks_stats.values()],
                'KS p-value': [s.pvalue for s in ks_stats.values()]
            })
            medians_df.to_csv(OUTPUT_PATH / f'medianAndKS_{dataset_name}_{num_items}.csv', index=False)
            print(f"\nMedian and KS Statistics for '{dataset_name}':\n{medians_df}")

            # --- Identifying best pipeline ---
            identify_best_pipelines(dataset_name, num_items, Pipelines, full_perf, al_df, rand_df, stra_df)

            # --- Plot raincloud plot ---
            plot_raincloud(dataset_name, num_items, full_perf, al_estimated_perf, rand_df, stra_df, indices['al_burnin'], indices['al_intelligent'])

            # --- Plot the sampled pipelines in space ---
            plot_sampled_pipelines_in_space(dataset_name, num_items, embed_features, indices['al'], indices['rand'], indices['stra'])

            # --- Measuring similarity ---
            dist_al = average_nearest_neighbor_distance(embed_features[indices['al']], embed_features)
            dist_rand = average_nearest_neighbor_distance(embed_features[indices['rand']], embed_features)
            dist_stra = average_nearest_neighbor_distance(embed_features[indices['stra']], embed_features)
            dist_df = pd.DataFrame({'Method': ['Active Learning', 'Random', 'Stratified'], 'Avg Nearest Neighbor Dist': [dist_al, dist_rand, dist_stra]})
            dist_df.to_csv(OUTPUT_PATH / f'distances_{dataset_name}_{num_items}.csv', index=False)
            
            # --- Spec Curve Plots ---
            spec_curve(al_spec_df, dataset_name, "AL", num_items, all_sampled_indices=indices['al'], burnin_indices=indices['al_burnin'])
            spec_curve(rand_df, dataset_name, "Rand", num_items)
            spec_curve(stra_df, dataset_name, "Stra", num_items)
        
        # --- Export all data to a single Excel file ---
        export_all_data_to_excel(num_items, data_for_export)


# =============================================================================
# 6. Execution
# =============================================================================

# --- Set Up ---
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / 'Data'
OUTPUT_PATH.mkdir(exist_ok=True)

# --- Flags to control execution ---
RUN_ANALYSIS = True
RUN_PLOTTING = True
RUN_VIF_ANALYSIS = True #Optional: Set to True if you want to run the VIF analysis

# --- Load and Prepare Data ---
features_sp, embed_features, Y_all = load_and_prepare_data(DATA_PATH)
Pipelines, model_config = create_pipeline_configurations()

n_spdefine, n_prediction, n_lockbox = 20, 50, 28
remaining_indices = np.arange(n_spdefine, len(Y_all))
rng = np.random.default_rng(15)
rng.shuffle(remaining_indices)

data_partitions = {
    'Prediction': (features_sp[remaining_indices[:n_prediction], :, :], Y_all[remaining_indices[:n_prediction]]),
    'Lockbox': (features_sp[remaining_indices[n_prediction:], :, :], Y_all[remaining_indices[n_prediction:]])
}
print(f"Data partitioned: Prediction set={len(data_partitions['Prediction'][1])}, Lockbox set={len(data_partitions['Lockbox'][1])}")

# --- Run Analyses ---
sample_sizes = [53]

if RUN_ANALYSIS:
    run_all_analyses(Pipelines, embed_features, model_config, data_partitions, sample_sizes)

if RUN_PLOTTING:
    generate_all_visuals(Pipelines, embed_features, sample_sizes)

# --- Optional: Run VIF Analysis ---
if RUN_VIF_ANALYSIS:
    print("\n" + "="*25 + " RUNNING VIF ANALYSIS " + "="*25)
    
    # We will run this on the Prediction dataset
    FeaturePrediction, YPrediction = data_partitions['Prediction']
    
    all_r2 = []
    all_vifs = []
    
    # Loop through all pipelines and calculate R2 and VIF
    for i in range(len(Pipelines)):
        r2, vifs = objective_func_reg_IVF(i, YPrediction, model_config, FeaturePrediction)
        all_r2.append(r2)
        all_vifs.append(vifs)
        
    # Create the results DataFrame
    vif_df = pd.DataFrame(all_vifs, columns=[f'VIF_Feature_{j+1}' for j in range(len(all_vifs[0]))])
    r2_df = pd.DataFrame(all_r2, columns=['R_Squared'])
    
    # Combine pipeline specs with the results
    final_results_df = pd.concat([Pipelines, r2_df, vif_df], axis=1)
    
    # Save the DataFrame to a CSV file
    output_csv_path = OUTPUT_PATH / 'R2_and_VIF_results.csv'
    final_results_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved R-squared and VIF results to: {output_csv_path}")
    
    # Plot the distribution of VIF scores
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=vif_df)
    plt.title('Distribution of Variance Inflation Factor (VIF) for each Feature', fontsize=16)
    plt.ylabel('VIF Value')
    plt.xlabel('Feature')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    output_plot_path = OUTPUT_PATH / 'VIF_distribution_plot.svg'
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved VIF distribution plot to: {output_plot_path}")


print("\nWorkflow complete.")

