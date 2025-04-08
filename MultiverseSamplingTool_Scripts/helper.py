from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import hypergeom, spearmanr, pearsonr
from sklearn.neighbors import NearestNeighbors
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import pickle
from scipy.spatial.distance import cdist
from matplotlib import gridspec
import seaborn as sns

PROJECT_ROOT = Path.cwd()
data_path = PROJECT_ROOT
output_path = PROJECT_ROOT / 'Output'  ##Please create an output folder within the directory
rng = np.random.default_rng(11)
np.random.seed(11)

def objective_func_reg(TempModelNum, Y, bl_run, rf_run, tw_run, el_run, embed_features):
    TempData = embed_features[:, :, TempModelNum]
    TotalRegions = TempData.shape[1]
    TotalSubjects = TempData.shape[0]

    Temp_bl = bl_run[TempModelNum]
    Temp_rf = rf_run[TempModelNum]
    Temp_tw = tw_run[TempModelNum]
    Temp_el = el_run[TempModelNum]

    model = Pipeline([('scaler', StandardScaler()), ('linear', LinearRegression())])
    model.fit(TempData, Y.ravel())
    pred = model.predict(TempData)

    scores_pred = r2_score(Y.ravel(), pred)

    # Save the weights of each feature
    weights = model.named_steps['linear'].coef_

    return scores_pred, weights

def posterior(gp, x_obs, y_obs, z_obs, grid_X):
    xy = (np.array([x_obs.ravel(), y_obs.ravel()])).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(grid_X.reshape(-1, 2), return_std=True)
    return mu, std, gp

def posteriorOnlyModels(gp, x_obs, y_obs, z_obs, AllModelEmb):
    xy = (np.array([x_obs.ravel(), y_obs.ravel()])).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(AllModelEmb, return_std=True)
    return mu, std, gp

def initialize_bo(ModelEmbedding, kappa, n_burnin=10, n_bayesopt=26):
    RandomSeed = 118
    np.random.seed(RandomSeed)

    kernel = 1.0 * Matern(length_scale=25, length_scale_bounds=(10,80), nu=2.5) \
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 0.1))

    lb1 = np.min(ModelEmbedding[:, 0])
    hb1 = np.max(ModelEmbedding[:, 0])
    lb2 = np.min(ModelEmbedding[:, 1])
    hb2 = np.max(ModelEmbedding[:, 1])
    pbounds = {'b1': (lb1, hb1), 'b2': (lb2, hb2)}

    nbrs = NearestNeighbors(n_neighbors=89, algorithm='ball_tree').fit(ModelEmbedding)

    utility = UtilityFunction(kind="ucb", kappa=kappa, xi=1e-1)

    init_points = n_burnin
    n_iter = n_bayesopt

    optimizer = BayesianOptimization(f=None, pbounds=pbounds, verbose=4, random_state=RandomSeed)
    optimizer.set_gp_params(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
    return kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed

def run_bo(optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed, ModelEmbedding, model_config, Y, X, MultivariateUnivariate=True, verbose=True):
    selected_models = []
    temp_model_nums = []
    # Convert temp_model_nums to a set to get the unique values
    target_nums = []
    BadIters = np.empty(0)
    LastModel = -1
    Iter = 0
    pbar = tqdm(total=(init_points) + n_iter)
    while Iter < init_points + n_iter:
        np.random.seed(RandomSeed+Iter)

        if Iter < init_points:
            next_point_to_probe = {'b1': np.random.uniform(pbounds['b1'][0], pbounds['b1'][1]),
                                   'b2': np.random.uniform(pbounds['b2'][0], pbounds['b2'][1])}
            s1, s2 = next_point_to_probe.values()
        else:
            next_point_to_probe = optimizer.suggest(utility)
            s1, s2 = next_point_to_probe.values()

        Model_coord = np.array([[s1, s2]])
        distances, indices = nbrs.kneighbors(Model_coord)

        # I order to reduce repeatedly sampling the same point, check if
        # suggested point was sampled last and then check in ModelNums what the
        # name/index of that model is, if was recently sampled then take the
        # second nearest point.
         # Check if the suggested model has been sampled before
        #if np.asscalar(indices[0][0]) not in temp_model_nums:
        if indices[0][0].item() not in [temp_model_nums[i] for i in np.where(BadIters == 0)[0]]: #or if np.asscalar(indices[0][0]) not in [temp_model_nums[i] for i in np.where(BadIters == 0)[0]]:
            # If it hasn't been sampled before, proceed as before
            TempModelNum = indices[0][0].item() #or TempModelNum = np.asscalar(indices[0][0])
            ActualLocation = ModelEmbedding[TempModelNum]
            Distance = distances[0][0]
        else:
            # If it has, find the next nearest model that hasn't been sampled yet
            for i in range(1, len(indices[0])):
                TempModelNum = indices[0][i].item() #or TempModelNum = np.asscalar(indices[0][i])
                if TempModelNum not in temp_model_nums:
                    ActualLocation = ModelEmbedding[TempModelNum]
                    Distance = distances[0][i]
                    break

        # Add the index of the sampled model to temp_model_nums
        temp_model_nums.append(TempModelNum)
        
        if (Distance <10 or Iter<init_points): #THIS to change - change it back to 10
            # Hack: because space is continuous but analysis approaches aren't,
            # we penalize points that are far (>10 distance in model space)
            # from any actual analysis approaches by assigning them the value of
            # the worst performing approach in the burn-in
            LastModel = TempModelNum
            BadIters = np.append(BadIters,0)
            # Call the objective function and evaluate the model/pipeline

            if MultivariateUnivariate:
                (target, _) = objective_func_reg(TempModelNum, Y, model_config['bl_run'],
                                                model_config['rf_run'], model_config['tw_run'],
                                                model_config['el_run'], X)

                if verbose:
                    print("Next Iteration")
                    print(Iter)
                    # print("Model Num %d " % TempModelNum)
                    print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
                    print(Distance)
                    print("Target Function: %.4f" % (target))
                    print(' ')
                np.random.seed(Iter)
                # This is a hack. Add a very small random number to the coordinates so
                # that even if the model has been previously selected the GP thinks its
                # a different point, since this was causing it to crash
                TempLoc1 = ActualLocation[0] + (np.random.random_sample(1) - 0.5)/10
                TempLoc2 = ActualLocation[1] + (np.random.random_sample(1) - 0.5)/10
                pbar.update(1)
                target_nums.append(target)
                
        else:
            newlist = sorted(optimizer.res, key=lambda k: k['target'])
            target = newlist[0]['target']
            LastModel = -1
            #selected_models.append(TempModelNum)  # Add the selected model index to the list ADDED NOW

            if verbose:
                print("Next Iteration")
                print(Iter)
                # print("Model Num %d " % TempModelNum)
                # print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
                print(Distance)
                print("Target Function Default Bad: %.4f" % (target))
                print(' ')

            BadIters = np.append(BadIters,1)
            TempLoc1 = Model_coord[0][0]
            TempLoc2 = Model_coord[0][1]
            n_iter = n_iter+1
            

        #selected_models.append(TempModelNum)  # Add the selected model index to the list ADDED NOW
        Iter = Iter+1
        # Update the GP data with the new coordinates and model performance
        register_sample = {'b1': TempLoc1, 'b2': TempLoc2}
        optimizer.register(params=register_sample, target=target)
        

    pbar.close()
    return BadIters, temp_model_nums, target_nums

def plot_bo_estimated_space(kappa, BadIter, optimizer, pbounds, ModelEmbedding,
                        PredictedAcc, kernel, output_path):
    x = np.linspace(pbounds['b1'][0] - 10, pbounds['b1'][1] + 10, 500).reshape(
    -1, 1)
    y = np.linspace(pbounds['b2'][0] - 10, pbounds['b2'][1] + 10, 500).reshape(
        -1, 1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=10)

    x_temp = np.array([[res["params"]["b1"]] for res in optimizer.res])
    y_temp = np.array([[res["params"]["b2"]] for res in optimizer.res])
    z_temp = np.array([res["target"] for res in optimizer.res])

    x_obs=x_temp[BadIter==0]
    y_obs=y_temp[BadIter==0]
    z_obs=z_temp[BadIter==0]

    NumSamplesToInclude=x_obs.shape[0]
    x1x2 = np.array(list(product(x, y)))
    X0p, X1p = x1x2[:, 0].reshape(500, 500), x1x2[:, 1].reshape(500, 500)

    mu, sigma, gp = posterior(gp, x_obs[0:NumSamplesToInclude],
                              y_obs[0:NumSamplesToInclude],
                              z_obs[0:NumSamplesToInclude], x1x2)

    Zmu = np.reshape(mu, (500, 500))
    Zsigma = np.reshape(sigma, (500, 500))

    conf0 = np.array(mu - 2 * sigma).reshape(500, 500)
    conf1 = np.array(mu + 2 * sigma).reshape(500, 500)

    X0p, X1p = np.meshgrid(x, y, indexing='ij')

    font_dict_title = {'fontsize': 25}
    font_dict_label = {'fontsize': 15}
    font_dict_label3 = {'fontsize': 15}
    vmax = Zmu.max()
    vmin = Zmu.min()

    cm = ['coolwarm', 'seismic']
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,8))

    ax = ax1
    pcm = ax.pcolormesh(X0p, X1p, Zmu, vmax=vmax, vmin=vmin, cmap=cm[0],
                        rasterized=True)
    ax.set_xlim(-100, 80) # mds
    ax.set_ylim(-80, 100) # mds
    ax.set_aspect('equal', 'box')
    ax = ax2

    pcm = ax.scatter(ModelEmbedding[0:PredictedAcc.shape[0],0],
                         ModelEmbedding[0:PredictedAcc.shape[0],1],
                         c=PredictedAcc, vmax=vmax, vmin=vmin,
                         cmap=cm[0], rasterized=True)

    ax.set_aspect('equal', 'box')

    fig.tight_layout()

    ax.set_xlim(-100, 80) # mds
    ax.set_ylim(-80, 100) # mds
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.35, 0.02, 0.3])
    fig.colorbar(pcm, cax=cbar_ax)

    fig.savefig(str(output_path / f'BOptAndTrueK{kappa}.svg'))

    return x_obs, y_obs, z_obs, x, y, gp, vmax, vmin


def plot_bo_evolution(kappa, x_obs, y_obs, z_obs, x, y, gp, vmax, vmin,
                      ModelEmbedding, PredictedAcc, n_samples, output_path):
    fig, axs = plt.subplots(5, 3, figsize=(12,18))
   
    divided_value = n_samples // 5
    array = [divided_value * i for i in range(1, 6)]
    array[-1] = n_samples
    cm = ['coolwarm', 'seismic']

    for idx, NumSamplesToInclude in enumerate(array):

        x1x2 = np.array(list(product(x, y)))
        X0p, X1p = x1x2[:, 0].reshape(500, 500), x1x2[:, 1].reshape(500, 500)
        mu, sigma, gp = posterior(gp, x_obs[0:NumSamplesToInclude],
                                   y_obs[0:NumSamplesToInclude],
                                   z_obs[0:NumSamplesToInclude], x1x2)

        muModEmb, sigmaModEmb, gpModEmb = posteriorOnlyModels(gp,
                                                  x_obs[0:NumSamplesToInclude],
                                                  y_obs[0:NumSamplesToInclude],
                                                  z_obs[0:NumSamplesToInclude],
                                                  ModelEmbedding)
        Zmu = np.reshape(mu, (500, 500))
        Zsigma = np.reshape(sigma, (500, 500))

        conf0 = np.array(mu - 2 * sigma).reshape(500, 500)
        conf1 = np.array(mu + 2 * sigma).reshape(500, 500)

        X0p, X1p = np.meshgrid(x, y, indexing='ij')

        ax = axs[idx, 0]
        pcm = ax.pcolormesh(X0p, X1p, Zmu, vmax=vmax, vmin=vmin,
                cmap=cm[0],rasterized=True)
        ax.set_aspect('equal', 'box')

        ax.set_xlim(-100, 80) # mds
        ax.set_ylim(-80, 100) # mds
        ax = axs[idx,1]
        pcm = ax.pcolormesh(X0p, X1p, Zsigma,cmap=cm[1],rasterized=True)#,vmax=vmax,vmin=vmin)
        ax.set_title("Iterations: %i" % (NumSamplesToInclude), fontsize=15,
                     fontweight="bold")
        ax.set_aspect('equal', 'box')

        ax.set_xlim(-100, 80) # mds
        ax.set_ylim(-80, 100) # mds
        ax = axs[idx,2]

        ax.set_xlim(-2.55, -2.25)
        ax.set_ylim(-2.55, -2.25)

        pcm=ax.scatter(muModEmb[PredictedAcc!=PredictedAcc.min()],
                       PredictedAcc[PredictedAcc!=PredictedAcc.min()],
                       marker='.', c='gray')

        ax.set_xlim(PredictedAcc.max(), PredictedAcc.min())
        ax.set_ylim(PredictedAcc.max(), PredictedAcc.min())

        ax.set_xlim(PredictedAcc.min(), PredictedAcc.max())
        ax.set_ylim(PredictedAcc.min(), PredictedAcc.max())

        ax.set_aspect('equal', 'box')

    fig.savefig(str(output_path / f'BOptEvolutionK{kappa}.svg'))

    corr = spearmanr(muModEmb, PredictedAcc)
    return corr, muModEmb  # Returning the Spearman correlation


def create_unique_balanced_stratified_sample(df, sample_size):
    # Define categories
    categories = ["bl_run", "rf_run", "tw_run", "el_run"]

    # Initialize selected pipelines
    selected_pipelines = []

    # Iteratively add pipelines to the selected list for each category
    remaining_sample_size = sample_size
    while remaining_sample_size > 0 and len(selected_pipelines) < sample_size:
        len_before = len(selected_pipelines)
        for category in categories:
            # Dynamically create the grouping
            category_data = df.loc[~df.index.isin(selected_pipelines)].groupby(category)

            if len(category_data) > 0:
                # Determine the number of pipelines to sample
                sample_size_cat = min(remaining_sample_size, len(category_data))
                if sample_size_cat > 0:
                    sampled_pipelines = category_data.sample(n=sample_size_cat, replace=False).index.tolist()

                # Update selected pipelines and remaining sample size
                selected_pipelines.extend(sampled_pipelines[:remaining_sample_size])
                remaining_sample_size = sample_size - len(selected_pipelines)

        if len(selected_pipelines) == len_before:
            break

    return df.loc[df.index.isin(selected_pipelines)]

def create_stratified_samples(df, sample_sizes):
    stratified_samples = {}

    for sample_size in sample_sizes:
        stratified_sample = create_unique_balanced_stratified_sample(df, sample_size)
        stratified_samples[f"Sample_{sample_size}"] = stratified_sample

    return stratified_samples

def average_nearest_neighbor_distance(sample, reference):
    # Calculates the average nearest neighbor distance from sample to reference
    distances = cdist(sample, reference)
    nearest_neighbor_distances = np.mean(np.mean(distances, axis=1))  # Find nearest neighbor for each dot in sample
    return np.mean(nearest_neighbor_distances)


def spec_curve(sampling_method='AL', num_items_spec=None):
    if sampling_method == 'AL':
        Pipelines = pd.read_csv(str(output_path / "Pipelines.csv"))
        PredictedAcc_AL = pickle.load(open(str(output_path / f"PredictedAcc_AL_{num_items_spec}.p"), "rb"))
        data_spec_curve = Pipelines.copy()
        data_spec_curve['perf_pipelines'] = PredictedAcc_AL
        n_burnin = 10
        AL_df = pd.read_csv(str(output_path / f"ALPipelines_{num_items_spec}.csv"))
    elif sampling_method == 'Stra':
        stratified_df = pd.read_csv(str(output_path / f'StratifiedPipelines_{num_items_spec}.csv'))
        data_spec_curve = stratified_df.copy()
        data_spec_curve = data_spec_curve.drop(columns=['indices'])
    elif sampling_method == 'Rand':
        random_df = pd.read_csv(str(output_path / f'RandomPipelines_{num_items_spec}.csv'))
        data_spec_curve = random_df.copy()
        data_spec_curve = data_spec_curve.drop(columns=['indices'])

    # Sort the data by PredictedAcc_AL in descending order
    data_sorted = data_spec_curve.sort_values(by='perf_pipelines', ascending=False).reset_index(drop=True)

    # Extract the relevant columns
    predicted_acc = data_sorted['perf_pipelines']
    df_forks = data_sorted.drop(columns=['perf_pipelines'])

    # Create list of pipeline choices
    pipe_choices = []
    for pipe_idx in range(df_forks.shape[0]):
        pipe_choices.append(' '.join(df_forks.iloc[pipe_idx].apply(str)))

    # Create forking paths dictionary
    fork_dict = {}
    for column in df_forks.columns:
        fork_dict[column] = df_forks[column].apply(str).unique().tolist()

    # Create boolean list for each item within each forking path
    bool_list = {}
    for key, values in fork_dict.items():
        for value in values:
            bool_list[value] = np.array([True if value in choice else False for choice in pipe_choices])

    items = list(bool_list.keys())
    bool_values = list(bool_list.values())

    # Sort the pipeline accordingly to accuracy
    sort_idx = np.argsort(-predicted_acc)  # Sort in descending order
    acc_sort = predicted_acc.iloc[sort_idx]
    pipe_choices_sort = np.asarray(pipe_choices)[sort_idx]
    bool_values_sort = np.asarray(bool_values)[:, sort_idx]

    # Create a grid for the subplots with specific heights for the plots
# %%
def spec_curve(sampling_method='AL', num_items_spec=None):
    if sampling_method == 'AL':
        Pipelines = pd.read_csv(str(output_path / "Pipelines.csv"))
        PredictedAcc_AL = pickle.load(open(str(output_path / f"PredictedAcc_AL_{num_items_spec}.p"), "rb"))
        data_spec_curve = Pipelines.copy()
        data_spec_curve['perf_pipelines'] = PredictedAcc_AL
        n_burnin = 10
        AL_df = pd.read_csv(str(output_path / f"ALPipelines_{num_items_spec}.csv"))
    elif sampling_method == 'Stra':
        stratified_df = pd.read_csv(str(output_path / f'StratifiedPipelines_{num_items_spec}.csv'))
        data_spec_curve = stratified_df.copy()
        data_spec_curve = data_spec_curve.drop(columns=['indices'])
    elif sampling_method == 'Rand':
        random_df = pd.read_csv(str(output_path / f'RandomPipelines_{num_items_spec}.csv'))
        data_spec_curve = random_df.copy()
        data_spec_curve = data_spec_curve.drop(columns=['indices'])
    elif sampling_method == 'Full':
        full_df = pd.read_csv(str(output_path / f'Pipelines.csv'))
        full_performance = pickle.load(open(str(output_path / 'PredictedAcc.p'), 'rb'))
        full_performance_df = pd.DataFrame(full_performance, columns=['perf_pipelines'])
        data_spec_curve = pd.concat([full_df, full_performance_df], axis=1)

    # Sort the data by PredictedAcc_AL in descending order
    data_sorted = data_spec_curve.sort_values(by='perf_pipelines', ascending=False).reset_index(drop=True)

    # Extract the relevant columns
    predicted_acc = data_sorted['perf_pipelines']
    df_forks = data_sorted.drop(columns=['perf_pipelines'])

    # Create list of pipeline choices
    pipe_choices = []
    for pipe_idx in range(df_forks.shape[0]):
        pipe_choices.append(' '.join(df_forks.iloc[pipe_idx].apply(str)))

    # Create forking paths dictionary
    fork_dict = {}
    for column in df_forks.columns:
        fork_dict[column] = df_forks[column].apply(str).unique().tolist()

    # Create boolean list for each item within each forking path
    bool_list = {}
    for key, values in fork_dict.items():
        for value in values:
            bool_list[value] = np.array([True if value in choice else False for choice in pipe_choices])

    items = list(bool_list.keys())
    bool_values = list(bool_list.values())

    # Sort the pipeline accordingly to accuracy
    sort_idx = np.argsort(-predicted_acc)  # Sort in descending order
    acc_sort = predicted_acc.iloc[sort_idx]
    pipe_choices_sort = np.asarray(pipe_choices)[sort_idx]
    bool_values_sort = np.asarray(bool_values)[:, sort_idx]

    # Create a grid for the subplots with specific heights for the plots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])  # 2 rows, 1 column, height ratio of 1:2

    # Plot the line plot in the upper subplot
    ax0 = plt.subplot(gs[0])
    if sampling_method == 'AL':
        AL_indices = AL_df['indices'].values.astype(int)[n_burnin:]  # Exclude the pipelines in the burn-in phase
        colors_dots = ['darkgreen' if i in AL_indices else 'lightgreen' for i in range(len(acc_sort))]
        ax0.scatter(range(len(acc_sort)), acc_sort, c=colors_dots, s=10, alpha=0.6)
    else:
        ax0.plot(acc_sort, marker='o', linestyle='None', markersize=2, color="black")
    ax0.axhline(y=0, color='blue', linestyle='dashed')
    ax0.set_ylabel('R-square', fontsize=14)
    ax0.set_title('Specification curve analysis', fontsize=16)
    ax0.set_xticks(np.arange(0, len(acc_sort), 100))  # Set specific x-ticks
    ax0.set_xticklabels(np.arange(0, len(acc_sort), 100), fontsize=12)  # Set specific x-tick labels
    ax0.set_xlim(0, len(acc_sort))  # Set the x-axis limits to match the number of data points
    ax0.grid(True, alpha=0.4)

    # Create a heatmap-friendly DataFrame
    heatmap_data = np.zeros((len(items), len(pipe_choices_sort)))

    for i, choice in enumerate(pipe_choices_sort):
        for j, item in enumerate(items):
            if item in choice:
                heatmap_data[j, i] = 1

    # Plot the heatmap in the lower subplot
    ax1 = plt.subplot(gs[1])

    # Initialize the heatmap with a white background
    sns.heatmap(np.ones_like(heatmap_data), cmap=['white'], cbar=False, xticklabels=False, yticklabels=items, ax=ax1, linewidths=0.5, linecolor='white')

    # Apply the specific colors to each decision point
    colors = {
        'bl_run': '#00008B',  # Dark Blue
        'rf_run': '#DAA520',  # Goldenrod (Dark Yellow)
        'tw_run': '#006400',  # Dark Green
        'el_run': '#800080'   # Dark Purple
    }

    # Create a dictionary to map decisions to colors
    decision_colors = {}
    for col in df_forks.columns:
        unique_decisions = df_forks[col].apply(str).unique().tolist()
        color = colors[col]
        for decision in unique_decisions:
            decision_colors[decision] = color

    # Color the heatmap cells accordingly
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if heatmap_data[i, j] == 1:
                ax1.scatter(j + 0.5, i + 0.5, color=decision_colors[items[i]], s=25, edgecolor='none', zorder=2)

    ax1.set_facecolor('white')  # Set the background color to white
    ax1.set_xlim(0, len(acc_sort))  # Set the x-axis limits to match the upper plot
    ax1.set_ylabel('Decision node in the pipeline', fontsize=14)
    ax1.set_yticklabels(items, fontsize=12)
    ax1.grid(True, alpha=0.5, zorder=3)  # Enable grid with 50% transparency and higher zorder

    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(str(output_path / f'spec_curve_{sampling_method}_{num_items_spec}samples.svg'))
    # plt.show()