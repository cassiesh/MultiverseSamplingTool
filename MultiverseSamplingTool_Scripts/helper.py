# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import pickle
from itertools import product
from pathlib import Path

# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Progress Bar
from tqdm import tqdm


# =============================================================================
# Global Variables & Configuration
# =============================================================================
PROJECT_ROOT = Path.cwd()
# Note: Ensure an 'Output' folder exists in your project directory.
OUTPUT_PATH = PROJECT_ROOT / 'Output'
RNG = np.random.default_rng(11)
np.random.seed(11)


# =============================================================================
# Core Analysis Functions
# =============================================================================

def objective_func_reg_IVF(model_index, y_data, model_config, features):
    """
    Evaluates a single pipeline's performance using linear regression.

    Args:
        model_index (int): The index of the model/pipeline to evaluate.
        y_data (np.ndarray): The target variable data.
        model_config (dict): A dictionary containing the parameter runs for each
                             pipeline component (bl_run, rf_run, etc.).
        features (np.ndarray): The feature data for all models.

    Returns:
        tuple: A tuple containing:
            - float: The R-squared score of the model.
            - np.ndarray: The VIF for each feature in the model.
    """
    # Select the feature data for the specified model
    temp_data = features[:, :, model_index]

    # Define and fit the model pipeline
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(temp_data)
    X_with_const = add_constant(scaled_data)
    
    # Calculate VIF for each feature. The first VIF (for the constant) is ignored.
    weights = [variance_inflation_factor(X_with_const, i) for i in range(1, X_with_const.shape[1])]
    
    # --- R-squared Calculation ---
    # We can now use the scaled data to fit the model.
    model = LinearRegression()
    model.fit(scaled_data, y_data.ravel())
    
    predictions = model.predict(scaled_data)
    score = r2_score(y_data.ravel(), predictions)

    return score, weights

def objective_func_reg(model_index, y_data, model_config, features):
    """
    Evaluates a single pipeline's performance using linear regression.

    Args:
        model_index (int): The index of the model/pipeline to evaluate.
        y_data (np.ndarray): The target variable data.
        model_config (dict): A dictionary containing the parameter runs for each
                             pipeline component (bl_run, rf_run, etc.).
        features (np.ndarray): The feature data for all models.

    Returns:
        tuple: A tuple containing:
            - float: The R-squared score of the model.
            - np.ndarray: The feature weights from the linear regression model.
    """
    # Select the feature data for the specified model
    temp_data = features[:, :, model_index]

    # Define and fit the model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])
    model.fit(temp_data, y_data.ravel())

    # Make predictions and calculate R^2 score
    predictions = model.predict(temp_data)
    score = r2_score(y_data.ravel(), predictions)

    # Extract feature weights
    weights = model.named_steps['linear'].coef_

    return score, weights

def posterior(gp, x_obs, y_obs, z_obs, grid_X):
    """
    Calculates the posterior distribution of a Gaussian Process model.

    Args:
        gp (GaussianProcessRegressor): The Gaussian Process regressor instance.
        x_obs (np.ndarray): Observed x-coordinates.
        y_obs (np.ndarray): Observed y-coordinates.
        z_obs (np.ndarray): Observed target values.
        grid_X (np.ndarray): The grid of points to predict on.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The mean of the posterior distribution.
            - np.ndarray: The standard deviation of the posterior distribution.
            - GaussianProcessRegressor: The fitted GP model.
    """
    xy = np.array([x_obs.ravel(), y_obs.ravel()]).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(grid_X.reshape(-1, 2), return_std=True)
    return mu, std, gp

def posterior_only_models(gp, x_obs, y_obs, z_obs, all_model_emb):
    """
    Calculates the posterior distribution only at the locations of actual models.

    Args:
        gp (GaussianProcessRegressor): The Gaussian Process regressor instance.
        x_obs (np.ndarray): Observed x-coordinates.
        y_obs (np.ndarray): Observed y-coordinates.
        z_obs (np.ndarray): Observed target values.
        all_model_emb (np.ndarray): The embeddings for all models.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The mean of the posterior distribution for each model.
            - np.ndarray: The standard deviation for each model.
            - GaussianProcessRegressor: The fitted GP model.
    """
    xy = np.array([x_obs.ravel(), y_obs.ravel()]).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(all_model_emb, return_std=True)
    return mu, std, gp


# =============================================================================
# Bayesian Optimization Functions
# =============================================================================

def initialize_bo(model_embedding, kappa, n_burnin=10, n_bayesopt=26):
    """
    Initializes the Bayesian Optimization components.

    Args:
        model_embedding (np.ndarray): The 2D embedding of the model space.
        kappa (float): The exploration-exploitation trade-off parameter for UCB.
        n_burnin (int): Number of random exploration steps.
        n_bayesopt (int): Number of optimization steps.

    Returns:
        tuple: A tuple containing all initialized BO components.
    """
    random_seed = 118
    np.random.seed(random_seed)

    kernel = 1.0 * Matern(length_scale=25, length_scale_bounds=(10, 80), nu=2.5) \
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 0.1))

    # Define parameter bounds from the model embedding space
    pbounds = {
        'b1': (np.min(model_embedding[:, 0]), np.max(model_embedding[:, 0])),
        'b2': (np.min(model_embedding[:, 1]), np.max(model_embedding[:, 1]))
    }

    # Nearest neighbors for finding models close to suggested points
    nbrs = NearestNeighbors(n_neighbors=89, algorithm='ball_tree').fit(model_embedding)
    utility = UtilityFunction(kind="ucb", kappa=kappa, xi=1e-1)
    optimizer = BayesianOptimization(f=None, pbounds=pbounds, verbose=4, random_state=random_seed)
    optimizer.set_gp_params(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

    return kernel, optimizer, utility, n_burnin, n_bayesopt, pbounds, nbrs, random_seed

def run_bo(optimizer, utility, init_points, n_iter, pbounds, nbrs, random_seed, model_embedding, model_config, y_data, x_data, verbose=True):
    """
    Runs the Bayesian Optimization loop.

    Args:
        optimizer (BayesianOptimization): The BO optimizer instance.
        utility (UtilityFunction): The acquisition function.
        init_points (int): Number of random exploration steps.
        n_iter (int): Number of optimization steps.
        pbounds (dict): The parameter bounds for the optimization.
        nbrs (NearestNeighbors): Fitted nearest neighbors model.
        random_seed (int): The random seed for reproducibility.
        model_embedding (np.ndarray): The 2D embedding of the model space.
        model_config (dict): Configuration for the pipelines.
        y_data (np.ndarray): The target variable data.
        x_data (np.ndarray): The feature data.
        verbose (bool): Whether to print progress information.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array indicating if an iteration was "bad" (far from a real model).
            - list: A list of the indices of the models selected at each iteration.
            - list: A list of the target performance values for each selected model.
    """
    temp_model_nums = []
    target_nums = []
    bad_iters = np.empty(0)
    total_iterations = init_points + n_iter
    
    with tqdm(total=total_iterations, desc="Bayesian Optimization") as pbar:
        iteration = 0
        while iteration < total_iterations:
            np.random.seed(random_seed + iteration)

            # Suggest a new point to probe
            if iteration < init_points:
                # Random exploration phase (burn-in)
                next_point = {
                    'b1': np.random.uniform(pbounds['b1'][0], pbounds['b1'][1]),
                    'b2': np.random.uniform(pbounds['b2'][0], pbounds['b2'][1])
                }
            else:
                # Guided exploration/exploitation phase
                next_point = optimizer.suggest(utility)
            
            s1, s2 = next_point.values()
            model_coord = np.array([[s1, s2]])
            distances, indices = nbrs.kneighbors(model_coord)

            # Find the nearest valid model that hasn't been sampled recently
            temp_model_num = -1
            actual_location = None
            distance = -1
            
            valid_indices = [i for i, is_bad in enumerate(bad_iters) if is_bad == 0]
            recently_sampled = [temp_model_nums[i] for i in valid_indices]

            for i in range(len(indices[0])):
                candidate_model_num = indices[0][i].item()
                if candidate_model_num not in recently_sampled:
                    temp_model_num = candidate_model_num
                    actual_location = model_embedding[temp_model_num]
                    distance = distances[0][i]
                    break
            
            # If all neighbors have been sampled, default to the closest one
            if temp_model_num == -1:
                temp_model_num = indices[0][0].item()
                actual_location = model_embedding[temp_model_num]
                distance = distances[0][0]

            temp_model_nums.append(temp_model_num)
            
            # Penalize points far from any actual model
            if distance < 10 or iteration < init_points:
                bad_iters = np.append(bad_iters, 0)
                (target, _) = objective_func_reg(
                    temp_model_num, y_data, model_config, x_data
                )
                target_nums.append(target)
                
                # Add small noise to coordinates to prevent GP from crashing on duplicate points
                temp_loc1 = actual_location[0] + (np.random.random_sample(1) - 0.5) / 10
                temp_loc2 = actual_location[1] + (np.random.random_sample(1) - 0.5) / 10
                pbar.update(1)
            else:
                # Assign the worst-performing score to points far away
                bad_iters = np.append(bad_iters, 1)
                target = sorted(optimizer.res, key=lambda k: k['target'])[0]['target']
                temp_loc1, temp_loc2 = model_coord[0]
                total_iterations += 1 # This iteration doesn't count as a real evaluation

            if verbose:
                print(f"\nIteration: {iteration}")
                print(f"Suggested Model Index: {temp_model_num}, Distance: {distance:.4f}")
                print(f"Target Value: {target:.4f}")

            # Register the sample with the optimizer
            optimizer.register(params={'b1': temp_loc1, 'b2': temp_loc2}, target=target)
            iteration += 1

    return bad_iters, temp_model_nums, target_nums


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_bo_estimated_space(kappa, bad_iter, optimizer, pbounds, model_embedding, predicted_acc, kernel):
    """
    Plots the true performance space vs. the BO estimated performance space.
    """
    x = np.linspace(pbounds['b1'][0] - 10, pbounds['b1'][1] + 10, 500).reshape(-1, 1)
    y = np.linspace(pbounds['b2'][0] - 10, pbounds['b2'][1] + 10, 500).reshape(-1, 1)
    
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

    x_temp = np.array([[res["params"]["b1"]] for res in optimizer.res])
    y_temp = np.array([[res["params"]["b2"]] for res in optimizer.res])
    z_temp = np.array([res["target"] for res in optimizer.res])

    x_obs = x_temp[bad_iter == 0]
    y_obs = y_temp[bad_iter == 0]
    z_obs = z_temp[bad_iter == 0]

    x1x2 = np.array(list(product(x, y)))
    mu, _, gp = posterior(gp, x_obs, y_obs, z_obs, x1x2)
    Zmu = np.reshape(mu, (500, 500))
    
    X0p, X1p = np.meshgrid(x, y, indexing='ij')
    vmax, vmin = Zmu.max(), Zmu.min()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot Estimated Performance
    ax1.set_title("BO Estimated Performance", fontsize=16)
    pcm1 = ax1.pcolormesh(X0p, X1p, Zmu, vmax=vmax, vmin=vmin, cmap='coolwarm', rasterized=True)
    ax1.set_xlim(-100, 80)
    ax1.set_ylim(-80, 100)
    ax1.set_aspect('equal', 'box')

    # Plot True Performance
    ax2.set_title("True Performance", fontsize=16)
    pcm2 = ax2.scatter(model_embedding[:, 0], model_embedding[:, 1], c=predicted_acc, vmax=vmax, vmin=vmin, cmap='coolwarm', rasterized=True)
    ax2.set_xlim(-100, 80)
    ax2.set_ylim(-80, 100)
    ax2.set_aspect('equal', 'box')

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.35, 0.02, 0.3])
    fig.colorbar(pcm1, cax=cbar_ax, label='R-squared')
    
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(OUTPUT_PATH / f'BOpt_vs_True_k{kappa}.svg')
    
    return x_obs, y_obs, z_obs, x, y, gp, vmax, vmin

def plot_bo_evolution(kappa, x_obs, y_obs, z_obs, x, y, gp, vmax, vmin, model_embedding, predicted_acc, n_samples):
    """
    Plots the evolution of the BO model's predictions over iterations.
    """
    fig, axs = plt.subplots(5, 3, figsize=(12, 18))
    sample_points = np.linspace(n_samples // 5, n_samples, 5, dtype=int)

    for idx, num_samples in enumerate(sample_points):
        x1x2 = np.array(list(product(x, y)))
        mu, sigma, _ = posterior(gp, x_obs[:num_samples], y_obs[:num_samples], z_obs[:num_samples], x1x2)
        mu_mod_emb, _, _ = posterior_only_models(gp, x_obs[:num_samples], y_obs[:num_samples], z_obs[:num_samples], model_embedding)
        
        Zmu = np.reshape(mu, (500, 500))
        Zsigma = np.reshape(sigma, (500, 500))
        X0p, X1p = np.meshgrid(x, y, indexing='ij')

        # Plot Mean Prediction
        ax = axs[idx, 0]
        ax.pcolormesh(X0p, X1p, Zmu, vmax=vmax, vmin=vmin, cmap='coolwarm', rasterized=True)
        ax.set_ylabel(f"Iter: {num_samples}", fontsize=14, fontweight='bold')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-100, 80)
        ax.set_ylim(-80, 100)

        # Plot Uncertainty
        ax = axs[idx, 1]
        ax.pcolormesh(X0p, X1p, Zsigma, cmap='seismic', rasterized=True)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-100, 80)
        ax.set_ylim(-80, 100)

        # Plot Correlation
        ax = axs[idx, 2]
        valid_mask = predicted_acc != predicted_acc.min()
        ax.scatter(mu_mod_emb[valid_mask], predicted_acc[valid_mask], marker='.', c='gray')
        ax.set_xlim(predicted_acc.min(), predicted_acc.max())
        ax.set_ylim(predicted_acc.min(), predicted_acc.max())
        ax.set_aspect('equal', 'box')

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / f'BOpt_Evolution_k{kappa}.svg')

    corr, _ = spearmanr(mu_mod_emb, predicted_acc)
    return corr, mu_mod_emb

# =============================================================================
# Sampling and Utility Functions
# =============================================================================

def average_nearest_neighbor_distance(sample, reference):
    """
    Calculates the average distance from each point in a sample to its nearest 
    neighbor in a reference set, excluding itself.
    """
    distances = cdist(sample, reference)
    # Sort distances for each sample point and take the second smallest,
    # as the smallest will be 0 (the distance to itself).
    sorted_distances = np.sort(distances, axis=1)
    min_distances_to_others = sorted_distances[:, 1]
    return np.mean(min_distances_to_others)
