################
# 1. Libraries #
################
import scipy
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.colorbar
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import ks_2samp
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")
np.random.seed(15)


#####################
# 1. Preparing data #
#####################
### Setting the paths
PROJECT_ROOT = Path.cwd()
data_path = PROJECT_ROOT / 'Data'
output_path = PROJECT_ROOT / 'Output'  ##Please create an output folder within the directory
rng = np.random.default_rng(15)
np.random.seed(15)

### Import data
features_sp_dict = io.loadmat(str(data_path / 'LPP_sep.mat')) #dictionary
features_sp_dict = features_sp_dict['LPP_sep']
features_sp_df = pd.DataFrame(features_sp_dict, columns = ['subject','emotion','path','LPP'])
emotion_ls = features_sp_df['emotion'].apply(lambda x: str(x)).unique()
subject_ls = features_sp_df['subject'].apply(lambda x: str(x)).unique()
subject_ls = [item.strip("['']").strip() for item in subject_ls]
path_ls = features_sp_df['path'].apply(lambda x: str(x)).unique()
features_sp = io.loadmat(str(data_path / 'mats.mat')) #dictionary
features_sp = features_sp['mats'] #array [98, 6, 528]
embed_features = io.loadmat(str(data_path / 'tSNEembeddingFeatures5.mat')) #dictionary
embed_features = embed_features['tSNEembedding'] #array [528, 2]
n_ind = len(features_sp)
n_paths = np.size(features_sp, 2)
scores = pd.read_csv(str(data_path / 'Extraversion.dat'), #should only contain 1 column of Extrav
                sep = "\t",
                skiprows = 1,
                names = ['subj', 'Extrav', 'NEOE_W', 'NEOE_G', 'NEOE_A', 'NEOE_AC', 'NEOE_ES', 'NEOE_PE'])
subject_ls = pd.DataFrame(np.array(subject_ls, dtype = int), columns=['subj'])
extrav_sc = pd.merge(subject_ls, scores, on='subj')
Y = extrav_sc['Extrav']

### Setting the forking paths
bl = ['b-200', 'b-100']
rf = ['rAvg', 'rMas', 'rCSD']
tw = ['500200', '500300', '600200', '600300', '600600', '700200', '700300',
              '700600', '450100', 'GAV400', 'SAV400']
el = ['CP1CP2PzP3P4', 'P3P4CP1CP2', 'P3PzP4', 'FzCzPz', 'CP1CP2', 'Cz',
             'Pz', 'around']
bl_run = {}
rf_run = {}
tw_run = {}
el_run = {}
n_bl = len(bl)
n_rf = len(rf)
n_tw = len(tw)
n_el = len(el)
count = 0
for bl_id, bl_temp in enumerate(bl):
    for rf_id, rf_temp in enumerate(rf):
        for tw_id, tw_temp in enumerate(tw):
            for el_id, el_temp in enumerate(el):
                bl_run[count] = bl_temp
                rf_run[count] = rf_temp
                tw_run[count] = tw_temp
                el_run[count] = el_temp
                count += 1
Pipelines = pd.DataFrame({
    'bl_run': bl_run,
    'rf_run': rf_run,
    'tw_run': tw_run,
    'el_run': el_run
})
Pipelines.to_csv(str(output_path / "Pipelines.csv"), index=False)

### Partitioning the data
SpDefine = 20 # The first 20 participants are used to calculate the similarity between pipelines 
#LockBox = 0
#RandomIndexes = rng.choice(n_ind, size=n_ind, replace=False)

#FeatureModelSpace = features_sp[RandomIndexes[0:SpDefine], :, :]
#FeatureLockBoxData = features_sp[RandomIndexes[SpDefine:LockBox], :, :]
FeaturePrediction = features_sp[SpDefine:, :, :]

#YModelSpace = Y[RandomIndexes[0:SpDefine]]
#YLockBoxData = Y[RandomIndexes[SpDefine:LockBox]]
YPrediction = Y[SpDefine:]


##########################
# 3. Exhaustive analysis #
##########################
'''
Note that this step is run to generate a benchmark for the sampling methods.
If the aim is only for comparison of sampling methods, this step can be skipped.
'''
from helper import objective_func_reg

### Initialize the arrays to hold the results
PredictedAcc = np.zeros((len(bl_run)))
Weights = []

### Perform the exhaustive search
for i in tqdm(range(len(bl_run))):
    tempPredAcc, tempWeights = objective_func_reg(i, YPrediction, bl_run, rf_run, tw_run, el_run, FeaturePrediction)
    PredictedAcc[i] = tempPredAcc
    Weights.append(tempWeights)

### Save the results
plt.scatter(embed_features[0: PredictedAcc.shape[0], 0],
            embed_features[0: PredictedAcc.shape[0], 1],
            c=PredictedAcc, cmap='bwr')
plt.colorbar()
#plt.show()
pickle.dump( PredictedAcc, open(str(output_path / "PredictedAcc.p"), "wb" ) )
pickle.dump(Weights, open(str(output_path / "Weights.p"), "wb"))


#####################################
# 4. Comparison of sampling methods #
#####################################
### Import necessary libraries
from helper import (create_stratified_samples, 
                        objective_func_reg, initialize_bo, run_bo, 
                        plot_bo_estimated_space, plot_bo_evolution)
warnings.filterwarnings("ignore")

### Define sample sizes
sample_sizes = [53, 79]

### Initialize DataFrames to hold all the data
dist = pd.DataFrame(index=sample_sizes, columns=['Full vs Stratified', 'Full vs Random', 'Full vs Active Learning'])

### Loop over sample sizes
for num_items in sample_sizes:
    ### Load data from exhaustive search
    PredictedAcc = pickle.load(open(str(output_path / "PredictedAcc.p"), "rb" ) )
    Pipelines = pd.read_csv(str(output_path / "Pipelines.csv"))

    ### Active learning
    # Set the parameters
    n_burnin = 20 #Number of initial points to sample
    n_bayesopt = num_items 
    n_samples = n_burnin + n_bayesopt
    ModelEmbedding = embed_features
    kappa = 10  # Exploratory setting
    model_config = {}
    model_config['bl_run'] = bl_run
    model_config['rf_run'] = rf_run
    model_config['tw_run'] = tw_run
    model_config['el_run'] = el_run

    # Running the active learning
    kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed = initialize_bo(ModelEmbedding, kappa, n_burnin, n_bayesopt)
    BadIter, selected_pipelines, perf_pipelines = run_bo(optimizer, utility, init_points,
                    n_iter, pbounds, nbrs, RandomSeed,
                    ModelEmbedding, model_config,
                    YPrediction, FeaturePrediction,
                    MultivariateUnivariate=True, verbose=False)
    x_exploratory, y_exploratory, z_exploratory, x, y, gp, vmax, vmin = \
                                            plot_bo_estimated_space(kappa, BadIter,
                                                optimizer, pbounds,
                                                ModelEmbedding, PredictedAcc,
                                                kernel, output_path)
    corr, PredictedAcc_AL = plot_bo_evolution(kappa, x_exploratory, y_exploratory, z_exploratory, x, y, gp,
                    vmax, vmin, ModelEmbedding, PredictedAcc, n_samples, output_path)
    pickle.dump( PredictedAcc_AL, open(str(output_path / f"PredictedAcc_AL_{num_items}.p"), "wb" ) )
    print(f'Spearman correlation {corr}')
    
    # Read the pipelines sampled by the active learning
    pipeline_index = np.where(BadIter == 0)[0]  # Good pipelines
    pipeline_index = pipeline_index.astype(int).tolist()
    pipeline_index = [selected_pipelines[i] for i in pipeline_index]
    AL_df = pd.DataFrame(columns=['indices','bl_run', 'rf_run', 'tw_run', 'el_run'])
    for i in range(len(pipeline_index)):
        AL_df.loc[i] = [pipeline_index[i], bl_run[pipeline_index[i]], rf_run[pipeline_index[i]], tw_run[pipeline_index[i]], el_run[pipeline_index[i]]]
    AL_df['perf_pipelines'] = perf_pipelines
    AL_df.to_csv(str(output_path / f'ALPipelines_{num_items}.csv'), index=False)

    ### Random sampling
    random_indices = np.random.choice(range(len(PredictedAcc)), num_items, replace=False)
    PredictedAcc_Rand = PredictedAcc[random_indices]
    random_df = pd.DataFrame(columns=['indices','bl_run', 'rf_run', 'tw_run', 'el_run'])
    for i in range(len(random_indices)):
        random_df.loc[i] = [random_indices[i], bl_run[random_indices[i]], rf_run[random_indices[i]], tw_run[random_indices[i]], el_run[random_indices[i]]]
    random_df['perf_pipelines'] = PredictedAcc_Rand
    random_df.to_csv(str(output_path / f'RandomPipelines_{num_items}.csv'), index=False)

    ### Stratified sampling
    stratified_samples = create_stratified_samples(Pipelines, [num_items])
    stratified_sample = stratified_samples[f'Sample_{num_items}']
    sampled_indices = stratified_sample.index
    PredictedAcc_Stra = PredictedAcc[sampled_indices]
    stratified_df = pd.DataFrame(columns=['indices','bl_run', 'rf_run', 'tw_run', 'el_run'])
    for i in range(len(sampled_indices)):
        stratified_df.loc[i] = [sampled_indices[i], bl_run[sampled_indices[i]], rf_run[sampled_indices[i]], tw_run[sampled_indices[i]], el_run[sampled_indices[i]]]
    stratified_df['perf_pipelines'] = PredictedAcc_Stra
    stratified_df.to_csv(str(output_path / f'StratifiedPipelines_{num_items}.csv'), index=False)

    ### Calculate the median and KS statistic
    median_PredictedAcc = pd.Series(PredictedAcc).median()
    median_PredictedAcc_AL = pd.Series(PredictedAcc_AL).median()
    median_PredictedAcc_Rand = pd.Series(PredictedAcc_Rand).median()
    median_PredictedAcc_Stra = pd.Series(PredictedAcc_Stra).median()

    # Create the DataFrame
    medians_df = pd.DataFrame({
        'Sampling Method': ['PredictedAcc', 'PredictedAcc_AL', 'PredictedAcc_Rand', 'PredictedAcc_Stra'],
        'Median': [median_PredictedAcc, median_PredictedAcc_AL, median_PredictedAcc_Rand, median_PredictedAcc_Stra]
    })

    # Calculate the KS statistic for each pair of samples
    medians_df['KS Statistic'] = [
        ks_2samp(PredictedAcc, PredictedAcc).statistic,
        ks_2samp(PredictedAcc, PredictedAcc_AL).statistic,
        ks_2samp(PredictedAcc, PredictedAcc_Rand).statistic,
        ks_2samp(PredictedAcc, PredictedAcc_Stra).statistic
    ]
    medians_df.to_csv(str(output_path / f'medianAndKS_{num_items}.csv'), index=False)

    ### Identifying best pipeline
    N = 10 #Set the number of best pipelines
    full_sample_sorted = Pipelines.iloc[PredictedAcc.argsort()[-N:][::-1]]
    full_sample_best = full_sample_sorted.apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    full_sample_best = full_sample_best + ', ' + PredictedAcc[full_sample_sorted.index].astype(str)
    AL_sorted_indices = PredictedAcc_AL.argsort()[-N:][::-1]
    AL_best = Pipelines.iloc[AL_sorted_indices].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    AL_best = AL_best + ', ' + PredictedAcc_AL[AL_sorted_indices].astype(str)
    random_sorted = random_df.sort_values(by='perf_pipelines', ascending=False).head(N)
    random_best = random_sorted.drop(columns='indices').apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    stratified_sorted = stratified_df.sort_values(by='perf_pipelines', ascending=False).head(N)
    stratified_best = stratified_sorted.drop(columns='indices').apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    best_pipeline = pd.DataFrame({
        'Best Full Sample': full_sample_best.values,
        'Best Active Learning': AL_best.values,
        'Best Random Sampling': random_best.values,
        'Best Stratified Sampling': stratified_best.values
    })
    print(best_pipeline)
    best_pipeline.to_csv(str(output_path / f'bestPipelines_{num_items}.csv'), index=False)

    ### Plot raincloud plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue','green', 'orange', 'red'] #Full sample, active learning, stratified sampling, random sampling
    # Boxplot data
    bp = ax.boxplot([PredictedAcc, PredictedAcc_AL, PredictedAcc_Stra, PredictedAcc_Rand], patch_artist = True, vert = False)
    # Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    # Violinplot data
    vp = ax.violinplot([PredictedAcc, PredictedAcc_AL, PredictedAcc_Stra, PredictedAcc_Rand], points=500, 
                showmeans=False, showextrema=False, showmedians=False, vert=False)
    for idx, b in enumerate(vp['bodies']):
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
        b.set_color(colors[idx])
    # Scatterplot data
    for idx, data in enumerate([PredictedAcc, PredictedAcc_AL, PredictedAcc_Stra, PredictedAcc_Rand]):
        y = np.full(len(data), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        if idx == 1:  # Active Learning
            AL_indices = AL_df['indices'].values.astype(int)[n_burnin:]  # Exclude the pipelines in the burn-in phase
            colors_dots = ['darkgreen' if i in AL_indices else 'lightgreen' for i in range(len(data))]
            plt.scatter(data, y, s=10, c=colors_dots, alpha=0.6)
        else:
            plt.scatter(data, y, s=10, c=colors[idx])  # Set a consistent size for the dots

    plt.yticks(np.arange(1,5,1), ['Full Sample', 'Active learning', 'Stratified Sampling', 'Random Sampling'])  # Set text labels.
    plt.xlabel('R-square')
    plt.title("Sampled multiverse analysis across different sampling methods")
    plt.tight_layout() 
    plt.subplots_adjust(left=0.2) 
    plt.savefig(str(output_path / f'raincloud_plot_{num_items}samples.svg'))
    #plt.show()

    ###Plot the sampled pipelines in space
    # Define the indices
    AL_indices = AL_df['indices'].values.astype(int)[n_burnin:]  # Exclude the pipelines in the burn-in phase
    random_indices = random_df['indices'].values.astype(int)
    stratified_indices = stratified_df['indices'].values.astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Arrange subplots horizontally with square plots

    # Plot for Random Sampling
    axs[0].scatter(embed_features[:, 0], embed_features[:, 1], color='blue', label='All Pipelines', alpha=0.1)
    axs[0].scatter(embed_features[random_indices, 0], embed_features[random_indices, 1], color='red', label='Random Sampling')
    axs[0].legend()
    axs[0].set_title('Random Sampling')
    axs[0].grid(True)  # Add grid lines

    # Plot for Stratified Sampling
    axs[1].scatter(embed_features[:, 0], embed_features[:, 1], color='blue', label='All Pipelines', alpha=0.1)
    axs[1].scatter(embed_features[stratified_indices, 0], embed_features[stratified_indices, 1], color='orange', label='Stratified Sampling')
    axs[1].legend()
    axs[1].set_title('Stratified Sampling')
    axs[1].grid(True)  # Add grid lines

    # Plot for Active Learning
    axs[2].scatter(embed_features[:, 0], embed_features[:, 1], color='blue', label='All Pipelines', alpha=0.1)
    axs[2].scatter(embed_features[AL_indices, 0], embed_features[AL_indices, 1], color='green', label='Active Learning')
    axs[2].legend()
    axs[2].set_title('Active Learning')
    axs[2].grid(True)  # Add grid lines

    # Set common labels
    fig.text(0.5, 0.01, 'Dimension 1', ha='center', va='center')
    fig.text(0.01, 0.5, 'Dimension 2', ha='center', va='center', rotation='vertical')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08, left=0.04)  # Adjust the position of the labels
    plt.savefig(str(output_path / f'samplePipelines_{num_items}samples.svg'))
    #plt.show()

    ###Measuring the similarity between the sampled pipelines from different methods with the full sample
    from helper import average_nearest_neighbor_distance
    AL_coor = embed_features[AL_indices]
    Rand_coor = embed_features[random_indices]
    Stra_coor = embed_features[stratified_indices]
    full_coor = embed_features
    dist_AL = average_nearest_neighbor_distance(AL_coor, full_coor)
    dist_Rand = average_nearest_neighbor_distance(Rand_coor, full_coor)
    dist_Stra = average_nearest_neighbor_distance(Stra_coor, full_coor)
    dist.loc[num_items, 'Full vs Active Learning'] = dist_AL
    dist.loc[num_items, 'Full vs Random'] = dist_Rand
    dist.loc[num_items, 'Full vs Stratified'] = dist_Stra #The closer the distance to 0, the more similar the samples are to the full sample
    dist.to_csv(str(output_path / 'distances.csv'))



######################
# 5. Spec Curve Plot #
######################
### Active learning -- full 528 pipelines
import matplotlib.gridspec as gridspec
from helper import spec_curve

# Note that for AL, the data would be for all the pipelines. However for Random and Stratified, the data would be for the sampled pipelines
spec_curve(sampling_method = 'AL', num_items_spec = sample_sizes[0]) #"AL" for Active Learning, "Rand" for Random, "Stra" for Stratified, "Full" for Full sample
