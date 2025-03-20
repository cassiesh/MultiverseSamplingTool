clear all, close all, clc

mainpath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis\fork1-2\All analysis\';
storepath = [mainpath 'MDS\'];
cd(storepath);

% Initialize an empty array to store all data
LPP_values = [];

% Get a list of all .mat files in the directory
files = dir([mainpath '*.mat']);  

% Loop through each file
for i = 1:length(files)
    currentData = load([mainpath files(i).name]);
    LPP_values = [LPP_values; currentData.LPP_values];
end

save('LPP_values.mat', 'LPP_values');

% Requires the getCosineSimilarity function

% Format data for subsequent multidimensional scaling:
% Separate participant and condition from the rest of the data
% Create cells with format: {participant, condition, pipeline (all forks combined in one string), LPP value}
LPP_sep = {};
for x = 1:size(LPP_values,1)
    strTmp = LPP_values{x,1};

    splitting = split(LPP_values{x,1},'_');
    LPP_sep{x,1} = splitting{2};

    if     contains(strTmp, 'anger')     LPP_sep{x,2} = 'anger';
    elseif contains(strTmp, 'disgust')   LPP_sep{x,2} = 'disgust';
    elseif contains(strTmp, 'fear')      LPP_sep{x,2} = 'fear';
    elseif contains(strTmp, 'happiness') LPP_sep{x,2} = 'happiness';
    elseif contains(strTmp, 'neutral')   LPP_sep{x,2} = 'neutral';
    elseif contains(strTmp, 'sadness')   LPP_sep{x,2} = 'sadness';
    elseif contains(strTmp, 'surprise')  LPP_sep{x,2} = 'surprise';  
    end
        
    strTmp = splitting{4};
    strTmp = strrep(strTmp, 'anger', '');
    strTmp = strrep(strTmp, 'disgust', '');
    strTmp = strrep(strTmp, 'fear', '');
    strTmp = strrep(strTmp, 'happiness', '');
    strTmp = strrep(strTmp, 'neutral', '');
    strTmp = strrep(strTmp, 'sadness', '');
    strTmp = strrep(strTmp, 'surprise', '');
    LPP_sep{x,3} = strTmp; 

    LPP_sep{x,4} = LPP_values{x,2};
end

save('LPP_sep.mat', 'LPP_sep');
%%
% Split neutral from the six emotion conditions
neutral_indices = cellfun(@(x) strcmp(x,'neutral'), LPP_sep(:,2));
neutrals = LPP_sep(neutral_indices,:);
emotionals = LPP_sep(~neutral_indices,:);

subjects = unique(LPP_sep(:,1));
conditions = unique(emotionals(:,2));
paths = unique(LPP_sep(:,3));
%%
% For all pipelines x participants x (emotional) conditions, calculate LPP difference to neutral condition
mats = [];
for p = 1:length(paths)
    list_p = LPP_sep(cellfun(@(x) strcmp(x,paths{p}), LPP_sep(:,3)),:); 
    for s = 1:length(subjects)
        list_ps = list_p(cellfun(@(x) strcmp(x,subjects{s}), list_p(:,1)),:);
        lpp_n = cell2mat(list_ps(cellfun(@(x) strcmp(x,'neutral'), list_ps(:,2)),4)); % lpp_neutral
        for c = 1:length(conditions)
            lpp_c = cell2mat(list_ps(cellfun(@(x) strcmp(x,conditions{c}), list_ps(:,2)),4)); % lpp_condition
            mats(s,c,p) = lpp_c - lpp_n; % has the format [n_participants x n_emotionalconditions x n_total_paths] 
        end
    end
end
%%
% Creating small matrix of 20 participants 
mainpath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis\fork1-2\All analysis\';
files = {dir([mainpath]).name};
n_files = 42*20; % number of files for first 20 participants 
matssmall = [];
for p = 1:length(paths);
    list_p = LPP_sep(cellfun(@(x) strcmp(x,paths{p}), LPP_sep(:,3)),:); 
     for s = 1:20;
        list_ps = list_p(cellfun(@(x) strcmp(x,subjects{s}), list_p(:,1)),:);
        lpp_n = cell2mat(list_ps(cellfun(@(x) strcmp(x,'neutral'), list_ps(:,2)),4)); % lpp_neutral
        for c = 1:length(conditions)
            lpp_c = cell2mat(list_ps(cellfun(@(x) strcmp(x,conditions{c}), list_ps(:,2)),4)); % lpp_condition
            matssmall(s,c,p) = lpp_c - lpp_n; % has the format [n_participants x n_emotionalconditions x n_total_paths] 
        end
    end
end
%%
% Calculating cosine similarities and Euclidean distances
addpath('C:\Users\short\Documents\ERPSamplingStudy')
sims = []; % cosine similaritieS
% From the previous loop s and p are still at their upper limit
for ps = 1:p % loop across pipelines
    comp = 1;
    for ss = 1:s % loop across participants
        for ss2 = 1:s % compare participants with each other
            if ss < ss2 % after comparing e.g. participant 1 & 5, we don't need to compare 5 & 1 later
                sims(comp,ps) = getCosineSimilarity(mats(ss,:,ps),mats(ss2,:,ps));
                comp = comp +1;
            end
        end
    end
end
%%
% Load('mats20.mat'); % Contains 'matssmall'
edists = []; % Euclidian distances

for ed = 1:p % loop across pipelines
    edists(:,ed) = pdist(matssmall(:,:,ed)); % pdist() computes pairwise distances between rows in the input matrix (here: pdist is calculating the pairwise Euclidean distances between participants, compared across different emotional conditions, for each pipeline).
end

save('edists.mat', 'edists')
%%
% Save files
pdsim = pdist(sims'); 
pdsim2 = pdist(edists');
MDSCosineEmbed = cmdscale(pdsim,2); % mds based on cosine similarity between participants
MDSembed = cmdscale(pdsim2,2); % mds based on euclidian distances between participants

save('MDSCosineEmbed.mat', 'MDSCosineEmbed')
save('MDSembed.mat', 'MDSembed')
save('sims2.mat', 'edists')
save('pdsim2.mat', 'pdsim2')
save('mats.mat', 'mats')
save('mats20.mat', 'matssmall')
%%
%% Alternative dimension reduction methods - found t-SNE with perplexity of 5 to be the optimal method
% Load the necessary data
load('mats20.mat'); % contains 'matssmall'
load('pdsim2.mat'); % contains 'pairwise euclidean distances

storepath = 'C:\Users\short\Documents\ERPSamplingStudy\Python\';


% t-SNE on feature data
% Reshape matssmall to have dimensions (20*6*528) x 2
reshaped_data = reshape(matssmall, [], size(matssmall, 3));
% Transpose reshaped_data to have 528 data points, each with 120 features
reshaped_data_transposed = reshaped_data';
tSNEembed = tsne(reshaped_data_transposed, 'Algorithm', 'exact', 'Distance', 'euclidean', 'NumPCAComponents', 2);
save('tSNEembed.mat', 'tSNEembed');
save([storepath 'tSNEembed.mat'], 'tSNEembed');

% t-SNE on precomputed distances
distanceMatrix = squareform(pdsim2);
tSNEembedParticipants = tsne(distanceMatrix, 'Algorithm', 'exact', 'Distance', 'precomputed');
save('tSNEembedParticipants.mat', 'tSNEembedParticipants');
save([storepath 'tSNEembedParticipants.mat'], 'tSNEembedParticipants');

%tSNEembedNEW = tsne(pdsim2, 'Algorithm', 'exact', 'Distance', 'euclidean', 'NumPCAComponents', 2);
%save('tSNEembedNEW.mat', 'tSNEembed');
%save([storepath 'tSNEembedNEW.mat'], 'tSNEembed');


%% using edists

pdsim2 = pdist(edists');
% 'edists' is the data matrix with 190 rows and 528 columns
% Standardize the data
%stdedists = zscore(edists);

% Calculate the distance matrix between pipelines based on their distance vectors
pipelineDistanceMatrix = pdist(edists', 'euclidean');
pipelineDistanceMatrixSq = squareform(pipelineDistanceMatrix);

% Apply MDS to the pipeline distance matrix
[MDSEmbedPipelines, stress] = cmdscale(pipelineDistanceMatrixSq, 2);
save('MDSEmbedPipelines.mat', 'MDSEmbedPipelines');
save('StressMDSEmbedPipelines.mat', 'stress');

% Apply PCA
% Suitable for the pipeline distances, tryimg addimg explained
[coeff, PCAembedding, ~, ~, explained] = pca(pipelineDistanceMatrix');
save('PCAembeddingPipelines.mat', 'PCAembedding');
save('PCAembeddingCoefficientsPipelines.mat', 'coeff');
save('PCAembeddingExplainedPipelines.mat', 'explained');

% On the feature space
[coeff, PCAembedding, ~, ~, explained] = pca(edists');
% 'explained' gives the variance explained by each principal component
save('PCAembeddingFeatures.mat', 'PCAembedding');
save('PCAembeddingCoefficientsFeatures.mat', 'coeff');
save('PCAembeddingExplainedFeatures.mat', 'explained');

% Apply tSNE to both 
% PERPLEXITY  - cannot be greater than the number of rows of X (for pipelines this is 2)
% To pipeline distances
tSNEembedding = tsne(pipelineDistanceMatrix);
save('tSNEembeddingPipelines.mat', 'tSNEembedding');

% to the feature space PERPLEXITY 5
tSNEembedding = tsne(edists', 'Perplexity', 5);
save('tSNEembeddingFeatures5.mat', 'tSNEembedding');

% PERPLEXITY 30
%to the feature space
tSNEembedding = tsne(edists', 'Perplexity', 30);
save('tSNEembeddingFeatures30.mat', 'tSNEembedding');

% PERPLEXITY 50
%to the feature space
tSNEembedding = tsne(edists', 'Perplexity', 50);
save('tSNEembeddingFeatures50.mat', 'tSNEembedding');

% PERPLEXITY 100
%to the feature space
tSNEembedding = tsne(edists', 'Perplexity', 100);
save('tSNEembeddingFeatures100.mat', 'tSNEembedding');

% PERPLEXITY 190
%to the feature space
tSNEembedding = tsne(edists', 'Perplexity', 190);
save('tSNEembeddingFeatures190.mat', 'tSNEembedding');

% Apply UMAP to both (pipeline distances to be the same, but more commonly
% used on feature spaces)
% N_NEIGHBOURS (default 15) 15
% To pipeline distances
UMAPembedding = run_umap(pipelineDistanceMatrix);
save('UMAPembeddingPipelines15N.mat', 'UMAPembedding');

% To the feature space
UMAPembedding = run_umap(edists');
save('UMAPembeddingFeatures15N.mat', 'UMAPembedding');

% N_NEIGHBOURS (default 15) 100
% To pipeline distances
UMAPembedding = run_umap(pipelineDistanceMatrix, 'n_neighbors', 100);
save('UMAPembeddingPipelines100N.mat', 'UMAPembedding');

% To the feature space
UMAPembedding = run_umap(edists', 'n_neighbors', 100);
save('UMAPembeddingFeatures100N.mat', 'UMAPembedding');

%DEFAULT N_NEIGHBOURS, VARY MIN DIST
%Typical Values: Between 0.001 and 0.5. Smaller values lead to more clustered embeddings.
% MIN DIST 0.1 (default in Python)
% To pipeline distances
UMAPembedding = run_umap(pipelineDistanceMatrix, 'min_dist', 0.1);
save('UMAPembeddingPipelines01MD.mat', 'UMAPembedding');

% To the feature space
UMAPembedding = run_umap(edists', 'min_dist', 0.1);
save('UMAPembeddingFeatures01MD.mat', 'UMAPembedding');

% MIN DIST 0.3 
% To pipeline distances
UMAPembedding = run_umap(pipelineDistanceMatrix, 'min_dist', 0.3);
save('UMAPembeddingPipelines03MD.mat', 'UMAPembedding');

% To the feature space
UMAPembedding = run_umap(edists', 'min_dist', 0.3);
save('UMAPembeddingFeatures03MD.mat', 'UMAPembedding');

% MIN DIST 0.5 
% To pipeline distances
UMAPembedding = run_umap(pipelineDistanceMatrix, 'min_dist', 0.5);
save('UMAPembeddingPipelines05MD.mat', 'UMAPembedding');

% To the feature space
UMAPembedding = run_umap(edists', 'min_dist', 0.5);
save('UMAPembeddingFeatures05MD.mat', 'UMAPembedding');

