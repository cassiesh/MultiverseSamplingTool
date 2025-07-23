clear; clc; close all;

% Setup
mainpath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis\fork1-2\All analysis\MDS';
cd(mainpath);
load('mats20.mat'); % loads mats20 [20 participants x 6 conditions x 528 pipelines]
load('LPP_sep.mat');   % loads LPP_sep for pipeline decoding

% Decode pipeline options
LPP_sepu = [LPP_sep(1:264,:); LPP_sep(1849:1849+263,:)];
for s = 1:size(LPP_sepu,1)
    allpaths{s,1} = LPP_sepu{s,3}(3:5);       % baseline
    allpaths{s,2} = LPP_sepu{s,3}(7:9);       % reference
    allpaths{s,3} = LPP_sepu{s,3}(10:15);     % timewindow
    allpaths{s,4} = LPP_sepu{s,3}(16:end);    % electrodes
end

% Compute pairwise distances across pipelines
for ed = 1:size(matssmall,3)
    edists(:,ed) = pdist(matssmall(:,:,ed)); % Euclidean distances between participants
end

% Marker setup
markers = {'450100', 'o'; '500200', '+'; '500300', '*'; '600200', 'x'; '600300', '_'; '600600', '|'; ...
           '700200', 's'; '700300', 'd'; '700600', '^'; 'GAV400', 'p'; 'SAV400', 'h'};
colors = {'CP1CP2', [1 0 0]; 'CP1CP2PzP3P4', [0 1 0]; 'Cz', [0 0 1]; 'FzCzPz', [1 1 0]; ...
          'P3P4CP1CP2', [1 0 1]; 'P3PzP4', [0 1 1]; 'Pz',[0.667 0.667 0.667]; 'around', [1 0.75 0]};
timewindowLabels = markers(:,1);
electrodeLabels  = colors(:,1);
referenceLabels  = {'Avg (darkest)', 'Mas (medium)', 'CSD (lightest)'};
baselineLabels   = {'-100 (smaller)', '-200 (bigger)'};

tsneResults = cell(10,1); % to store embeddings

% Run and plot t-SNE with 10 seeds
for seed = 1:10
    rng(seed); % set seed
    tSNEembedding = tsne(edists', 'Perplexity', 5, 'Distance', 'euclidean', 'Algorithm', 'exact');
    tsneResults{seed} = tSNEembedding;

    figure('Visible','off'); hold on;

    for p = 1:size(allpaths,1)
        mtype = 'o'; msize = 6; mcolor = [0 0 0];

        mt = find(strcmp(allpaths{p,3},markers(:,1)));
        if ~isempty(mt), mtype = markers{mt,2}; end

        cs = find(strcmp(allpaths{p,4},colors(:,1)));
        if ~isempty(cs), mcolor = colors{cs,2}; end

        if strcmp(allpaths{p,1}, '200'), msize = 10; else, msize = 6; end

        switch allpaths{p,2}
            case 'Avg'
                mcolor = mcolor * 0.4;
            case 'Mas'
                mcolor = mcolor * 0.8;
            case 'CSD'
                mcolor = (mcolor + [1 1 1] * 1.5) / 2.5;
        end

        plot(tSNEembedding(p,1), tSNEembedding(p,2), mtype, ...
            'MarkerEdgeColor', mcolor, 'MarkerFaceColor', mcolor, 'MarkerSize', msize);
    end

    axis off;
    set(gca, 'Color', 'w');
    title(sprintf('t-SNE Embedding (Seed %d)', seed));
    set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.75, 0.75]);

    % Create empty handles for legend
    h = [];

    % Time window (marker shape)
    for t = 1:length(timewindowLabels)
        h(end+1) = plot(NaN, NaN, markers{t,2}, 'MarkerEdgeColor', [0 0 0], ...
                        'MarkerFaceColor', [0 0 0], 'MarkerSize', 6);
    end
    twLabels = timewindowLabels;

    % Electrode cluster (marker color)
    for e = 1:length(electrodeLabels)
        h(end+1) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', colors{e,2}, ...
                        'MarkerFaceColor', colors{e,2}, 'MarkerSize', 6);
    end
    elLabels = electrodeLabels;

    % Reference (brightness)
    refColors = {[0 0 1]*0.4, [0 0 1]*0.8, ([0 0 1] + [1 1 1]*1.5)/2.5};
    for r = 1:length(referenceLabels)
        h(end+1) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', refColors{r}, ...
                        'MarkerFaceColor', refColors{r}, 'MarkerSize', 6);
    end

    % Baseline (marker size)
    h(end+1) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', [0 0 1], ...
                    'MarkerFaceColor', [0 0 1], 'MarkerSize', 6); % -100
    h(end+1) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', [0 0 1], ...
                    'MarkerFaceColor', [0 0 1], 'MarkerSize', 10); % -200

    % Assemble full label list
    allLabels = [twLabels; electrodeLabels; referenceLabels'; baselineLabels'];

    % Create and place legend
    lgd = legend(h, allLabels);
    lgd.Location = 'eastoutside';
    lgd.FontSize = 10;

    % Save figure
    saveName = sprintf('figure_tSNE_seed%d', seed);
    print(gcf, saveName, '-dpng', '-r300');
end

disp('All 10 t-SNE runs completed and saved.');

% Check each embedding is a numeric 2D matrix
for i = 1:length(tsneResults)
    if ~isnumeric(tsneResults{i}) || ~ismatrix(tsneResults{i})
        fprintf('Problem with tsneResults{%d}: %s\n', i, class(tsneResults{i}));
    end
end

%% Compute Procrustes distances between embeddings
n = length(tsneResults);
procrustesMatrix = zeros(n,n);

for i = 1:n
    for j = i+1:n
        [procrustesMatrix(i,j), ~, ~] = procrustes(tsneResults{i}, tsneResults{j});
        procrustesMatrix(j,i) = procrustesMatrix(i,j); % symmetric
    end
end

save('procrustes_tSNE_seeds.mat', 'procrustesMatrix');
meanProcrustes = mean(procrustesMatrix(triu(true(n),1)));
fprintf('Mean Procrustes distance across t-SNE seeds: %.4f\n', meanProcrustes);

% Compute Spearman correlation between pairwise distances of t-SNE embeddings
n = length(tsneResults);
spearmanMatrix = zeros(n, n);

for i = 1:n
    D1 = squareform(pdist(tsneResults{i}));  % 528 x 528 pairwise distance matrix
    D1_vec = D1(triu(true(size(D1)), 1));    % upper triangle vector

    for j = i+1:n
        D2 = squareform(pdist(tsneResults{j}));
        D2_vec = D2(triu(true(size(D2)), 1));

        rho = corr(D1_vec, D2_vec, 'Type', 'Spearman');  % Spearman rank correlation
        spearmanMatrix(i,j) = rho;
        spearmanMatrix(j,i) = rho;  % symmetry
    end
end

% Save results
save('spearman_tSNE_seeds.mat', 'spearmanMatrix');

% Compute and print mean Spearman correlation
meanSpearman = mean(spearmanMatrix(triu(true(n),1)));
fprintf('Mean Spearman correlation of pairwise distances across t-SNE seeds: %.4f\n', meanSpearman);


%% Quantify local structure similarity across seeds

% Compute Local Continuity Meta-Criterion (LCMC) across t-SNE seeds
% Assumes tsneResults is a cell array of 10 [528 x 2] t-SNE embeddings

k = 20; % number of neighbors to consider
nSeeds = length(tsneResults);
N = size(tsneResults{1}, 1); % number of pipelines
LCMC_matrix = zeros(nSeeds, nSeeds); % to store LCMC values

% Precompute neighbors for each embedding
allNeighbors = cell(nSeeds,1);
for i = 1:nSeeds
    allNeighbors{i} = knnsearch(tsneResults{i}, tsneResults{i}, 'K', k+1); % includes self
    allNeighbors{i} = allNeighbors{i}(:,2:end); % remove self from neighbors
end

% Compute LCMC for each embedding pair
for i = 1:nSeeds
    for j = i+1:nSeeds
        overlapSum = 0;
        for n = 1:N
            neighbors_i = allNeighbors{i}(n,:);
            neighbors_j = allNeighbors{j}(n,:);
            overlap = numel(intersect(neighbors_i, neighbors_j));
            overlapSum = overlapSum + overlap;
        end
        % Normalize LCMC value
        maxOverlap = N * k;
        LCMC_val = overlapSum / maxOverlap;
        LCMC_matrix(i,j) = LCMC_val;
        LCMC_matrix(j,i) = LCMC_val;
    end
end

% Visualize LCMC heatmap
figure;
heatmap(LCMC_matrix, ...
    'Colormap', parula, ...
    'ColorLimits', [0 1], ...
    'Title', 'LCMC (Local Neighborhood Overlap)', ...
    'XDisplayLabels', 1:nSeeds, ...
    'YDisplayLabels', 1:nSeeds);

% Compute mean LCMC (excluding diagonal)
meanLCMC = mean(LCMC_matrix(triu(true(size(LCMC_matrix)),1)));
fprintf('Mean LCMC (k=%d) across t-SNE seeds: %.4f\n', k, meanLCMC);
