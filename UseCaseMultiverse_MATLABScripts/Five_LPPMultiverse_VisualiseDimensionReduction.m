clear all, close all, clc

mainpath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis\fork1-2\All analysis\MDS';
addpath(mainpath);
storepath = [mainpath];
cd(storepath);
load("LPP_sep.mat");

%% load the embedded data file needed (t-SNE perplexity 5 in the main paper, the rest in supplementary) - uncomment the embedded data file needed - here t-SNE perplexity 5 (use case in paper) is uncommented.

% to load t-SNE .mat files:
% Specify the perplexity value
 perplexity = 5; % Change this value to 5, 30, 50, 100, or 190 to load the corresponding t-SNE file
% Dynamically construct the file name based on the specified perplexity
 tsneFileName = sprintf('tSNEembeddingFeatures%d.mat', perplexity);
% Load the t-SNE embedding data from the dynamically constructed file name
 load(fullfile(mainpath, tsneFileName), 'tSNEembedding'); % Load the variable tSNEembedding

%load("MDSEmbedPipelines.mat");
%load("PCAembeddingFeatures.mat");  
%load("UMAPembeddingFeatures15N.mat");
%load("UMAPembeddingFeatures100N.mat");
%load("UMAPembeddingFeatures01MD.mat");
%load("UMAPembeddingFeatures03MD.mat");
%load("UMAPembeddingFeatures05MD.mat");


% Previously I separated participant from condition and pipeline
% Here I split the pipelines into the 4 decision nodes (baseline, reference, timewindow, electrodes)
allpaths = {};

LPP_sepu = [LPP_sep(1:264,:); LPP_sep(1849:1849+263,:)];

for s = 1:size(LPP_sepu,1)
    allpaths{s,1} = LPP_sepu{s,3}(3:5);
    allpaths{s,2} = LPP_sepu{s,3}(7:9);
    allpaths{s,3} = LPP_sepu{s,3}(10:15);
    allpaths{s,4} = LPP_sepu{s,3}(16:end);
end

% Plot
markers = {'450100', 'o'; '500200', '+'; '500300', '*'; '600200', 'x'; '600300', '_'; '600600', '|'; ...
    '700200', 's'; '700300', 'd'; '700600', '^'; 'GAV400', 'p'; 'SAV400', 'h'};
colors = {'CP1CP2', [1 0 0]; 'CP1CP2PzP3P4', [0 1 0]; 'Cz', [0 0 1]; 'FzCzPz', [1 1 0]; 'P3P4CP1CP2', [1 0 1];...
    'P3PzP4' , [0 1 1]; 'Pz',[0.667 0.667 0.667];'around', [1 0.75 0]};

% Plot datapoints one-by-one and choose the shape, color, and size based on the pipeline options

% time window determines the symbol
% electrodes determines the color
% reference determines darkness of the color
% baseline determines size 

figure
hold on
for p = 1:size(allpaths,1)
    msize = 0;
    mcolor = 'x';
    mtype = 'no';

    mt = find(strcmp(allpaths{p,3},markers));
    mtype = markers{mt,2};
    cs = find(strcmp(allpaths{p,4},colors));
    mcolor = colors{cs,2};

    switch allpaths{p,1}
        case '100'
            msize = 6;
        case '200'
            msize = 10;
    end
    switch allpaths{p,2}
        case 'Avg'
            mcolor = mcolor*0.4;
        case 'Mas'
            mcolor = mcolor*0.8;
        case 'CSD' % mixing with white for extra distinction of brightness
            mcolor = (mcolor + [1 1 1] * 1.5) / 2.5;
    end
   
  %% Uncomment the plot to be created
  plot(tSNEembedding(p,1),tSNEembedding(p,2),mtype,'MarkerEdgeColor',mcolor,'MarkerFaceColor',mcolor,'MarkerSize',msize)
  % plot(MDSEmbedPipelines(p,1),MDSEmbedPipelines(p,2),mtype,'MarkerEdgeColor',mcolor,'MarkerFaceColor',mcolor,'MarkerSize',msize)
  % plot(PCAembedding(p,1),PCAembedding(p,2),mtype,'MarkerEdgeColor',mcolor,'MarkerFaceColor',mcolor,'MarkerSize',msize)
  % plot(UMAPembedding(p,1),UMAPembedding(p,2),mtype,'MarkerEdgeColor',mcolor,'MarkerFaceColor',mcolor,'MarkerSize',msize)
  
hold on   
end
% Set axes background color to white
set(gca, 'Color', 'w'); 

%%
% Legend
% since I constructed the plot with a for-loop and 'hold', I construct a matching legend with NaN data points

timewindows = unique(allpaths(:,3));
electrodes = unique(allpaths(:,4));
references = unique(allpaths(:,2));
baselines = unique(allpaths(:,1));

% Time windows
for t = 1:size(timewindows, 1)
    h(t) = plot(NaN, NaN, markers{t, 2}, 'MarkerEdgeColor', [0, 0, 1], 'MarkerFaceColor', [0, 0, 1], 'MarkerSize', 6);
end
h(t+1) = plot(NaN, NaN, 'w'); % spacer

% Electrodes
offset = t + 1;
for e = 1:size(electrodes, 1)
    h(offset + e) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', colors{e, 2}, 'MarkerFaceColor', colors{e, 2}, 'MarkerSize', 6);
end
h(offset + e + 1) = plot(NaN, NaN, 'w'); % spacer

% References
offset = offset + e + 1;
for r = 1:size(references, 1)
    switch r
        case 1
            refcoeff = 0.4; % sets darkness level
            mcolor = [0 0 1] * refcoeff;
        case 2
            refcoeff = 0.8;
            mcolor = [0 0 1] * refcoeff;
        case 3
            refcoeff = 1.0;
            mcolor = ([0 0 1] + [1 1 1] * 1.5) / 2.5; % white blending
    end
    h(offset + r) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', mcolor, 'MarkerFaceColor', mcolor, 'MarkerSize', 6);
end
h(offset + r + 1) = plot(NaN, NaN, 'w'); % spacer

% Baselines
offset = offset + r + 1;
for b = 1:size(baselines, 1)
    switch b
        case 1
            bslsize = 10;
        otherwise
            bslsize = 6;
    end
    h(offset + b) = plot(NaN, NaN, 'o', 'MarkerEdgeColor', [0, 0, 1], 'MarkerFaceColor', [0, 0, 1], 'MarkerSize', bslsize);
end

% Create the legend

lgd = legend(h, [timewindows; ' '; electrodes; ' '; {'Avg (darkest)'; 'Mas (medium)'; 'CSD (lightest)'}; ' '; {'-200 (bigger)'; '-100 (smaller)'}]);
lgd.Location = 'eastoutside';
lgd.Orientation = 'vertical';
lgd.FontSize = 10;

% Adjust the figure size and layout
set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]); % Set figure size and position

% Saving figure - uncomment the correct chunk depending on embedded data plotted

% Saving for tsne figures
% Dynamically construct the file name based on the perplexity value
 outputFileName = sprintf('figure_tSNEPerplexity%d', perplexity); 
% Save as PNG with 300 DPI
 print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for cMDS figure
% Construct the file name for cMDS
%outputFileName = 'figure_cMDS'; % Static file name for cMDS plot
% Save the figure as a PNG file with 300 DPI
%print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for PCA figure
% Construct the file name for cMDS
%outputFileName = 'figure_PCA'; % Static file name for cMDS plot
% Save the figure as a PNG file with 300 DPI
%print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for UMAP n_neighbours 15 figure
% Construct the file name for UMAP figure
%outputFileName = 'figure_UMAP_15'; % Static file name for UMAP plot
% Save the figure as a PNG file with 300 DPI
%print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for UMAP n_neighbours 100 figure
% Construct the file name for UMAP figure
%outputFileName = 'figure_UMAP_100'; % Static file name for UMAP plot
% Save the figure as a PNG file with 300 DPI
%print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for UMAP min_dist 0.1 figure
% Construct the file name for UMAP figure
% outputFileName = 'figure_UMAP_01MD'; % Static file name for UMAP plot
% Save the figure as a PNG file with 300 DPI
% print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for UMAP min_dist 0.3 figure
% Construct the file name for UMAP figure
%outputFileName = 'figure_UMAP_03MD'; % Static file name for UMAP plot
% Save the figure as a PNG file with 300 DPI
%print(gcf, outputFileName, '-dpng', '-r300'); 

% Saving for UMAP min_dist 0.5 figure
% Construct the file name for UMAP figure
% outputFileName = 'figure_UMAP_05MD'; % Static file name for UMAP plot
% Save the figure as a PNG file with 300 DPI
% print(gcf, outputFileName, '-dpng', '-r300'); 

