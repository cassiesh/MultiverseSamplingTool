%% Preparing Raw Data & Preprocessing up to and Including ICA

% Events

% S 18 & S 19 > fear condition
% S 28 & S 29 > anger condition
% S 38 & S 39 > disgust condition
% S 48 & S 49 > happiness condition
% S 56 & S 58 > neutral condition
% S 68 & S 69 > sadness condition
% S 78 & S 79 > surprised condition

clear all; close all; clc;

% DEFINE FOLDERS

mainpath = 'C:\Users\short\Documents\ERPSamplingStudy'; % main path
path_eeglab = [mainpath, '\EEGLAB']; % where eeglab is located
path_chanloc = [mainpath, '\RecordingInfo']; % where chanloc struct is located
path_rawdata = [mainpath, '\RawEEGData']; % where raw data is located
path_preprocessed = [mainpath, '\PreprocessedData']; % where pre-processed data is saved
path_condspecific = [mainpath, '\ConditionSpecificData']; % where condition specific datasets are saved
path_StdDevImages = [mainpath, '\StdDevImages']; % where standard deviation maps are saved for each participant, used to identify bad channels

% EXTRACT PARTICIPANT IDs & CREATE FOLDERS FOR EACH PARTICIPANT

% Extract participant IDs:
cd (path_rawdata)
files = dir('*.vhdr');
files = {files.name}; % participant IDs are stored here
for i = 1:length(files)
    files{i} = files {i}(1:8); % remove .vhdr extension and only keep the participant IDs
end
clear i;

% Load participant IDs:
load([mainpath, '\files.mat'], 'files');  

% CREATE INDIVIDUAL FOLDERS FOR MAIN PATH

% Create individual folders for primary path
for i = 1:length(files)
    mkdir([path_preprocessed, '\', 'primary', '\', files{i}]);
end
clear i;

% Create individual folders in a separate folder for condition specific datasets:
for i = 1:length(files)
    mkdir([path_condspecific, '\', 'primary', '\', files{i}]);
end
clear i;

% CREATE INDIVIDUAL FOLDERS FOR FORKING PATHS

% Create individual folders where data processed with -100 ms baseline are to be saved:
for i = 1:length(files)
    mkdir([path_preprocessed, '\', 'baseline_100', '\'  files{i}]);
end
clear i;

% Create individual folders (condition specific) where data with average reference are to be saved:
for i = 1:length(files)
    mkdir([path_condspecific, '\', 'reref_av', '\', files{i}]);
end
clear i;

% Create individual folders (condition specific) where data processed with -100 ms baseline are to be saved:
for i = 1:length(files)
    mkdir([path_condspecific, '\', 'baseline_100', '\', files{i}]);
end
clear i;

% Create individual folders (condition specific) where data processed with -100 ms baseline + average reference are to be saved:
for i = 1:length(files)
    mkdir([path_condspecific, '\', 'baseline_100', '\', 'reref_av', '\', files{i}]);
end
clear i;

% DEFINE PARAMETERS

irr_1 = 'Corr'; % irrelevant channel 1
irr_2 = 'Zyg'; % irrelevant channel 2
irr_3 = 'Orbi'; % irrelevant channel 3
resamp = 500; % resample to this value
ref_1 = 'Cz'; % first channel to re-reference (for cleaning)
highpassFreq = 1; % cut-off for the first high-pass filter
linenoisefreq = [50 100 150 200]; % line noise frequency 
events = {'S 18', 'S 19' 'S 28', 'S 29', 'S 38', 'S 39', 'S 48', 'S 49','S 56', 'S 58', 'S 68', ...
    'S 69', 'S 78', 'S 79'}; % list of the triggers (1st digit indicates emotion (i.e 1 = Fear) and 2nd digit indicates intensitiy level (i.e 8 = moderate / 9 = full)
responses = {'S  1', 'S  2', 'S  3', 'S  4', 'S  5', 'S  6', 'S  7'};
conds = {'fear', 'anger', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'}; % intensity level will be collapsed for analysis
epoch_start = -0.2;
epoch_end = 1;
base_start = -100;
lowpass = 30; % cut-off for the low-pass filter
mast_1 = 'A2'; % right mastoid
mast_2= 'A1'; % left mastoid

% Start loop across participants and load data

for i = 1:length(files)

    cd(path_eeglab);
    close all; clc;
    eeglab; % start eeglab (and restart it after every iteration of the loop)

    % Import raw data (.vhdr):
    EEG = pop_fileio([path_rawdata, '\' files{i}, '.vhdr']);
    EEG.setname = files{i};

    % Transform the data into a (.set) file:
    EEG = pop_saveset( EEG, 'filename',[files{i},'.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]); % save the data
    close all; clc;
    eeglab; % restart eeglab

    % Load the (.set) file:
    EEG = pop_loadset('filename',[files{i},'.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]);

    % Add channel info
    % Load channel location file
    load([path_chanloc, '\chanlocs.mat'])

    % Delete irrelevant channels:
    new_channel_irr_1 = numel(chanlocs) + 1; % the index for the new channel
    chanlocs(new_channel_irr_1).labels = irr_1; % name of a channel that was missing from chanlocs
    new_channel_irr_2 = numel(chanlocs) + 1; % the index for the new channel%chanlocs(new_channel_irr_2).labels = 'irr_2'; % name of a channel that was missing from chanlocs
    chanlocs(new_channel_irr_2).labels = irr_2; % name of a channel that was missing from chanlocsnew_channel_irr_3 = numel(chanlocs) + 1; % the index for the new channel
    new_channel_irr_3 = numel(chanlocs) +1;
    chanlocs(new_channel_irr_3).labels = irr_3; % name of a channel that was missing from chanlocs
    new_channel_ho1 = numel(chanlocs) + 1; % the index for the new channel
    chanlocs(new_channel_ho1).labels = 'HO1'; % name of a channel that was missing from chanlocs
    new_channel_ho2 = numel(chanlocs) + 1; % the index for the new channel
    chanlocs(new_channel_ho2).labels = 'HO2'; % name of a channel that was missing from chanlocs
    new_channel_vo2 = numel(chanlocs) + 1; % the index for the new channel
    chanlocs(new_channel_vo2).labels = 'VO2'; % name of a channel that was missing from chanlocs
    EEG = pop_select( EEG,'nochannel',{irr_1 irr_2 irr_3 });

    EEG.chanlocs = chanlocs;
    save('C:\Users\short\Documents\ERPSamplingStudy\RecordingInfo\ChanlocsBerlinData.mat', 'chanlocs');
    EEG = eeg_checkset(EEG);

    % Adding channel locations:
    for y = 1:EEG.nbchan
        EEG.chanlocs(y).sph_radius = chanlocs(y).sph_radius;
        EEG.chanlocs(y).sph_theta = chanlocs(y).sph_theta;
        EEG.chanlocs(y).sph_phi = chanlocs(y).sph_phi;
        EEG.chanlocs(y).theta = chanlocs(y).theta;
        EEG.chanlocs(y).radius = chanlocs(y).radius;
        EEG.chanlocs(y).X = chanlocs(y).X;
        EEG.chanlocs(y).Y = chanlocs(y).Y;
        EEG.chanlocs(y).Z = chanlocs(y).Z;
    end
    clear y;

    EEG.chanlocs(size(EEG.data,1)+1).labels = 'A1'; % add a new channel info to chanlocs struct (for left mastoid)
    EEG.data(size(EEG.data,1)+1,:) = 0; % add a new line on data (for left mastoid)
    EEG.nbchan = size(EEG.data,1); % update the number of channels
    EEG = eeg_checkset(EEG);

    EEG = pop_saveset( EEG, 'filename',[files{i}, '_chanlocsadded', '.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]); % save the dataset

    % Fixing the latency (150ms delay) of stimulus markers
    for e = 1:length(EEG.event)
        if ismember(EEG.event(e).type, events)
            EEG.event(e).latency = EEG.event(e).latency+150;
        end
    end
    clear e; clear x;

    eeglab redraw;

    EEG = pop_saveset( EEG, 'filename',[files{i}, '_latenciesfixed', '.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]); % save the dataset

    % Delete irrelevant markers so all markers match logfiles

    EEG_event_type = {EEG.event.type};
    g = ismember(EEG_event_type, events)  | ismember(EEG_event_type, responses); % find the stimulus & response markers
    EEG.event = EEG.event(g); % select only the stimulus & response markers, delete the rest
    if i == 3;
        EEG.event(670) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 6;
        EEG.event(1323) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 9;
        EEG.event(596) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 27;
        EEG.event(980) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 29;
        EEG.event(851) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 45;
        EEG.event(1077) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 61;
        EEG.event(1292) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 69;
        EEG.event(1554) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 73;
        EEG.event(670) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 80;
        EEG.event(165) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 81;
        EEG.event(1302) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 83;
        EEG.event(1041) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 91;
        EEG.event(1191) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end
    if i == 96;
        EEG.event(1469) = []; % there is an extra response marker for this participant here (which is not present in the Log file)
    end

    %% Add RTs from the logfiles to EEG.event struct

    Log_file = [];
    Log_file =  dlmread([mainpath, '\RawEEGData\LogFiles\', files{i}(1:5), 'E4.txt'],'\t', 1, 0);

    k = 1;
    for e = 1:2:length(EEG.event)
        EEG.event(e).RT = Log_file(k,5);
        k = k+1;
    end
    clear e; clear k;

    % Index to stimulus markers that are followed by incorrect responses & responses that appeared before 200ms
    % Reject trials with respect to responses & reaction times (RT):
    % (Response criteria: incorrect responses i.e S18 (stimuli: fear) followed by S7 (response: surprise))
    % (Reaction time criteria: < 200ms)
    e = 1;
    x = 1;
    while e < numel(EEG.event)
        if EEG.event(e).type(size(EEG.event(e).type,2)-1) ~= EEG.event(e+1).type(size(EEG.event(e+1).type,2))
            incorr(:,x) = e; % indices of stimulus markers with incorrect responses
            remaining_incorr(x) = e; % for sanity check - indices of stimulus markers with remaining incorrect trials
            x = x+1;
        end
        e = e+2;
    end
    if exist('incorr')
        EEG.event(incorr) = []; % delete those markers
        xlswrite([path_preprocessed, '\primary\' files{i}, '\incorrect_trials.xlsx'], incorr); % save the incorrect trials
        clear incorr;
    end

    % Identify and remove trial with RT < 200ms
    e = 1;
    x = 1;
    while e < numel(EEG.event)
        if EEG.event(e).RT < 200
            fast_rt(:,x) = e; % indices of stimulus markers with responses appearing before 200ms
            x = x+1;
        end
        e = e+2;
    end
    if exist('fast_rt')
        EEG.event(fast_rt) = []; % delete those markers
        xlswrite([path_preprocessed, '\primary\' files{i}, '\trials_with_fast_RTs.xlsx'], fast_rt) % save the trials with fast RTs.
        clear fast_rt;
    end

    clear e; clear x;

    % Keep only participants who have markers present for all conditions
    markers_present = [];
    markers_present = {EEG.event.type};

    if exist('r')
    else
        r = 1;
    end
    if any(strcmp(markers_present, 'S 18')) == 0 | any(strcmp(markers_present, 'S 19')) == 0 | any(strcmp(markers_present, 'S 28')) == 0 | any(strcmp(markers_present, 'S 29')) == 0 ...
            | any(strcmp(markers_present, 'S 38')) == 0 | any(strcmp(markers_present, 'S 39')) == 0 | any(strcmp(markers_present, 'S 48')) == 0 | any(strcmp(markers_present, 'S 49')) == 0 ...
            | any(strcmp(markers_present, 'S 56')) == 0 | any(strcmp(markers_present, 'S 58')) == 0 | any(strcmp(markers_present, 'S 68')) == 0 | any(strcmp(markers_present, 'S 69')) == 0 ...
            | any(strcmp(markers_present, 'S 78')) == 0 | any(strcmp(markers_present, 'S 79')) == 0
        removeParticipants(:,r) = i; % if not, save the indices for that participant
        r = r+1;
    end

    clear markers_present;
    EEG = pop_saveset( EEG, 'filename',[files{i}, '_rtincorrremoved', '.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]); % save the dataset
end

if exist('removeParticipants')
    files(removeParticipants) = []; % delete participants (they will not be analysed further)
end
clear i;

% resample and rereference

for i = 1:length(files)

    cd(path_eeglab);
    close all; clc;
    eeglab; % start eeglab (and restart it after every iteration of the loop)

    EEG = pop_loadset('filename',[files{i}, '_rtincorrremoved', '.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]);

    % Resample:
    EEG = pop_resample( EEG, resamp);

    % Re-reference (to Cz, for cleaning):
    EEG = pop_reref( EEG, find(strcmpi(ref_1,{EEG.chanlocs.labels})));

    % Save the dataset with both resampled and rereferenced data:
    EEG = pop_saveset(EEG, 'filename', [files{i}, '_resampled_rereferenced.set'], 'filepath', [path_preprocessed, '\primary\', files{i}]);

    % High-pass filter data for SD maps

    EEG = pop_eegfiltnew(EEG, highpassFreq);

    % Create standard deviation voltage maps to manually identify bad channels

    % Calculate the standard deviation of the voltage for each channel
    stdDevChannels = std(EEG.data, 0, 2);

    % Create the color map based on the color range
    colormap(jet); 
    cmap = colormap;
    colorIdx = interp1(colorRange, 1:size(cmap, 1), stdDevChannels, 'nearest', 'extrap');

    windowSize = 5000;  % number of data points in each iteration
    numIterations = floor(EEG.pnts / windowSize);  % calculate the total number of iterations

    % Preallocate a matrix for storing standard deviations at each time point
    stdDevData = zeros(EEG.nbchan, numIterations);

    % Create the time vector
    timeVector = zeros(1, numIterations);  % initialize the time vector

    for b = 1:numIterations
        startIndex = (b - 1) * windowSize + 1;  % calculate the start index for the current iteration
        endIndex = b * windowSize;  % calculate the end index for the current iteration
        % Calculate the time value for the current window
        timeVector(b) = EEG.times(startIndex)/500;  % EEG.times is the time vector for the full recording
        for channel = 1:EEG.nbchan
            tempData = EEG.data(channel, startIndex:endIndex);  % extract data within the current window
            tempDev = std(tempData, 'omitnan');
            stdDevData(channel, b) = std(tempData, 'omitnan');  % calculate the standard deviation within the window
        end
    end

    % Calculate the maximum standard deviation value
    maxStdDev = max(stdDevData(:));

    % Plot the standard deviation of the voltage
    figure;
    imagesc(timeVector, 1:EEG.nbchan, stdDevData);
    colormap(gca, cmap);
    colorbar('location', 'eastoutside');
    set(gca, 'YDir', 'normal');
    xlabel('Time');
    ylabel('Channel');
    title('Standard Deviation of Voltage');

    % Save the image
    fileName = sprintf('Participant%d_SDVoltageNew.png', i);
    fullPath = fullfile(path_StdDevImages, fileName);
    saveas(gcf, fullPath);

    close;
end

% Define the list of channels to be removed for each participant 
% decided by manual inspection of the SD map for each participant
badChannels = {11, 39, {}, {}, 39, {}, {}, {}, 39, 39, [6, 10, 15, 18], {}, {}, {}, {}, {}, 15, {}, 39, {}, {}, 40, {}, {}, {}, {}, {}, {}, 39, {}, {}, {}, [18, 39], {}, {}, {}, {}, {}, {}, {}, {}, 39, 39, {}, {}, 39, {}, [6, 10, 14, 15, 18], {}, {}, {}, {}, {}, {}, 40, [39, 40], 39, {}, {}, 8, {}, {}, {}, {}, {}, 39, 39, {}, {}, [15, 39], {}, {}, 33, 39, {}, 39, {}, {}, 20, 39, {}, 40, {}, [19, 24], [6, 11, 15, 37, 39], {}, {}, {}, {}, [10, 11, 14, 15, 39], {}, {}, 39, {}, {}, 39, 39, {}, 39, 19, {}, {}};
metrics_all = cell(length(files), 1);

bsl_fork = {-200; -100};
ref_fork = {'Mas'; 'Avg'; 'REST'};

responses = {'S  1', 'S  2', 'S  3', 'S  4', 'S  5', 'S  6', 'S  7'};
conds = {'fear', 'anger', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'};

win_fork = {500,200; 500,300; 600,200; 600,300; 600,600; 700,200; 700,300;...
    700,600; 450,100; 'GAV', 200; 'SAV', 200};
elec_fork = {{'CP1', 'CP2', 'Pz', 'P3', 'P4'}; {'P3', 'P4', 'CP1', 'CP2'}; {'P3','Pz','P4'};...
    {'Fz','Cz', 'Pz'}; {'CP1', 'CP2'}; {'Cz'}; {'Pz'};{'around_peak'}};

midline = {'Fpz', 'Fz', 'Cz', 'Pz', 'Oz', 'Iz'};

storepath = 'C:\Users\short\Documents\EEG_Active_Learning_Project\May2023\Analysis\'; % main path

% Iterate over participants
for i = 1:length(files)

    cd(path_eeglab);
    close all; clc;
    eeglab; % start eeglab (and restart it after every iteration of the loop)

    % Load the EEG data for the current participant
    EEG = pop_loadset('filename',[files{i}, '_resampled_rereferenced', '.set'],'filepath',[path_preprocessed, '\', 'primary', '\', files{i}]);

    % Define the list of channels to be marked as bad for the current participant
    participantBadChannels = badChannels{i};

    % Mark bad channels
    % Check if participantBadChannels is empty
    if ~isempty(participantBadChannels)
        % Mark bad channels 
        EEG = pop_rejchan(EEG, 'elec', participantBadChannels);
    end

    % High-pass filter
    EEG = pop_eegfiltnew(EEG, highpassFreq);

    % Run ICA
    EEG = pop_runica(EEG, 'icatype', 'runica');

    % Use ICLabel to detect bad components
    EEG = iclabel(EEG);

    % Define the threshold for component rejection
    threshold = 0.85;

    % Get the ICLabel probabilities
    iclabelProb = squeeze(EEG.etc.ic_classification.ICLabel.classifications(:, 1));

    % Identify bad components based on the threshold
    badComponents = find(iclabelProb > threshold);

    % Reject bad components
    EEG = pop_subcomp(EEG, badComponents, 0);

    % Low-pass filter
    lowpassFreq = 30;
    EEG = pop_eegfiltnew(EEG, [], lowpassFreq, [], 0, [], 0);

    % Interpolate bad channels using spherical spline interpolation
    if ~isempty(participantBadChannels)
        EEG = pop_interp(EEG, participantBadChannels, 'spherical');
    end

    % save as post ICA and interpolated.
    EEG = pop_saveset(EEG, 'filename', [files{i}, '_postICA_interpolated.set'], 'filepath', [path_preprocessed, '\primary\', files{i}]);

end

