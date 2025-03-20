% Define the path to the folder containing the files
filePath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis\fork1-2';

% Get a list of all .set files in the folder
fileInfo = dir(fullfile(filePath, '*.set')); % Only files with .set extension

% Extract the full filenames into a cell array
fileNames = {fileInfo.name};

% Extract IDs
idPattern = 'fork1-2_(\d+)_'; % Looks for "fork1-2_" followed by digits
allIDs = cellfun(@(x) regexp(x, idPattern, 'tokens'), fileNames, 'UniformOutput', false);

% Flatten the nested cell array and convert to numeric values
allIDs = cellfun(@(x) str2double(x{1}), allIDs(~cellfun('isempty', allIDs)));

% Get unique IDs
uniqueIDs = unique(allIDs);

% Display the unique IDs as a check
disp('Unique Subject IDs:');
disp(uniqueIDs);

% SPECIFY DATA TO BE PLOTTED

% Set the channels of interest manually
channelsOfInterest = {'CP1', 'CP2'};

% Define the reference types
referenceTypes = {'CSD', 'Avg', 'Mas'};

% Load one EEG dataset to initialise the time vector
EEG = pop_loadset('filename', ['fork1-2_' num2str(uniqueIDs(1)) '_E4_b-200neutralrCSD.set'], 'filepath', filePath);
% Extract the time vector
time1 = EEG.times;

% Loop through each reference type
for refIdx = 1:length(referenceTypes)
    refType = referenceTypes{refIdx}; 

    % Dynamically construct the file names for the current reference type
    fileNames = {
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200neutralr' refType '.set'], uniqueIDs, 'UniformOutput', false),    
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200fearr' refType '.set'], uniqueIDs, 'UniformOutput', false),
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200angerr' refType '.set'], uniqueIDs, 'UniformOutput', false),
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200happinessr' refType '.set'], uniqueIDs, 'UniformOutput', false),
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200disgustr' refType '.set'], uniqueIDs, 'UniformOutput', false),
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200sadnessr' refType '.set'], uniqueIDs, 'UniformOutput', false),
        arrayfun(@(x) ['fork1-2_' num2str(x) '_E4_b-200surpriser' refType '.set'], uniqueIDs, 'UniformOutput', false)
    };

    % Initialise variables for averaging ERP waves for each set
    averagedERPs = zeros(length(fileNames), length(time1));
    
    % Loop through each set of filenames
    for setNum = 1:length(fileNames)
        % Loop through the datasets in the current set and calculate the average ERP wave
        for i = 1:length(fileNames{setNum})
            % Load the dataset
            EEG = pop_loadset('filename', fileNames{setNum}{i}, 'filepath', filePath);

            % Extract the data for the specified channels and accumulate the ERP waveforms
            data = pop_select(EEG, 'channel', channelsOfInterest);
            averagedERPs(setNum, :) = averagedERPs(setNum, :) + mean(mean(data.data, 3), 1);
        end

        % Divide the accumulated ERP waveform by the number of datasets in the current set
        averagedERPs(setNum, :) = averagedERPs(setNum, :) / length(fileNames{setNum});
    end

    % Create the waveform figure for the current reference type
    figure;
    hold on; % all conditions on the same figure
    defaultLineWidth = 1; % slimmer line width for all except neutral
    neutralLineWidth = 3; % thicker line width for neutral
    colors = lines(length(fileNames)); 

    for setNum = 1:length(fileNames)
        if contains(fileNames{setNum}{1}, 'neutral') % check
            plot(time1, averagedERPs(setNum, :), 'LineWidth', neutralLineWidth, 'Color', 'black');
        else
            plot(time1, averagedERPs(setNum, :), 'LineWidth', defaultLineWidth, 'Color', colors(setNum, :));
        end
    end

% SET Y-AXIS RANGE

% Default y-axis range
defaultYMin = -6;
defaultYMax = 10;

% Find the overall minimum and maximum values across all waveforms
overallMin = min(averagedERPs(:)); % min
overallMax = max(averagedERPs(:)); % max

% Set the y-axis range dynamically
if overallMin < defaultYMin || overallMax > defaultYMax
    % If data goes beyond default range, use the actual data range for the y axis
    yMin = floor(overallMin); 
    yMax = ceil(overallMax);  
else
    % Use the default range
    yMin = defaultYMin;
    yMax = defaultYMax;
end

% Apply the y-axis limits
ylim([yMin, yMax]);

% PLOT SETTINGS
xlabel('Time (ms)');
ylabel('Amplitude (µV)');
title(['Average ERP Waveforms']);
lgd = legend('Neutral', 'Fear', 'Anger', 'Happiness', 'Disgust', 'Sadness', 'Surprise');
lgd.Location = 'northwest'; 

    % Construct a string of channel names joined by underscores
    channelsString = strjoin(channelsOfInterest, '_'); 

    % Save the figure dynamically based on the reference type and channels of interest
    outputFileName = sprintf('ERP_Waveforms_%s_%s', refType, channelsString); 
    print(gcf, outputFileName, '-dpng', '-r300'); 

    close(gcf);
end
