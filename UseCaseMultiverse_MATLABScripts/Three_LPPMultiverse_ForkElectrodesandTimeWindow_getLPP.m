clear all, close all, clc;
%
mainpath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis\fork1-2';
peakspath = 'C:\Users\short\Documents\ERPSamplingStudy\Analysis';
storepath = [mainpath 'fork3-4\'];

win_fork = {500,200; 500,300; 600,200; 600,300; 600,600; 700,200; 700,300;...
    700,600; 450,100; 'GAV', 400; 'SAV', 400}; % {window center, window width}; e.g.  {500,200} is 400 to 600 ms
elec_fork = {{'CP1', 'CP2', 'Pz', 'P3', 'P4'}; {'P3', 'P4', 'CP1', 'CP2'}; {'P3','Pz','P4'};...
    {'Fz','Cz', 'Pz'}; {'CP1', 'CP2'}; {'Cz'}; {'Pz'};{'around_peak'}};
peakelecs = {'Fpz', 'Fp1', 'Fp2', 'AF7', 'AF8'; 'Fz', 'F3', 'F4', 'FC1', 'FC2';...
    'Cz', 'C3', 'C4', 'CP1', 'CP2'; 'Pz', 'P3', 'P4', 'CP1', 'CP2'; ...
    'Oz', 'O1', 'O2', 'PO7', 'PO8'; 'Iz', 'O1', 'O2', 'PO9', 'PO10'}; % first electrode is midline electrode, the following 4 are the 4 electrodes around this midline electrode; relevant for the 'around peak' electrode fork

cd(peakspath)
load('SAVpeaks.mat')
load('GAVpeaks.mat')

% Extract a list of all participants x conditions x pipelines without the '.set' ending from the files in the storage folder created in the preprocessing script

files = {};
files2 = {dir([mainpath]).name};
for s = 3:length(dir([mainpath]))
    filename = files2{s};
    files{s-2} = filename(1:end-4);
end

% create a table with the mean LPP for the different subjects, paths & conditions
LPP_values = {};

eeglab nogui;
for f = 6801:length(files)
    EEG = pop_loadset('filename',[files{f} '.set'],'filepath',[mainpath]);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0 );
    for w = 1:size(win_fork,1) % window fork
        if strcmp(win_fork{w,1}, 'SAV') == 1 % defining the timewindow
            fileTmp = files{f};
            fileTmp = strrep(fileTmp, 'anger', '');
            fileTmp = strrep(fileTmp, 'disgust', '');
            fileTmp = strrep(fileTmp, 'fear', '');
            fileTmp = strrep(fileTmp, 'happiness', '');
            fileTmp = strrep(fileTmp, 'neutral', '');
            fileTmp = strrep(fileTmp, 'sadness', '');
            fileTmp = strrep(fileTmp, 'surprise', '');
            si = find(strcmp(SAVpeaks(:,1),[fileTmp(9:17) fileTmp(end-8:end)])); % finding the correct participant
            trange = [SAVpeaks{si,2} - win_fork{w,2}/2 SAVpeaks{si,2}+win_fork{w,2}/2]; % = SAV peak time of this participant +- 200 ms
        elseif strcmp(win_fork{w,1}, 'GAV') == 1
            fileTmp = files{f};
            fileTmp = strrep(fileTmp, 'anger', '');
            fileTmp = strrep(fileTmp, 'disgust', '');
            fileTmp = strrep(fileTmp, 'fear', '');
            fileTmp = strrep(fileTmp, 'happiness', '');
            fileTmp = strrep(fileTmp, 'neutral', '');
            fileTmp = strrep(fileTmp, 'sadness', '');
            fileTmp = strrep(fileTmp, 'surprise', '');
            si = find(strcmp(GAVpeaks(:,1),fileTmp(end-8:end)));
            trange1 = GAVpeaks{si,2}-win_fork{w,2}/2;
            trange2 = GAVpeaks{si,2}+win_fork{w,2}/2;
            trange =[trange1 trange2];
        else
            trange = [win_fork{w,1}-win_fork{w,2}/2 win_fork{w,1}+win_fork{w,2}/2];
        end
        Dpoints = round(eeg_lat2point(trange/1000, [1 1], EEG.srate, [EEG.xmin EEG.xmax]));
        for ef = 1:length(elec_fork) % electrodes fork
            chans = [];
            for ch = 1:length(elec_fork{ef})
                if strcmp(elec_fork{ef},'around_peak') == 1 % the fork '4 electrodes around midline peak' needs to be coded separately
                    mci = find(strcmp(GAVpeaks{1,4},peakelecs(:,1)));
                    for apch = 1:size(peakelecs,2)-1
                        chans(end+1) = find(strcmpi(peakelecs{mci,apch},{EEG.chanlocs.labels}));
                    end
                else
                    chans(end+1) = find(strcmpi(elec_fork{ef}{ch},{EEG.chanlocs.labels}));
                end
            end
            if isnumeric(win_fork{w}) == 1
                win_m = num2str(win_fork{w,1});
            else
                win_m = win_fork{w,1};
            end
            LPP_values(end+1,1) = {[files{f}, win_m,...
                num2str(win_fork{w,2}), elec_fork{ef}{:}]}; % saving the participant x condition x pipeline name
            ERP(:,:) = mean(mean(EEG.data(chans,:,:),3),1);
            ERP_maintimewindow = mean(ERP(:,Dpoints(1):Dpoints(2)),2);
            LPP_values(end,2) = {ERP_maintimewindow}; % saving the LPP value: mean amplitude
        end
    end
    save([files{f} 'LPP_values.mat'], 'LPP_values') % to be imported into the next script
    LPP_values = {};
end
