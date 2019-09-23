classdef SMM < handle
    % Time/delay sensitivity map modeling
    
    properties (Constant)
        ISSURF = false;
    end
    
    properties
        folders % folder pathes
        filenames % filename of input data
        dbounds % delay bounds
        
        width % width of spatial kernel
        height % height of spatial kernel
        tmin % minimum time
        tmax % maximum time
        dmin % minimum delay
        dmax % maximum delay
        window % window length of smoothing
        threshold % threshold of noise removing
        procedureName %'z', 'x2', 'roc', 'sta'
        erodeR % radius of erosion 
        dilateR % radius of dilation
        
        models % cell array of trained logistic for each time
        kernels % linear kernels (spatiotemporal sensitivity maps) for each time
    end
    
    % Constructor
    methods
        function this = SMM(filename)
            
            % constructor
            this.width = 9;
            this.height = 9;
            this.tmin = -500;
            this.tmax = 500;
            this.dmin = 0;
            this.dmax = 200;
            this.window = 15;
            this.threshold = 0.95;
            this.procedureName = 'z';
            this.erodeR = 2;
            this.dilateR = 14;
            
            this.initFolders();
            
            if nargin < 1
                filename = fullfile(this.folders.data, '2015051115.mat');
            end
            this.initFilenames(filename);
        end
        
        function initFolders(this)
            assets = './assets';
            this.folders = struct(...
                'assets', assets, ...
                'data', fullfile(assets, 'data'), ...
                'results', fullfile(assets, 'results'));
        end
        
        function initFilenames(this, dataFilename)
            [~, name, ~] = fileparts(dataFilename);
            resultsFilename = sprintf(...
                '%s-%s-%d.mat', ... % todo: make a directory instead of renames files
                name, ...
                this.procedureName, ...
                this.window);
            
            this.filenames = struct(...
                'data', dataFilename, ...
                'results', fullfile(this.folders.results, resultsFilename));
        end
        
        function saveConfig(this, appendFlag)
            % Save configuration
            
            if ~exist('appendFlag', 'var')
                appendFlag = false;
            end
            
            fprintf('\nSave configuration (`config`) in `%s`: ', this.filenames.results);
            tic();
            
            config = struct(...
                'filename', this.filenames.data, ...
                'width', this.width, ...
                'height', this.height, ...
                'tmin', this.tmin, ...
                'tmax', this.tmax, ...
                'dmin', this.dmin, ...
                'dmax', this.dmax, ...
                'window', this.window, ...
                'threshold', this.threshold, ...
                'procedureName', this.procedureName, ...
                'erodeR', this.erodeR, ...
                'dilateR', this.dilateR);
            
            if appendFlag
                save(this.filenames.results, 'config', '-append');
            else
                save(this.filenames.results, 'config');
            end
            toc();
        end
        
        function loadConfig(this, filename)
            % Save configuration
            
            fprintf('\nLoad configuration (`config`) from `%s`: ', filename);
            tic();
            
            load(filename, 'config');
            
            this.initFilenames(config.filename);
            this.width = config.width;
            this.height = config.height;
            this.tmin = config.tmin;
            this.tmax = config.tmax;
            this.dmin = config.dmin;
            this.dmax = config.dmax;
            this.window = config.window;
            this.threshold = config.threshold;
            this.procedureName = config.procedureName;
            this.erodeR = config.erodeR;
            this.dilateR = config.dilateR;
            
            toc();
        end
        
        function loadModels(this)
            fprintf('\nGet nonlinear transforms: ');
            tic();
            
            load(this.filenames.results, 'models');
            this.models = models;
            
            toc();
        end
        
        function loadKernels(this)
            this.kernels = this.getKernels();
            this.kernels(isnan(this.kernels)) = 0;
        end
    end
    
    % Modeling (Coding)
    methods
        function fitModel(this)
            % Fit model
            
            % save config
            this.saveConfig();
            
            % require `x`, `y`
            x = this.getPredictor();
            y = this.getTrueResponse();
            
            sprintf('\nFit model:\n');
            tic();
            
            times = this.getTimes(); % time of interests
            T = numel(times); % number of times
            
            this.models = cell(T, 1);
            for t = 1:T
                fprintf('Time: %d\n', times(t));
                
                this.models{t} = fitglm(...
                    squeeze(x(:, t)), ...
                    squeeze(y(:, t)), ...
                    'linear', ...
                    'Distribution', 'binomial', ...
                    'Link', 'logit');
            end
            
            models = this.models;
            save(this.filenames.results, 'models', '-append');
            toc();
        end
        
        function y = getTrueResponse(this)
            info = who('-file', this.filenames.results);
            if ~ismember('y', info)
                this.makeTrueResponse();
            end
            
            load(this.filenames.results, 'y');
        end
        
        function makeTrueResponse(this)
            % true response (`y`)
            fprintf('\nMake true response (`y`):\n');
            tic();
            
            [~, resp] = this.getStimuliResponses();
            y = logical(resp);
            
            y = y(:, (this.dmax + 1):end);
            
            save(this.filenames.results, 'y', '-append');
            toc();
        end
        
        function x = getPredictor(this)
            
            info = who('-file', this.filenames.results);
            
            predictorName = 'x';
                
            if ~ismember(predictorName, info)
                this.makePredictor();
            end

            S = load(this.filenames.results, predictorName);
            x = S.(predictorName);
        end
        
        function makePredictor(this)
            % Make predictor/response variables
            
            % require `STIM`, `y`, `W`
            % stimuli (`STIM`), responses (`y`)
            [stim, ~] = this.getStimuliResponses();
            STIM = this.code2stim(stim);
            
            % weights/kernels (`W`)
            W = this.getKernels();
            W(isnan(W)) = 0;
            
            % predictor variables (`x`)
            fprintf('\nMake predictor (`x`):\n');
            tic();
            
            % time/delay sensitivity map
            times = this.getTimes();
            delays = this.getDelays();
            
            N = size(STIM, 1); % number of trials
            T = numel(times); % number of times
            D = numel(delays); % number of delays
            
            x = zeros(N, T);
            for t = 1:T
                fprintf('Time: %d\n', times(t));
                for i = 1:N
                    x(i, t) = sum(...
                        squeeze(W(:, :, t, :)) .* ...
                        squeeze(STIM(i, :, :, (t + D - 1):-1:t)), 'all');
                end
            end
                        
            save(this.filenames.results, 'x', '-append');
                
            toc();
        end
        
        function [stim, resp] = getStimuliResponses(this)
            info = who('-file', this.filenames.results);
            if ~ismember('stim', info) || ~ismember('resp', info)
                this.makeStimuliResponses();
            end
            
            load(this.filenames.results, 'stim', 'resp');
        end
        
        function makeStimuliResponses(this)
            % Load, align and smooth data

            % load
            fprintf('\nLoad stimuli/responses from `%s`: ', this.filenames.data);
            tic();
            
            file = load(this.filenames.data);
            STIM = double(file.stimcode);
            RESP = double(file.resp);
            tsac = double(file.tsaccade);
            
            toc();
            
            % align
            fprintf('Align stimuli/responses to saccade: ');
            tic();
            times = (this.tmin - this.dmax):this.tmax;
            
            N = numel(tsac); % number of trials
            T = numel(times); % number of times
            
            stim = zeros(N, T);
            resp = zeros(N, T);
            
            for i = 1:N
                t = times + tsac(i);
                
                stim(i, :) = STIM(i, t);
                resp(i, :) = RESP(i, t);
            end
            
            toc();
            
            % smooth
            fprintf('Smooth stimuli/responses: ');
            tic();
            
            resp = smoothdata(resp, 2, 'gaussian', this.window);
            
            save(this.filenames.results, 'stim', 'resp', '-append');
            toc();
        end
        
        function W = getKernels(this)
            
            fprintf('\nGet linear kernels: ');
            tic();
            
            info = who('-file', this.filenames.results);
            if ~ismember('W', info)
                W = this.getSMap();
%                 W = this.getMMap();
                save(this.filenames.results, 'W', '-append');
            end
            
            load(this.filenames.results, 'W');
            
            toc();
        end
    end
    
    % - Helper methods
    methods
        function stim = code2stim(this, stimcode)
            % Convert coded stimuli to boolean
            %
            % Parameters
            % ----------
            % - stimcode: integer matrix(trial,time)
            %   Coded stimuli
            %
            % Returns
            % -------
            % - stim: boolean array(trial,width,height,time)
            %   Stimulus
            
            % N: Number of trials
            % T: Number of times
            [N,T] = size(stimcode); % trial x time
            
            stim = zeros(N, this.width, this.height, T);
            
            sz = [this.width, this.height];
            for trial = 1:N
                for time = 1:T
                    index = stimcode(trial,time);
                    
                    if index
                        [x,y] = ind2sub(sz,index);
                        stim(trial,x,y,time) = 1;
                    end
                end
            end
        end
        
        function neuronName = getNeuronName(this)
            [~, neuronName, ~] = fileparts(this.filenames.data);
        end
        
        function probeIndex = getProbeIndex(this, probe)
            probeIndex = sub2ind([this.width, this.height], probe(1), probe(2));
        end
        
        function fullName = getEffectFullName(~, shortName)
            fullName = '';
            switch shortName
                case 'ss'
                    fullName = 'Saccadic suppression';
                case 'ff'
                    fullName = 'FF-remapping';
                case 'st'
                    fullName = 'ST-remapping';
                case 'pa'
                    fullName = 'Persistent activity';
            end
        end
        
        function effect = getEffect(this, effectName)
            neuronName = this.getNeuronName();
            model = load(fullfile('./assets/models', [neuronName '.mat']));
            effect = model.effects.(effectName);
        end
    end
    
    % Mapping
    methods
        function smap = getSMap(this)
            info = who('-file', this.filenames.results);
            if ~ismember('smap', info)
                this.makeSMap();
            end
            
            load(this.filenames.results, 'smap');
        end
        
        function makeSMap(this)
            % Make time/delay sensitivity map
            
            % require `stim`, `resp`
            [stim, resp] = this.getStimuliResponses();
            
            fprintf('\nMake sensitivity map (`smap`):\n');
            tic();
            
            times = this.getTimes();
            delays = this.getDelays();
            
            tnum = numel(times);
            dnum = numel(delays);
            
            switch this.procedureName
                case 'z'
                    procedure = @SMM.ptestz;
                case 'x2'
                    procedure = @SMM.ptestx2;
                case 'roc'
                    procedure = @SMM.roc;
                case 'sta'
                    procedure = @SMM.sta;
            end
            
            % probe
            sz = [this.width, this.height];
            smap = zeros(this.width, this.height, tnum, dnum);
            for x = 1:this.width
                for y = 1:this.height
                    fprintf('Probe: (%d, %d)\n', x, y);
                    probe = sub2ind(sz, x, y);
                    
                    map = nan(tnum, dnum);
                    for it = 1:tnum % index of time
                        t = it + this.dmax;
                        for id = 1:dnum % index of delay
                            d = delays(id);

                            idx = stim(:, t - d) == probe;

                            pref = resp(idx, t);
                            npref = resp(~idx, t);
                            
                            if ~isempty(pref) && ~isempty(npref)
                                s = procedure(pref, npref);

                                map(it, id) = s;
                            end
%                              data(it, id) = struct('pref', pref, 'npref', npref', 'p', p);
                        end
                    end

%                     save('data.mat', 'data');
                    smap(x, y, :, :) = map;
                end
            end
            
            save(this.filenames.results, 'smap', '-append');
            toc();
        end
        
        function bmap = getBMap(this)
            load(this.filenames.results, 'config');
            if this.threshold ~= config.threshold || ...
               this.erodeR ~= config.erodeR || ...
               this.dilateR ~= config.dilateR
                this.makeBMap();
                this.saveConfig(true);
            end
            
            info = who('-file', this.filenames.results);
            if ~ismember('bmap', info)
                this.makeBMap();
            end
            
            load(this.filenames.results, 'bmap');
        end
        
        function makeBMap(this)
            % Make boolean (responsive times) map
            
            % require `smap`
            smap = this.getSMap();
            
            % boolean map (`bmap`)
            fprintf('\nMake responsive times map (`bmap`):\n');
            tic();
            
            bmap = abs(smap) >= this.threshold;
            
            % errosion/dilation structuring element
            erodeSE = strel('disk', this.erodeR);
            dilateSE = strel('disk', this.dilateR);
            for x = 1:this.width
                for y = 1:this.height
                    fprintf('Probe: (%d, %d)\n', x, y);
                    
                    bmap(x, y, :, :) = imerode(squeeze(bmap(x, y, :, :)), erodeSE);
                    bmap(x, y, :, :) = imdilate(squeeze(bmap(x, y, :, :)), dilateSE);
                end
            end
            
            save(this.filenames.results, 'bmap', '-append');
            toc();
        end
    
        function mmap = getMMap(this)
            info = who('-file', this.filenames.results);
            if ~ismember('mmap', info)
                this.makeMMap();
            end
            
            load(this.filenames.results, 'mmap');
        end
        
        function makeMMap(this)
            % Make masked (sensitivity for responsive times) map
            
            % require `smap`, `bmap`
            smap = this.getSMap();
            bmap = this.getBMap();
            
            % masked map (`mmap`)
            fprintf('\nMake sensitivity for responsitve times map (`mmap`):\n');
            tic();
            
            mmap = zeros(size(smap));
            mmap(bmap) = smap(bmap);
            
            save(this.filenames.results, 'mmap', '-append');
            toc();
        end
    
        function makeDBounds(app)
            % Make masked map
            
            app.dbounds = zeros(app.width, app.height, size(app.bmap, 3), 3);
            for x = 1:app.width
                for y = 1:app.height
                    map = squeeze(app.bmap(x, y, :, :));
                    
                    T = size(map, 1); % number of times
                    bounds = nan(T, 3); % [Time, Min, Max, Duration]
                    
                    for t = 1:T
                        dfirst = find(map(t, :), 1, 'first');
                        if isempty(dfirst)
                            continue;
                        end

                        dlast = find(map(t, :), 1, 'last');
                        if isempty(dlast)
                            continue;
                        end

                        bounds(t, :) = [dfirst, dlast, dlast - dfirst];
                    end
                    
                    app.dbounds(x, y, :, :) = bounds;
                end
            end
        end
        
        function times = getTimes(this)
            times = this.tmin:this.tmax;
        end
        
        function delays = getDelays(this)
            delays = this.dmin:this.dmax;
        end
        
        function skrn = getSKrn(this)
            modelName = sprintf('./assets/models/%s.mat', this.getNeuronName());
            load(modelName, 'skrn');
        end
    end
    
    % - Procedures
    methods (Static)
        function s = ptestz(pref, npref)
            % Two-proportion z-test, pooled for H0: p1=p2
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            x1 = sum(pref);
            x2 = sum(npref);
            n1 = numel(pref);
            n2 = numel(npref);
            
            p1 = x1 / n1;
            p2 = x2 / n2;
            
            % pooled estimate of proportion
            p = (x1 + x2) / (n1 + n2);
            z = (p1 - p2) / sqrt(p * (1 - p) * ((1 / n1) + (1 / n2)));
            
%             s = 2 * normcdf(abs(z)) - 1;
            s = normcdf(z) - normcdf(-z);
        end
    
        function s = ptestx2(pref, npref)
            % Two-proportion chi-squared test for goodness of fit
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            x1 = sum(pref);
            x2 = sum(npref);
            n1 = numel(pref);
            n2 = numel(npref);
            
            % pooled estimate of proportion
            p = (x1 + x2) / (n1 + n2);
            
            % expected counts
            x1_ = n1 * p;
            x2_ = n2 * p;
            
            observed = [x1, n1 - x1, x2, n2 - x2];
            expected = [x1_, n1 - x1_, x2_, n2 - x2_];
            
            chi2stat = sum((observed - expected) .^ 2 ./ expected);
            s = chi2cdf(chi2stat, 1);
        end
    
        function s = roc(pref, npref)
            % Receiver operating characteristic (roc)
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            th = unique([pref; npref]); % thresholds
            n = numel(th); % number of unique thresholds
            
            FPR = zeros(n, 1); % false positive rate
            TPR = zeros(n, 1); % true positive rate
            
            P = numel(pref); % condition positive
            N = numel(npref); % condition negative
            
            for i = 1:n
                TPR(i) = numel(find(pref >= th(i))) / P; % true positive rate
                FPR(i) = numel(find(npref >= th(i))) / N; % false positive rate
            end
        
            auc = -trapz(FPR, TPR); % area under curve
            
            % s = 2 * abs(auc - 0.5);
            s = auc - 0.5;
        end
    
        function s = sta(pref, npref)
            % Spike-triggered average (sta)
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            x1 = sum(pref);
            x2 = sum(npref);
            
            s = x1 / (x1 + x2);
        end
    end
    
    % Performance
    methods
        function perfs = getPerfs(this, probe)
            % Get performance measures
            
            % S-Kernel
            skrn = this.getSKrn();
            skrn = squeeze(skrn(probe(1), probe(2), (1 + 40):(end - 40), :)); % 1081 -> 1001
            skrn = abs(skrn);
            skrn = imbinarize(skrn);
            skrn = skrn(:);
            skrn_ = ~skrn;
            
            % Sensitivity Map
            smap = this.getSMap();
            smap = squeeze(smap(probe(1), probe(2), :, 1:(end - 1))); % 151 -> 150
            smap = abs(smap);
            smap = imbinarize(smap);
            smap(isnan(smap)) = 0;
            smap = smap(:);
            smap_ = ~smap;
            
            % condition positive (P)
            P = sum(skrn);
            % condition negative (N)
            N = sum(skrn_);
            % true positive (TP)
            TP = sum(skrn & smap);
            % false positive (FP)
            FP = sum(skrn_ & smap);
            % true negative (TN)
            TN = sum(skrn_ & smap_);
            % false negative (FN)
            FN = sum(skrn & smap_);
            % true positive rate (TPR)
            TPR = TP / P;
            % true negative rate (TNR)
            TNR = TN / N;
            % Informedness or Bookmaker Informedness (BM)
            BM = TPR + TNR - 1;
            
            perfs = struct(...
                'P', P, ...
                'N', N, ...
                'TP', TP, ...
                'FP', FP, ...
                'TN', TN, ...
                'FN', FN, ...
                'TPR', TPR, ...
                'TNR', TNR, ...
                'BM', BM);
        end
    end
    
    % Plotting
    methods
        function plotMap(this, probe, type)
            % Plot time delay sensitivity map
            %
            % Parameters
            % ----------
            % - isBoolean: boolean
            %   Is boolean map or not
            % - map: matrix
            %   (time x delay) sensitivity map
            
            SMM.createFigure('Spatiotemporal sensitivity map');
            
            switch type
                case 'skrn'
                    skrn = this.getSKrn();
                    map = squeeze(skrn(probe(1), probe(2), (1 + 40):(end - 40), :)); % 1081 -> 1001
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    % caxis([-1, 1]);
                    % c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c = colorbar();
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'S-Kernel';
                case 'sensitivity'
                    smap = this.getSMap();
                    map = squeeze(smap(probe(1), probe(2), :, :));
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    caxis([-1, 1]);
                    c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity';
                case 'boolean'
                    bmap = this.getBMap();
                    map = squeeze(bmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(double(map'));
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    titleTxt = 'Map of Responsive Times';
                case 'masking'
                    mmap = this.getMMap();
                    map = squeeze(mmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    caxis([-1, 1]);
                    c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity for Responsive Times';
                case 'dbounds' % delay bounds
                    plot(...
                        smoothdata(this.dbounds, 1, 'gaussian', this.window), ...
                        'LineWidth', 6);
                    lgd = legend({'Begin', 'End', 'Length'});
                    title(lgd, 'Delay');
                    titleTxt = 'Bounds of Temporal Kernels';
            end
            
            switch type
                case 'skrn'
                    title(titleTxt);
                otherwise
                    title({titleTxt, this.getTitleInfo(probe)});
            end
            
            this.setTimeAxis();
            this.setDelayAxis();
            
            SMM.setFontSize();
            axis('tight');
        end
        
        function plotMapBW(this, probe, type)
            % Plot time delay sensitivity map
            %
            % Parameters
            % ----------
            % - isBoolean: boolean
            %   Is boolean map or not
            % - map: matrix
            %   (time x delay) sensitivity map
            
            SMM.createFigure('Spatiotemporal sensitivity map');
            
            switch type
                case 'skrn'
                    skrn = this.getSKrn();
                    map = squeeze(skrn(probe(1), probe(2), (1 + 40):(end - 40), :)); % 1081 -> 1001
                    map = abs(map);
                    
                    map = imbinarize(map);
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    caxis([0, 1]);
                    c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'S-Kernel';
                case 'sensitivity'
                    smap = this.getSMap();
                    map = squeeze(smap(probe(1), probe(2), :, :));
                    
                    map = abs(map);
                    map = imbinarize(map);
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    caxis([0, 1]);
                    c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity';
                case 'boolean'
                    bmap = this.getBMap();
                    map = squeeze(bmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(double(map'));
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    titleTxt = 'Map of Responsive Times';
                case 'masking'
                    mmap = this.getMMap();
                    map = squeeze(mmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    caxis([-1, 1]);
                    c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity for Responsive Times';
                case 'dbounds' % delay bounds
                    plot(...
                        smoothdata(this.dbounds, 1, 'gaussian', this.window), ...
                        'LineWidth', 6);
                    lgd = legend({'Begin', 'End', 'Length'});
                    title(lgd, 'Delay');
                    titleTxt = 'Bounds of Temporal Kernels';
            end
            
            switch type
                case 'skrn'
                    title(titleTxt);
                otherwise
                    title({titleTxt, this.getTitleInfo(probe)});
            end
            
            this.setTimeAxis();
            this.setDelayAxis();
            
            SMM.setFontSize();
            axis('tight');
        end
        
        function plotMapAll(this, type)
            SMM.createFigure('Spatiotemporal sensitivity map');
            
            switch type
                case 'skrn'
                    maps = this.getSKrn();
                case 'sensitivity'
                    maps = this.getSMap();
                case 'boolean'
                    maps = this.getBMap();
                case 'masking'
                    maps = this.getMMap();
            end
            
            for x = 1:this.width
                for y = 1:this.height
                    ax = subplot(this.width, this.height, this.getIndex(x, y));
                    
                    map = squeeze(maps(x, y, :, :));
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    switch type
                        case {'sensitivity', 'masking'}
                            caxis(ax, [-1, 1]);
                            colormap(ax, 'jet');
                        case 'boolean'
                            caxis(ax, [0, 1]);
                            colormap(ax, 'gray');
                        case 'dbounds'
                            plot(ax, ...
                                smoothdata(squeeze(this.dbounds(x, y, :, :)), 1, 'gaussian', this.window), ...
                                'LineWidth', 6);
                    end
                    
                    xticks(ax, []);
                    yticks(ax, []);
                    axis(ax, 'tight');
                    box(ax, 'on');
                    title(sprintf('(%d, %d)', x, y));
                end
            end
            
            % first axes
            ax = subplot(this.width, this.height, this.getIndex(1, 1));
            suptitle(this.getTitleInfo());
            this.setTimeAxis();
            this.setDelayAxis();
            
            switch type
                case {'sensitivity', 'masking'}
                    colorbar(ax, ...
                        'Location', 'east', ...
                        'Limits', [-1, 1], ...
                        'XTick', [-1, 0, 1]);
                case 'dbounds'
                    lgd = legend({'Begin', 'End', 'Length'});
                    title(lgd, 'Delay');
            end
            
%             name = this.getNeuronName();
%             saveas(gcf, sprintf('%s-%s.png', name, type));
        end
        
        function playMapAll(this, type)
            switch type
                case 'skrn'
                    skrn = this.getSKrn();
                    I = max(skrn, [], 4); % todo: how to figure out minimum values?!
                case 'sensitivity'
                    smap = this.getSMap();
                    I = max(smap, [], 4); % todo: how to figure out minimum values?!
                case 'boolean'
                    bmap = this.getBMap();
                    I = max(bmap, [], 4);
                    % plot `Number of sources`
                    C = squeeze(sum(I, [1, 2]));
                    SMM.createFigure('Number of sources');
                    
                    % bar(C);
                    stairs(C, 'LineWidth', 4);

                    title(this.getTitleInfo());
                    
                    this.setTimeAxis();
                    
                    ylabel('Number of sources');
                    yticks(1:max(C));
                    
                    SMM.setFontSize();
                    
                    axis('tight');
                case 'masking'
                    mmap = this.getMMap();
                    I = max(mmap, [], 4);
            end
            
            I = permute(I, [2, 1, 3]);
            I = flipud(I);
            implay(I);
        end
        
        function index = getIndex(this, x, y)
            r = this.height - y + 1;
            c = x;
            index = (r - 1) * this.width + c;
        end
    end
    
    methods
        function titleInfo = getTitleInfo(this, probe)
            % Neuron id
            [~, neuronTxt, ~] = fileparts(this.filenames.data);
            
            % Probe location
            if exist('probe', 'var')
                probeTxt = sprintf('(%d, %d)', probe(1), probe(2));
            else
                probeTxt = 'All';
            end
            
            % Procedure name
            switch this.procedureName
                case 'z'
                    procedureTxt = 'Pooled z-test';
                case 'x2'
                    procedureTxt = 'Chi-squared test';
                case 'roc'
                    procedureTxt = 'ROC';
                case 'sta'
                    procedureTxt = 'STA';
            end
            
            % Title info
            titleInfo = sprintf(...
                'Neuron: ''%s'', Probe: %s, Procedure: ''%s''', ...
                neuronTxt, probeTxt, procedureTxt);
        end
        
        function setTimeAxis(this)
            times = this.getTimes();
            
            xlabel('Time form saccade onset (ms)');
            
            T = numel(times);
            tidx = [1, ceil(T / 2), T];
            xticks(tidx);
            
            xticklabels(string(times(tidx)));
        end
        
        function setDelayAxis(this)
            delays = this.getDelays();
            ylabel('Delay (ms)');
            yticks(delays(1):50:delays(end));
        end
    end
    
    methods (Static)
        function h = createFigure(name)
            % Create `full screen` figure
            %
            % Parameters
            % ----------
            % - name: string
            %   Name of figure
            %
            % Return
            % - h: matlab.ui.Figure
            %   Handle of created figure
            
            h = figure(...
                'Name', name, ...
                'Color', 'white', ...
                'NumberTitle', 'off', ...
                'Units', 'normalized', ...
                'OuterPosition', [0, 0, 1, 1] ...
            );
        end
        
        function setFontSize()
            set(gca(), 'FontSize', 18);
        end
    end
    
    % Main
    methods (Static)
        function main()
            close('all');
            clc();
            
            % copy command widnow to `log.txt` file
            diary('log.txt');
            
            fprintf('Sensitivity Map Modeling (SMM): %s\n', datetime());
            mainTimer = tic();
            
            % crete model
            filename = './assets/data/2015051115.mat';
            smm = SMM(filename);
            
            % fit model
            smm.fitModel();
            
            % results
            probe = [7, 6];
            
            smm.plotMap(probe, 'sensitivity');
            smm.plotMap(probe, 'skrn');
            
            smm.plotMapAll('sensitivity');
            smm.plotMapAll('skrn');

            toc(mainTimer);
            
            diary('off');
            % to see log file
            % >>> type('log.txt');
        end
    end
end

