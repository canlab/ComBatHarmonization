
%data = randn(1000,10);
%batch = [1 1 1 1 1 2 2 2 2 2];
%mod = [1 2 1 2 1 2 1 2 1 2]';



function [bayesdata, gamma_star, delta_star] = combat(dat, batch, mod, parametric, varargin)
    if any(isnan(dat(:)))
        error('Input data contains nan entries. These will break combat. Please fix and rerun');
    end

    dat = double(dat); % there are EB algorithm convergence issues if input data is single
    [sds] = std(dat')';
    wh = find(sds==0);
    [ns,ms] = size(wh);
    if ns>0
        error('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')
    end
    if ischar(batch) % needed for sort function to work reliably
        batch = cellstr(batch);
    end
    
    ref = [];
    meanOnly = false;
    for i = 1:length(varargin)
        if ischar(varargin{i})
            switch varargin{i}
                case 'ref'                
                    ref = varargin{i+1};
                    if ~ismember(ref,batch)
                        error('Reference is not in batch list, please check inputs');
                    else
                        if ischar(ref)
                            fprintf('[combat] Harmonizing to reference batch %s\n', ref);
                        elseif isnumeric(ref) || islogical(ref)
                            fprintf('[combat] Harmonizing to reference batch %d\n', ref);
                        else
                            error('Ref argument must be type numeric or char\n');
                        end
                    end
                case 'meanOnly'
                    if varargin{i+1} == 1
                        meanOnly = true;
                    elseif varargin{i+1} == 0
                        meanOnly = false;
                    else
                        error('Did not understand argument to meanOnly');
                    end
                        
            end
        end
    end
    
    batchmod = categorical(batch);
    batchmod = dummyvar({batchmod});
	n_batch = size(batchmod,2);
	uniq_batch = unique(batch,'stable');
	fprintf('[combat] Found %d batches\n', n_batch);

	batches = cell(0);
	for i=1:n_batch
		batches{i}=find(batch == uniq_batch(i));
	end
	n_batches = cellfun(@length,batches);
	n_array = sum(n_batches);

	% Creating design matrix and removing intercept:
	design = [batchmod mod];
	intercept = ones(1,n_array)';
	wh = cellfun(@(x) isequal(x,intercept),num2cell(design,1));
	bad = find(wh==1);
	design(:,bad)=[];
            
    if ~isempty(ref)
        design(:,ismember(uniq_batch,ref)) = 1;
    end

	fprintf('[combat] Adjusting for %d covariate(s) of covariate level(s)\n',size(design,2)-size(batchmod,2))
	% Check if the design is confounded
	if rank(design)<size(design,2)
		nn = size(design,2);
	    if nn==(n_batch+1) 
	      error('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
	    end
	    if nn>(n_batch+1)
	      temp = design(:,(n_batch+1):nn);
	      if rank(temp) < size(temp,2)
	        error('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
	      else 
	        error('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')
	      end
	    end
	 end


	fprintf('[combat] Standardizing Data across features\n')
	B_hat = inv(design'*design)*design'*dat';
	%Standarization Model
    if isempty(ref)
        grand_mean = (n_batches/n_array)*B_hat(1:n_batch,:);
        var_pooled = ((dat-(design*B_hat)').^2)*repmat(1/n_array,n_array,1);
    else
        grand_mean = B_hat(ismember(uniq_batch,ref),:);
        
        ref_dat = dat(:,ismember(batch,ref));
        ref_n = n_batches(ismember(uniq_batch,ref));
        try
        var_pooled = ((ref_dat - (design(ismember(batch,ref),:)*B_hat)').^2) * ...
            repmat(1/ref_n, ref_n, 1);
        catch
            keyboard
        end
    end
	stand_mean = grand_mean'*repmat(1,1,n_array);

	if not(isempty(design))
		tmp = design;
		tmp(:,1:n_batch) = 0;
		stand_mean = stand_mean+(tmp*B_hat)';
	end	
	s_data = (dat-stand_mean)./(sqrt(var_pooled)*repmat(1,1,n_array));

	%Get regression batch effect parameters
	fprintf('[combat] Fitting L/S model and finding priors\n')
	batch_design = design(:,1:n_batch);
	gamma_hat = inv(batch_design'*batch_design)*batch_design'*s_data';
	delta_hat = [];
	for i=1:n_batch
		indices = batches{i};
        if meanOnly
            delta_hat = [delta_hat; ones(1,size(s_data,1))];
        else
            delta_hat = [delta_hat; var(s_data(:,indices)')];
        end
	end

	%Find parametric priors:
	gamma_bar = mean(gamma_hat');
	t2 = var(gamma_hat');
	delta_hat_cell = num2cell(delta_hat,2);
	a_prior=[]; b_prior=[];
	for i=1:n_batch
		a_prior=[a_prior aprior(delta_hat_cell{i})];
		b_prior=[b_prior bprior(delta_hat_cell{i})];
	end

	
	if parametric
        fprintf('[combat] Finding parametric adjustments\n')
        gamma_star =[]; delta_star=[];
        for i=1:n_batch
            if meanOnly
                gamma_star = [gamma_star; postmean(gamma_hat(i,:),gamma_bar(i), 1, 1, t2(i))];
                delta_star = [delta_star; ones(1,size(gamma_hat(i,:),2))];
            else
                indices = batches{i};
                temp = itSol(s_data(:,indices),gamma_hat(i,:),delta_hat(i,:),gamma_bar(i),t2(i),a_prior(i),b_prior(i), 0.001);
                gamma_star = [gamma_star; temp(1,:)];
                delta_star = [delta_star; temp(2,:)];
            end
        end
    end
	    
    if (1-parametric)
        gamma_star =[]; delta_star=[];
        fprintf('[combat] Finding non-parametric adjustments\n')
        for i=1:n_batch
            if meanOnly
                delta_hat(i,:) = 1;
            end
            indices = batches{i};
            temp = inteprior(s_data(:,indices),gamma_hat(i,:),delta_hat(i,:));
            gamma_star = [gamma_star; temp(1,:)];
            delta_star = [delta_star; temp(2,:)];
        end
    end
    
    if ~isempty(ref) %this will be approximately true, but this is a good fix all the same
        gamma_star(ismember(uniq_batch,ref),:) = 0;
        delta_star(ismember(uniq_batch,ref),:) = 1;
    end
	    
	fprintf('[combat] Adjusting the Data\n')
	bayesdata = s_data;
	j = 1;
	for i=1:n_batch
		indices = batches{i};
		bayesdata(:,indices) = (bayesdata(:,indices)-(batch_design(indices,:)*gamma_star)')./(sqrt(delta_star(j,:))'*repmat(1,1,n_batches(i)));
		j = j+1;
	end
	bayesdata = (bayesdata.*(sqrt(var_pooled)*repmat(1,1,n_array)))+stand_mean;

end

