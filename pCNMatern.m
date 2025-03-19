% Copyright (c) 2025 Matteo Giordano
%
% Codes accompanying the article "Bayesian nonparametric inference in elliptic 
% PDEs: Convergence rates and implementation" by Matteo Giordano

%%
% Bayesian nonparametric inference for the diffusivity f:O->[f_min,+infty) 
% with Matérn process priors via the pCN algorithm
%
% Requires output of GenerateObservations.m (including f_0, observations 
% and geometry), and the file K_mat.m (Matérn kernel)

%%
% Mesh for discretisation of parameter space via piecewise linear functions

model_prior = createpde(); 
geometryFromMesh(model_prior,tnodes,telements);
generateMesh(model_prior,'Hmax',0.075);
mesh_nodes_prior=model_prior.Mesh.Nodes;
mesh_nodes_num_prior=size(mesh_nodes_prior); 
mesh_nodes_num_prior=mesh_nodes_num_prior(2);
    % discretised parameter space dimension
mesh_elements_prior = model_prior.Mesh.Elements;
mesh_elements_num_prior = size(mesh_elements_prior); 
mesh_elements_num_prior = mesh_elements_num_prior(2); 
[~,mesh_elements_area_prior] = area(model_prior.Mesh);

% Compute barycenters of triangular mesh elements
barycenters_prior = zeros(2,mesh_elements_num_prior);
for i=1:mesh_elements_num_prior
    barycenters_prior(:,i) = mean(mesh_nodes_prior(:,mesh_elements_prior(1:3,i)),2);
end

f0_mesh_prior = f0(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:));
f0_bary_prior = f0(barycenters_prior(1,:),barycenters_prior(2,:));
F0_mesh_prior = log(f0_mesh_prior);
F0_bary_prior=log(f0_bary_prior);

%%
% pCN initialisation

% pCN initialisation at 0
f_init=@(location,state) exp(0*location.x);
F_init_mesh_prior = 0*mesh_nodes_prior(1,:); 

% pCN initialisation at f0
%f_init=@(location,state) f0(location.x,location.y);
%F_init_mesh_prior = F0_mesh_prior;

% Solve elliptic PDE for initial pCN state
specifyCoefficients(model,'m',0,'d',0,'c',f_init,'a',0,'f',s0_fun);
results_current = solvepde(model);
u_current = results_current.NodalSolution; 

% Compute likelihood for initialisation point and compare it to likelihood
% of f_0
loglik_init = -sum( (observations-u_current(rand_index)).^2 )/(2*sigma^2);
disp(['Log-likelihood of pCN initialisation = ', num2str(loglik_init)])
disp(['Log-likelihood of f_0 = ', num2str(loglik0)])

%%
% Prior covariance matrix for prior draws

M = mesh_nodes_num_prior;
% Compute covariance matrix for Matérn process on mesh for discretisation
prior_regularity = 3; l = .25; % hyper-parameters in Mtern kernel
Cov_matr = zeros(M,M);
for i=1:M
    for j=i:M
        Cov_matr(i,j) = K_mat(mesh_nodes_prior(:,i),mesh_nodes_prior(:,j),prior_regularity,l);
        Cov_matr(j,i) = Cov_matr(i,j);
    end
end
scaling = n^(-2/(2*prior_regularity+2+4));
    % n dependent scaling of covariance
Cov_matr = scaling^2*Cov_matr;

rng(1)
% Sample and plot a prior draw
prior_draw = mvnrnd(zeros(M,1),Cov_matr,1)';
f_rand = exp(prior_draw);
figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model_prior,'XYData',f_rand,'ColorMap',jet);
title('f=e^F, F\sim\Pi','FontSize',20)
colorbar('FontSize',20)
xlabel('x','FontSize', 20)
ylabel('y', 'FontSize', 20)
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
crameri vik

%%
% pCN algorithm

% Parameters for pCN algorithms
delta=.00025; 
    % smaller delta imply shorter moves for the pCN proposal
MCMC_length=250; 
    % number of pCN draws

% Trackers
accept_count = zeros(1,MCMC_length); 
    % initialise vector % to keep track of acceptance steps
alphas = ones(1,MCMC_length); 
    % initialise vector to keep  track of acceptance probability
unifs = rand(1,MCMC_length); 
    % i.i.d. Un(0,1) random variables for Metropolis-Hastings updates
F_MCMC = zeros(M,MCMC_length); 
    % initialise marix to store the MCMC chain of functions
F_MCMC(:,1)=F_init_mesh_prior; 
    % set pCN initialisation point
loglik_MCMC = zeros(1,MCMC_length); 
    % initialise vector to keep track of the loglikelihood of the pCN chain
loglik_MCMC(1) = loglik_init; 
    % set initial loglikelihood for initialisation point
loglik_current = loglik_MCMC(1);
F_current_num = F_MCMC(:,1);

tic

for MCMC_index=2:MCMC_length
    fprintf('pCN step n. %d\n',MCMC_index);
    
    % Construct pCN proposal
    prior_draw = mvnrnd(zeros(M,1),Cov_matr,1)';
    F_proposal_num = sqrt(1-2*delta)*F_current_num + sqrt(2*delta)*prior_draw;
        
    % Define proposal diffusivity function to pass to PDE solver
    F_proposal=@(location,state) griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),F_proposal_num,location.x,location.y);
    f_proposal=@(location,state) exp(F_proposal(location,state));

    % Solve PDE with coefficient equal to the proposal
    specifyCoefficients(model,'m',0,'d',0,'c',f_proposal,'a',0,'f',s0_fun);
    results_proposal = solvepde(model);
    u_proposal = results_proposal.NodalSolution; 

    % Compute acceptance probability
    loglik_proposal = -sum( (observations-u_proposal(rand_index)).^2 )/(2*sigma^2);
    alpha=exp(loglik_proposal-loglik_current);
    alphas(MCMC_index)=alpha;

    if unifs(MCMC_index)<alpha % if verified, accept proposal
        F_current_num = F_proposal_num;
        loglik_current = loglik_proposal;
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1)+1;
    else
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1);
    end
    F_MCMC(:,MCMC_index) = F_current_num;
    loglik_MCMC(MCMC_index) = loglik_current;
end

toc

%%
% Acceptance and loglikelihood along the MCMC chain

n_accept_steps=accept_count(1,MCMC_length);
accept_ratio=accept_count./(1:MCMC_length);

figure()
axes('FontSize', 20, 'NextPlot','add')
plot(accept_ratio,'LineWidth',2)
title('Acceptance ratio','FontSize',20)
xlabel('MCMC step','FontSize', 20)

figure()
axes('FontSize', 20, 'NextPlot','add')
plot(loglik_MCMC,'LineWidth',2)
%title('Loglikelihood along the MCMC chain','FontSize',20)
xlabel('MCMC step','FontSize', 20)
yline(loglik0,'r','LineWidth',2)
legend('Loglikelihood along MCMC chain','Loglikelihood of f_0','Fontsize',20)

%%
% MCMC average and estimation error

% Compute MCMC average
burnin=min(5000,MCMC_length/2);
F_mean_mesh_prior = mean(F_MCMC(:,burnin+1:MCMC_length),2);
f_mean_mesh_prior=exp(F_mean_mesh_prior);

% Compute L^2-estimation error
f_mean_bary_prior = griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),f_mean_mesh_prior,barycenters_prior(1,:),barycenters_prior(2,:));
estim_error = sqrt(sum((f0_bary_prior-f_mean_bary_prior).^2.*mesh_elements_area_prior));
disp(['L^2 approximation estimation error = ', num2str(estim_error)])
disp(['Relative error = ', num2str(estim_error/f0_norm)])

% Plot f_0 and posterior mean estimate
figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model,'XYData',f0_mesh,'ColorMap',jet)
clim([min(f0_mesh_prior),max(f0_mesh_prior)])
colorbar('Fontsize',20)
title('True conductivity f_0','FontSize',20);
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
xlabel('x', 'FontSize', 20);
ylabel('y', 'FontSize', 20);
crameri vik

figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model_prior,'XYData',f_mean_mesh_prior,'ColorMap',jet)
clim([min(f0_mesh_prior),max(f0_mesh_prior)])
colorbar('Fontsize',20)
%title(['n = ',num2str(n),'; estimation error = ',num2str(estim_error)],'FontSize',20);
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
xlabel('x', 'FontSize', 20);
ylabel('y', 'FontSize', 20);
crameri vik
