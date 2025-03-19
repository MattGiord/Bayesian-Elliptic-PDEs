% Copyright (c) 2025 Matteo Giordano
%
% Codes accompanying the article "Bayesian nonparametric inference in elliptic 
% PDEs: Convergence rates and implementation" by Matteo Giordano

%%
% Bayesian nonparametric inference for the diffusivity f:O->[f_min,+infty) 
% with truncated Gaussian series priors on the Dirichlet Laplacian eigebasis
% via the pCN algorithm
%
% Requires output of GenerateObservations.m (including f_0, observations 
% and geometry)

%%
% Mesh for computation of the Dirichlet-Laplacian eigenpairs

model_prior = createpde(); 
geometryFromMesh(model_prior,tnodes,telements);
generateMesh(model_prior,'Hmax',0.075);
mesh_nodes_prior=model_prior.Mesh.Nodes;
mesh_nodes_num_prior=size(mesh_nodes_prior); 
mesh_nodes_num_prior=mesh_nodes_num_prior(2);
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
% Solve elliptic eigenvalue problem for the Dirichlet-Laplacian

tic

% Specity homogeneous Dirichlet boundary conditions
applyBoundaryCondition(model_prior,'dirichlet','Edge', ...
    1:model.Geometry.NumEdges,'u',0); 

% Specify coefficients for eigenvalue equation
specifyCoefficients(model_prior,'m',0,'d',1,'c',1,'a',0,'f',0);
range = [-1,500]; 
    % range of search for eigenvalues
results = solvepdeeig(model_prior,range); 
    % solve eigenvalue equation
lambdas_basis = results.Eigenvalues; 
    % extract eigenvalues
J_basis = length(lambdas_basis); 
    % number of eigenvalues (dimension of discretised parameter space)
e_basis = results.Eigenvectors; 
    % extract eigenfunctions

toc

%figure() 
%subplot(1,3,1)
%axes('FontSize', 20, 'NextPlot','add')
%pdeplot(model_prior,'XYData',e_basis(:,1),'ColorMap',jet);
%clim([min(e_basis(:,J_basis)),max(e_basis(:,J_basis))])
%colorbar('Fontsize',20)
%title('e_0','FontSize',20)
%xlabel('x','FontSize', 20)
%ylabel('y', 'FontSize', 20)
%xticks([-1,-.5,0,.5,1])
%yticks([-1,-.5,0,.5,1])
%crameri roma

%figure() 
%subplot(1,3,2)
%axes('FontSize', 20, 'NextPlot','add')
%pdeplot(model_prior,'XYData',e_basis(:,2),'ColorMap',jet); 
%clim([min(e_basis(:,J_basis)),max(e_basis(:,J_basis))])
%title('e_1','FontSize',20)
%xlabel('x','FontSize', 20)
%ylabel('y', 'FontSize', 20)
%xticks([-1,-.5,0,.5,1])
%yticks([-1,-.5,0,.5,1])
%crameri roma

%figure() 
%subplot(1,3,3)
%axes('FontSize', 20, 'NextPlot','add')
%pdeplot(model_prior,'XYData',e_basis(:,J_basis),'ColorMap',jet); 
%clim([min(e_basis(:,J_basis)),max(e_basis(:,J_basis))])
%title('e_J','FontSize',20)
%xlabel('x','FontSize', 20)
%ylabel('y', 'FontSize', 20)
%xticks([-1,-.5,0,.5,1])
%yticks([-1,-.5,0,.5,1])
%crameri roma

% Plot the eigenvalues
%figure()
%axes('FontSize', 20, 'NextPlot','add')
%plot(lambdas_basis,'*','Linewidth',1)
%xlabel('j', 'FontSize', 20);
%ylabel('\lambda_j', 'FontSize', 20);
%legend('\lambda_j=O(j)','FontSize',25)

%%
% Projection of F_0 onto the Dirichlet-Laplacian eigenbasis

F0_coeff=zeros(J_basis,1); 
    % initialises vector to store the Fourier coefficients of F0 in the 
    % Dirichlet-Laplacian eigenbasis

for j=1:J_basis
    ej_basis_interp=scatteredInterpolant(mesh_nodes_prior(1,:)',mesh_nodes_prior(2,:)',e_basis(:,j));
    ej_basis_bary_prior=ej_basis_interp(barycenters_prior(1,:),barycenters_prior(2,:));
    F0_coeff(j)=sum(mesh_elements_area_prior.*F0_bary_prior.*ej_basis_bary_prior);
end

F0_proj=zeros(1,mesh_nodes_num_prior);
for j=1:J_basis
    F0_proj = F0_proj+F0_coeff(j)*e_basis(:,j)';
end

figure()
subplot(1,2,1)
pdeplot(model_prior,'XYData',F0_mesh_prior,'ColorMap',jet)
title('True F_0','FontSize',20)
colorbar('Fontsize',20)
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])

subplot(1,2,2)
pdeplot(model_prior,'XYData',F0_proj,'ColorMap',jet)
title('Projection of F_0','FontSize',20)
colorbar('Fontsize',20)
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])

% L^2 norm of F0
F0_norm=norm(F0_coeff);
disp(['L^2 norm of F_0 = ', num2str(F0_norm)])

% Approximate L^2 distance between F_0 and projection
F0_proj_bary_prior = griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),F0_proj,...
    barycenters_prior(1,:),barycenters_prior(2,:));
approx_error = sqrt(sum((F0_bary_prior-F0_proj_bary_prior).^2.*mesh_elements_area_prior));
disp(['L^2 approximation error via projection = ', num2str(approx_error)])
disp(['Relative error = ', num2str(approx_error/F0_norm)])

%%
% Prior covariance matrix for Gaussian series prior draws

prior_regularity=3/2; 
prior_cov=diag(lambdas_basis.^(-prior_regularity)); 
    % diagonal prior covariance matrix

rng(1)
    %set seed

% Sample and plot a prior draw
theta_rand=mvnrnd(zeros(J_basis,1),prior_cov,1)'; 
    % sample Fourier coefficients from prior

F_rand=zeros(1,mesh_nodes_num_prior);
for j=1:J_basis
    F_rand = F_rand+theta_rand(j)*e_basis(:,j)';
end

f_rand=exp(F_rand);

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
% pCN initialisation

% pCN initialisation at 0
theta_init = zeros(J_basis,1);
F_init_num=zeros(1,mesh_nodes_num_prior);
f_init_num=exp(F_init_num);

% pCN initialisation at f_0
%theta_init = F0_coeff;
%F_init_num=F0_mesh_prior;
%F_init=F0_proj;
%f_init=f_min+ exp(F_init);

% Specify f_init as a function of (location,state) to pass to elliptic PDE solver
F_init=@(location,state) griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),F_init_num,location.x,location.y);
f_init=@(location,state) exp(F_init(location,state));

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
% pCN algorithm

% Parameters for pCN algorithms
delta=.00025; 
    % smaller delta imply shorter moves for the pCN proposal. Use to
    % stabilise pCN acceptance rate at around 30%
MCMC_length=25000; 
    % number of pCN draws

% Trackers
accept_count = zeros(1,MCMC_length); 
    % initialise vector % to keep track of acceptance steps
alphas = ones(1,MCMC_length); 
    % initialise vector to keep  track of acceptance probability
unifs = rand(1,MCMC_length); 
    % i.i.d. Un(0,1) random variables for Metropolis-Hastings updates
theta_MCMC = zeros(J_basis,MCMC_length); 
    % initialise matrix to store the MCMC chain of Fourier coefficients
theta_MCMC(:,1)=theta_init; % set pCN initialisation point
loglik_MCMC = zeros(1,MCMC_length); 
    % initialise vector to keep track of the loglikelihood of the pCN chain
loglik_MCMC(1) = loglik_init; 
    % set initial loglikelihood for initialisation point
loglik_current = loglik_MCMC(1);
theta_current = theta_MCMC(:,1);

tic

for MCMC_index=2:MCMC_length
    fprintf('pCN step n. %d\n',MCMC_index);
    
    % Construct pCN proposal
    theta_rand=mvnrnd(zeros(J_basis,1),prior_cov,1)';
    theta_proposal = sqrt(1-2*delta)*theta_current + sqrt(2*delta)*theta_rand;

    F_proposal_num=zeros(mesh_nodes_num_prior,1);
    for j=1:J_basis
        F_proposal_num = F_proposal_num+theta_proposal(j)*e_basis(:,j);
    end
        
    % Define proposal diffusivity function to pass to PDE solver
    F_proposal=@(location,state) griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),...
        F_proposal_num,location.x,location.y);
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
        theta_current = theta_proposal;
        loglik_current = loglik_proposal;
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1)+1;
    else
        accept_count(1,MCMC_index)=accept_count(1,MCMC_index-1);
    end
    theta_MCMC(:,MCMC_index) = theta_current;
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
theta_mean = mean(theta_MCMC(:,burnin+1:MCMC_length),2);
F_mean_mesh_prior=zeros(mesh_nodes_num_prior,1);
    for j=1:J_basis
        F_mean_mesh_prior = F_mean_mesh_prior+theta_mean(j)*e_basis(:,j);
    end
f_mean_mesh_prior=exp(F_mean_mesh_prior);

% Compute L^2-estimation error
f_mean_bary_prior = griddata(mesh_nodes_prior(1,:),mesh_nodes_prior(2,:),...
    f_mean_mesh_prior,barycenters_prior(1,:),barycenters_prior(2,:));
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
title(['n = ',num2str(n),'; relative error = ',num2str(estim_error/f0_norm)],'FontSize',20);
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
xlabel('x', 'FontSize', 20);
ylabel('y', 'FontSize', 20);
crameri vik