% Copyright (c) 2025 Matteo Giordano
%
% Codes accompanying the article "Bayesian nonparametric inference in elliptic 
% PDEs: Convergence rates and implementation" by Matteo Giordano

%%
% Discrete noisy observations of elliptic PDE solution
%
% Let O be a smooth domain in R^2. Consider the 2D elliptic PDE in
% divergence form with homogeneous Dirichlet boundary conditions:
%
%   div(f_0 grad u)=s_0, in O
%   u=0, on boundary of O
%
% where s_0:O -> R is the (sufficiently smooth) source function and 
% f_0:O -> [f_min,+infty), f_min>0, is the (sufficiently smooth)
% diffusivity.
%
% There exists a unique classical solution G(f_0)=u_f0. We observe data
%
%   Y_i = u_{f0,s_0}(X_i) + sigma W_i,   i = 1,...,n
%
% where X_i are uniform random locations in O, sigma>0 and W_i are i.i.d. 
% N(0,1) random variables.
%
% The following code generates n observations (Y_i,X_i), i=1,...,n

%%
% Create domain and triangular mesh

% Display more digits
format long

ax_h = 1; 
    % length of horizontal semiaxis
ax_v = .75; 
    % length of vertical semiaxis
rot = pi/6;
    % angle of rotation
t = linspace(0,2*pi,1000);
pgon = polyshape({ax_h*cos(t)*cos(rot) - ax_v*sin(t)*sin(rot)},...
    {ax_v*sin(t)*cos(rot) + ax_h*cos(t)*sin(rot)});
vol = pi*ax_h*ax_v;

% Create a triangulation representation of pgon
tr = triangulation(pgon);

% Create a PDE model
model = createpde;

% With the triangulation data as a mesh, use the geometryFromMesh function
% to create a geometry
tnodes = tr.Points';
telements = tr.ConnectivityList';
geometryFromMesh(model,tnodes,telements);
%pdegplot(model)

% Generate and plot triangular mesh
generateMesh(model,'Hmax',.05);
%figure()
%axes('FontSize', 20, 'NextPlot','add')
%pdemesh(model)
%xticks([-1,-.5,0,.5,1])
%yticks([-1,-.5,0,.5,1])
%xlabel('x', 'FontSize', 20);
%ylabel('y', 'FontSize', 20);
mesh_nodes = model.Mesh.Nodes; 
    % 2 x mesh_nodes_num matrix whose columns contain the (x,y) coordinates 
    % of the nodes in the mesh
mesh_nodes_num = size(mesh_nodes); 
mesh_nodes_num=mesh_nodes_num(2); 
    % number of nodes in the mesh
mesh_elements = model.Mesh.Elements; 
    % 6 x mesh_elements_num whose columns contain the 6 node indices 
    % identifying each triangle. The first 3 elements of each column contain 
    % the indices of the 3 vertices of the triangle 
mesh_elements_num = size(mesh_elements); 
mesh_elements_num = mesh_elements_num(2); 
    % number of triangles in the mesh
[~,mesh_elements_area] = area(model.Mesh); 

% Compute barycenters of triangular mesh elements
barycenters = zeros(2,mesh_elements_num);
for i=1:mesh_elements_num
    barycenters(:,i) = mean(mesh_nodes(:,mesh_elements(1:3,i)),2);
end

%%
% Specify true diffusivity f_0

% Specify f_0 as a function of (x,y)
f_min = 1;
% Main experiment
f0 = @(x,y) f_min + exp(-(5*x-1.75).^2-(5*y-1.75).^2)...
    + exp(-(5*x-1.75).^2-(5*y+1.75).^2)...
    + exp(-(5*x+1.75).^2-(5*y+1.75).^2)...
    +exp(-(5*x+1.75).^2-(5*y-1.75).^2);

% Further experiments
%f0 = @(x,y) f_min + exp(-(4*x-1.75).^2-(4*y-1).^2)...
%    +exp(-(4*x+1.75).^2-(4*y+1).^2);
%f0 = @(x,y) f_min + exp(-(5*x-1.75).^2-(2.5*y-.25).^2)...
%    +exp(-(5*x+1.75).^2-(5*y+1).^2);
%f0 = @(x,y) f_min + .5*exp(-(5*x-1.75).^2-(5*y-1.75).^2)...
%    -.5*exp(-(5*x+1.75).^2-(5*y+1.75).^2);
f0_mesh = f0(mesh_nodes(1,:),mesh_nodes(2,:));
f0_bary = f0(barycenters(1,:),barycenters(2,:));
f0_norm=sqrt(sum(f0_bary.*mesh_elements_area));
f0_fun=@(location,state) f0(location.x,location.y);
    % specify diffusivity as a functions of (location,state) for elliptic PDE solver

% Plot f0
figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model,'XYData',f0_mesh,'ColorMap',jet)
colorbar('Fontsize',20)
title('True diffusivity f_0','FontSize',20);
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
xlabel('x', 'FontSize', 20);
ylabel('y', 'FontSize', 20);
crameri vik

% Specify the unknown source s_0 as a function of (x,y)
s0 = @(x,y) exp(-(5*x-2.5).^2-(5*y).^2)+exp(-(7.5*x).^2-(2.5*y).^2)...
    +exp(-(5*x+2.5).^2-(5*y).^2);
s0_mesh=s0(mesh_nodes(1,:),mesh_nodes(2,:));
s0_bary = s0(barycenters(1,:),barycenters(2,:));
s0_fun=@(location,state) f0(location.x,location.y);
    % specify diffusivity as a functions of (location,state) for elliptic PDE solver

% Plot s0
figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model,'XYData',s0_mesh,'ColorMap',jet)
colorbar('Fontsize',20)
title('True source s_0','FontSize',20);
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
xlabel('x', 'FontSize', 20);
ylabel('y', 'FontSize', 20);
crameri vik

%%
% Elliptic PDE solution u_{f_0,s_0}

% Specify zero Dirichlet boundary conditions on all edges
applyBoundaryCondition(model,'dirichlet','Edge',1:model.Geometry.NumEdges,'u',0);

% Specify the coefficients with c=f_0 and f=1 (the sintax for the source in MATLAB)
specifyCoefficients(model,'m',0,'d',0,'c',f0_fun,'a',0,'f',s0_fun);

% Solve PDE and plot solution
results = solvepde(model);
u0 = results.NodalSolution; 
figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model,'XYData',u0)
title('PDE solution u_{f_0,s_0}','FontSize', 20)
xlabel('x','FontSize', 20)
ylabel('y', 'FontSize', 20)
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
crameri batlowW

% Compute norm of u_{f_0,s_0} for SNR
u0_interp=scatteredInterpolant(mesh_nodes(1,:)',mesh_nodes(2,:)',u0);
u0_bary=u0_interp(barycenters(1,:),barycenters(2,:));
u0_norm = sqrt(sum((u0_bary).^2.*mesh_elements_area));
%disp(['Norm of u0 = ', num2str(u0_norm)]);

%%
% Noisy observations of PDE solution and log-likelihood computation

rng(1)

% Sample design points and noise variales
n=1000; 
    % number of observations
sigma=.0025; 
    % noise standard deviation
disp(['Signal to noise ratio = ', num2str(u0_norm/sigma)])
rand_index=sort(randsample(mesh_nodes_num,n));
    % random indices in the mesh
rand_mesh=mesh_nodes(:,rand_index); 
    % random sample of mesh points drawn uniformly at random

figure()
axes('FontSize', 20, 'NextPlot','add')
scatter(rand_mesh(1,:),rand_mesh(2,:),'filled') 
    % plot of sampled locations
title('X_1,...,X_n','FontSize', 20)
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
xlabel('x','FontSize', 20)
ylabel('y', 'FontSize', 20)

% Sample observations
observations=u0(rand_index)+(mvnrnd(zeros(n,1),sigma^2*eye(n)))'; 
    % add i.i.d N(0,sigma^2) noise to the observation
u0_corrupted=u0;
u0_corrupted(rand_index)=observations;

% Plot corrupted PDE solution
figure()
axes('FontSize', 20, 'NextPlot','add')
pdeplot(model,'XYData',u0_corrupted)
title('Observations Y_i=u_{f_0,s_0}(X_i)+\sigma W_i','FontSize', 20);
xlabel('x','FontSize', 20)
ylabel('y', 'FontSize', 20)
xticks([-1,-.5,0,.5,1])
yticks([-1,-.5,0,.5,1])
crameri batlowW

% Compute log-likelihood of f_0
loglik0=-sum((observations-u0(rand_index)).^2 )/(2*sigma^2);
disp(['Log-likelihood of f_0 = ', num2str(loglik0)])

% Compare likelihood with constant diffusivity for comparison
specifyCoefficients(model,'m',0,'d',0,'c',1,'a',0,'f',s0_fun);
results = solvepde(model);
u_const_diff = results.NodalSolution;
loglik_const_diff=-sum((observations-u_const_diff(rand_index)).^2 )/(2*sigma^2);
disp(['Log-likelihood of constant f =', num2str(loglik_const_diff)])
