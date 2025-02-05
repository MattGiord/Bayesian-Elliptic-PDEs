# Bayesian-Elliptic-PDEs

MATLAB code for Bayesian nonparametric estimation with Gaussian priors in elliptic inverse problems.

Author: Matteo Giordano, https://matteogiordano.weebly.com.

This repository is associated with the article "Bayesian nonparametric inference in PDE models: asymptotic theory and implementation" 
by Matteo Giordano. The abstract of the paper is:

"Parameter identification problems in partial differential equations (PDEs) consist in determining one or more unknown functional 
parameters in a PDE. Here, the Bayesian nonparametric approach to such problems is considered. Focusing on the representative example 
of inferring the diffusivity function in an elliptic PDE from noisy observations of the PDE solution, the performance of Bayesian procedures 
based on Gaussian process priors is investigated. Recent asymptotic theoretical guarantees establishing posterior consistency and convergence 
rates are reviewed and expanded upon. An implementation of the associated posterior-based inference is provided, and illustrated via a numerical 
simulation study where two different discretisation strategies are devised."

This repository contains the MATLAB code to replicate the numerical simulation study presented in Section 4 of the article. It contains the 
following:
1. GenerateObservations.m code to generate the observations (discrete point evaluations of an elliptic PDE solution corrupted by additive Gaussian measurement errors)
2. pCNMatern.m code to implement posterior inference based on the Matérn process prior, via the pCN algorithm. It requires the output of GenerateObservations.m and the auxiliary function in K_mat.m
3. K_mat.m auxiliary code for the Matérn covariance kernel, required by pCNMatern.m
4. pCNSeries.m code to implement posterior inference based on truncated Dirichlet-Laplacian Gaussian series expansions, via the pCN algorithm.

For questions or for reporting bugs, please e-mail Matteo Giordano (matteo.giordano@unito.it)

Please cite the following article if you use this repository in your research: Giordano, M (2023). Bayesian nonparametric inference in PDE models: asymptotic theory and implementation
