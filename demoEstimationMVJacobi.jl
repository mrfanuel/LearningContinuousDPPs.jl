include("algo/estimateMVJacobi.jl")

# number of DPP samples
s = 2;

# number of uniform samples for Fredholm
n = 300 # is better;

# number of uniform samples for correlation kernel
p = 500;

# type of kernel
#kernel = "MaternKernel"
kernel = "SqExponentialKernel"

# Matern nu, only for Matern (Float64)
#nu = 0.5; # Laplace
#nu = 1.5; # linear times Laplace
#nu = 5/2.;

# kernel bw (Float64)

#sigma = 2.; # for Matern larger than SqExponentialKernel
#sigma = 0.5; # for SqExponentialKernel # good 0.5

sigma = 2. # not bad choice

# regularization (Float64)

# for SqExponentialKernel
lambda =  1.# too large 2# too small 0.5; # good 1.

# for Matern
#lambda =  0.5; # more or less fine

# regularizer for K positive definite
epsilon = 1e-6; # especially for SqExponentialKernel

# max number of iteration
it_max = 5000;

# relative objective tolerance
tol = 1e-6;

# Plotting number of grid points along x-axis
n_step_plot = 100; 


GramMatrix = estimateMVJacobi(s,n,p,kernel,nu,sigma,lambda,epsilon,it_max,tol,n_step_plot);

# heatmap(GramMatrix)
