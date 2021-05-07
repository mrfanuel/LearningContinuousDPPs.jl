
using LinearAlgebra
using Plots
include("regularizedPicardWithUniform.jl")

## Warning: we solve only the case where S includes all the dpp samples

# total number of points
n = 50;

# random K
B = randn(n,n);
K = B*B';

# define some samples (not clean yet)
id1 = [1,7,16, 20, 35, 42];
id2 = [3,9,12, 17, 21, 49];
samples = [[id1];[id2]]; # array of arrays

# for approximating the Fredholm det
unifSample = setdiff(1:n,[id1 ;id2]); 

# max number of iteration
it_max = 1000;
tol = 1e-4;

# regularization
lambda = 1.;

# iterations
X, obj, i_stop = regularizedPicardWithUniform(K, samples, unifSample, lambda, it_max ,tol)

# plotting objectives
plot(1:i_stop, obj[1:i_stop], xlabel = "iteration", ylabel = "objective value")

