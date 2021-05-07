using LinearAlgebra
using Plots
include("regularizedPicard.jl")

## Warning: we solve only the case where S includes all the dpp samples

# total number of points
n = 100;

# random K
B = randn(n,n);
K = B*B';

# define some samples
id1 = [1,7,16, 20];
id2 = [3,9,12, 17];
samples = [[id1];[id2]]; # array of arrays

# max number of iteration
it_max = 1000;
tol = 1e-4;

# regularization
lambda = 1.;

# iterations
X, obj, i_stop = regularizedPicard(K, samples, lambda, it_max ,tol)

# plotting objectives
plot(1:i_stop, obj[1:i_stop], xlabel = "iteration", ylabel = "objective value")



