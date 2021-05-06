
using LinearAlgebra
using Plots



# random K
B = randn(N,N);
K = A*A';
invK = inv(K);

# sampling matrix
id1 = [1,7,16, 20];
id2 = [3,9,12, 17];
samples = [[id1];[id2]]; # array of arrays
nb_samples = length(samples); # number of samples

#id = 1:N;
identity = Diagonal(vec(ones(n,1)));

# number of iteration
n_it = 50;

# step size
a = 1;

# regularization
lambda = 1;
epsilon = 1e-15;

# initialization
R = cholesky(K).U;
Rinv = inv(R);
obj = zeros(n_it,1);

# iterations
for i in 1:n_it
    # construct  Delta
    Delta = zeros(n,n);
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        Delta = Delta + U *inv(U'*(X+ epsilon*I)*U)*U';
    end
    Delta = Delta/nb_samples - inv(I + X);
    # Picard iteration
    gX = a*X*Delta*X +X;
    # final expression
    X = (0.5/lambda)*R'*( sqrt(I + 4*lambda*Rinv'*gX*Rinv) -I )*R;
    # track the objective values
    obj[i] = -log(det(U'*X*U))+log(det(I + X)) + lambda*tr(X*invK); # check if tr log is not better.
end


plot(1:n_it, obj, xlabel = "iteration", ylabel = "objective value")


