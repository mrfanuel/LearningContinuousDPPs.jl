
using LinearAlgebra
using Plots

# random L 
N = 20;
A = randn(N,N);
L = A*A';
n = size(L,1);

# random K
B = randn(N,N);
K = A*A';
invK = inv(K);

# sampling matrix
id = [1,7,16, 20];
#id = 1:N;
U = Diagonal(vec(ones(n,1)));
U = U[:,id];

# number of iteration
n_it = 50;

# step size
a = 1;

# regularization
lambda = 0;
epsilon = 1e-10;

# initialization
# M = lambda *invK
invM = K/lambda;

obj = zeros(n_it,1);

# iterations
for i in 1:n_it
    D = U *inv(U'*(L+ epsilon*I)*U)*U' - inv(I + L);
    L = a*L*D*L +L;

    obj[i] = -log(det(U'*L*U))+log(det(I + L)) + lambda*tr(L*invK);
end

print("\n")
print("objectives:\n")
print(obj')

plot(1:n_it, obj, xlabel = "iteration", ylabel = "objective value")



