
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
lambda = 1;
epsilon = 1e-15;

# initialization
invM = K/lambda; 
M = lambda *invK;
sqrtM = sqrt(lambda)*sqrt(invK);
sqrtinvM = sqrt(K)/sqrt(lambda);


obj = zeros(n_it,1);

# iterations
for i in 1:n_it
    D = U *inv(U'*(L+ epsilon*I)*U)*U' - inv(I + L);
    fL = a*L*D*L +L;
    L = 0.5*sqrtinvM*( sqrt(I + 4*sqrtM*fL*sqrtM) -I )*sqrtinvM;
    obj[i] = -log(det(U'*L*U))+log(det(I + L)) + lambda*tr(L*invK);
end

#print("\n")
#print("objectives:\n")
#print(obj')

# When id = 1:N
#L_exact = 0.5*(sqrt(I + 4*K/lambda) -I);
#print("\n")
#print(norm(L-L_exact)/norm(L))

plot(1:n_it, obj, xlabel = "iteration", ylabel = "objective value")


