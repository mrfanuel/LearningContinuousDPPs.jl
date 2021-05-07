"""
    regularizedPicard(K::Array{Float64,2}, samples::Array{Array{Int64,1},1}, lambda::Float64, it_max::Int64 ,tol::Float64)

Compute the a fixed point of the regularized Picard iteration

# Arguments
- `K`: is an invertible nxn kernel matrix
- `samples`: is an array containing arrys of indices from 1 to n
- `lambda`: is a real regularizer
- `it_max`: is the max number of iterations
- `tol`: is stopping criterion 

"""
function regularizedPicard(K::Array{Float64,2}, samples::Array{Array{Int64,1},1}, lambda::Float64, it_max::Int64 ,tol::Float64)

## Warning: we solve only the case where S includes all the dpp samples

# number of samples
nb_samples = length(samples); 

# define identity matrix    
n = size(K,1);
identity = Diagonal(vec(ones(n,1)));

# define inverse kernel matrix    
invK = inv(K);

# Chol decomposition
R = cholesky(K).U;
Rinv = inv(R);

# step size
a = 1; 

# regularizer
epsilon = 1e-15; 

# initialization
obj = zeros(n_it,1);
i_stop = n_it;

# initial iterate
X = randn(n,n);
X = X*X';

# iterations
for i in 1:it_max
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
    temp = sqrt(I + 4*lambda*Rinv'*gX*Rinv)
    X = (0.5/lambda)*R'*( temp -I )*R;

    # track the objective values
    obj[i] = -log(det(U'*X*U))+log(det(I + X)) + lambda*tr(X*invK); # check if tr log is not better.

    # stopping criterion
    if i>1 && abs(obj[i]-obj[i-1])/abs(obj[i])< tol
        i_stop = i;
        print("Relative tolerance $(tol) attained after $(i) iterations.\n")
        break
    end
    if i==it_max
        print("iteration has not yet converged.")
    end
end

return X, obj, i_stop

end