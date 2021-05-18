"""
    regularizedPicard(K::Array{Float64,2}, samples::Array{Array{Int64,1},1}, unifSample::Array{Int64,1} , lambda::Float64, it_max::Int64 ,tol::Float64)

Compute the a fixed point of the regularized Picard iteration

# Arguments
- `K`: is an invertible nxn kernel matrix
- `samples`: is an array containing arrays of indices from 1 to n
- `unifSample`: is a sample for approximating the Fredholm det.
- `lambda`: is a real regularizer
- `it_max`: is the max number of iterations
- `tol`: is stopping criterion 

"""
function regularizedPicard(K::Array{Float64,2}, samples::Array{Array{Int64,1},1}, unifSample::Array{Int64,1}, lambda::Float64, it_max::Int64 ,tol::Float64)

## Warning: test to see if S can be chosen uniformly (it seems to work)

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

# sampling matrix for uniformSample
unifU = identity[:,unifSample];
nb_unif = length(unifSample);

# step size
a = 1.; # not changed for the moment 

# initialization
obj = zeros(it_max,1);
i_stop = it_max;

# initial positive definite iterate
epsilon = 1e-10; # for positive definiteness
X = randn(n,n);
X = X*X'+ epsilon*I;

# iterations
for i in 1:it_max
    # construct  Delta
    Delta = zeros(n,n);
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        Delta = Delta + U *inv(U'*(X+ epsilon*I)*U)*U';
        #Delta = Delta + U *((U'*(X+ epsilon*I)*U)\U');
    end
    
    Delta = Delta/nb_samples - unifU*inv(unifU'*nb_unif*I + X)*unifU';
    #Delta = Delta/nb_samples - unifU*((unifU'*(I + X)*unifU)\unifU');
    Delta = 0.5*(Delta + Delta');

    # Picard iteration
    gX = a*X*Delta*X +X;

    # final expression
    temp = real(sqrt(I + 4*lambda*Rinv'*gX*Rinv))
    X = (0.5/lambda)*R'*( temp -I )*R;

    # track the objective values
    ob = 0
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        ob = ob - log(det(U'*X*U+ epsilon*I));
    end
    ob = ob/nb_samples;
    ob = ob+log(det(I + (1/nb_unif)*unifU'*X*unifU)) + lambda*tr(X*invK);
    obj[i] = ob;

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