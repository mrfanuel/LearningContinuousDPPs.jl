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

# number of samples
nb_samples = length(samples); 

# define identity matrix    
m = size(K,1);
identity = Diagonal(vec(ones(m,1)));

# define inverse kernel matrix    
invK = inv(K);

# Chol decomposition
R = cholesky(K).U;
Rinv = inv(R);

# sampling matrix for uniformSample
unifU = identity[:,unifSample];
nb_unif = length(unifSample)

# initialization
obj = zeros(it_max,1);
i_stop = it_max;

# initial positive definite iterate
epsilon = 1e-10; # for positive definiteness
X = randn(m,m);
X = X*X'+ epsilon*I;

# iterations
for i in 1:it_max
    # construct  Delta
    Delta = zeros(m,m);
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        Delta = Delta + U *inv(U'*(X+ epsilon*I)*U)*U';
    end

    Delta = Delta/nb_samples - unifU*inv(unifU'*(nb_unif*I + X)*unifU)*unifU';
    Delta = 0.5*(Delta + Delta');

    # Picard iteration
    gX = X*Delta*X +X;

    # final expression
    temp = real(sqrt(I + 4*lambda*Rinv'*gX*Rinv))
    X = (0.5/lambda)*R'*( temp -I )*R;

    # track the objective values
    ob = 0
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        ob = ob - logdet(U'*X*U + 1e-10*I);
        if ob==Inf
            error("singular determinant in objective")
        end
    end
    #print("ob $(ob) \n" )
    ob = ob/nb_samples;
    meanlodetXCC = ob;
    ob = ob+logdet(I + (1/nb_unif)*unifU'*X*unifU) + lambda*tr(X*invK);
    obj[i] = ob;

    if i%100 == 0
        print("---------------------------------------------------------------\n")
        print("$(i) / $it_max\n")
        print("relative objective variation $(abs(obj[i]-obj[i-1])/abs(obj[i]))\n")
        print("objective = $ob \n")
        print("mean lodet(X_CC) = - $meanlodetXCC\n")
        print("norm(X) = $(norm(X))\n")

    end
    # stopping criterion
    if i>1 && abs(obj[i]-obj[i-1])/abs(obj[i])< tol
        i_stop = i;
        print("---------------------------------------------------------------\n")
        print("Relative tolerance $(tol) attained after $(i) iterations.\n")
        print("Final objective= $(obj[i])\n")
        print("---------------------------------------------------------------\n")
        break
    end
    if i==it_max
        print("iteration has not yet converged.")
    end
end

return X, obj, i_stop

end