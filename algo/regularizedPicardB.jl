include("PicardObjectiveB.jl")

function regularizedPicardB(K::Array{Float64,2}, samples::Array{Array{Int64,1},1}, unifSample::Array{Int64,1}, lambda::Float64, it_max::Int64 ,tol::Float64)

# number of samples
nb_samples = length(samples); 

# define identity matrix    
m = size(K,1);
identity = Diagonal(vec(ones(m,1)));

# Chol decomposition
R = cholesky(K).U;

# sampling matrix for uniformSample
unifU = identity[:,unifSample];
nb_unif = length(unifSample)

# initialization
obj = zeros(it_max,1);
i_stop = it_max;

# initial positive definite iterate
epsilon = 1e-10; # for positive definiteness
X = randn(m,m);
B = X*X'+ 1e-14*I;

# iterations
for i in 1:it_max
    # construct  Delta
    Delta = zeros(m,m);
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        Delta = Delta + U *inv(U'*(R'*B*R+ epsilon*I)*U)*U';
    end

    Delta = Delta/nb_samples - unifU*inv(unifU'*(nb_unif*I + R'*B*R)*unifU)*unifU';
    Delta = 0.5*(Delta + Delta');

    # Picard iteration
    pB = B + B*R*Delta*R'*B;

    # final expression
    B = (0.5/lambda)*(real(sqrt(I+4*lambda*pB))-I);
    B = 0.5*(B+B');

    # track the objective values
    obj_det0,ob_reg0 = PicardObjectiveB(B, samples, unifSample, R,lambda);
    obj[i] = obj_det0 + ob_reg0;

    if i%10 == 0
        print("---------------------------------------------------------------\n")
        print("$(i) / $it_max\n")
        print("relative objective variation $(abs(obj[i]-obj[i-1])/abs(obj[i]))\n")
        print("objective = $(obj[i]) \n")
        print("norm(B) = $(norm(B))\n")

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

return B, obj, i_stop

end