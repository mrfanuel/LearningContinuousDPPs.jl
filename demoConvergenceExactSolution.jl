using LinearAlgebra
using Plots
using CSV
using KernelFunctions
using Distributions
using DataFrames

include("algo/estimateMVJacobi.jl")
include("algo/regularizedPicard.jl")
include("algo/integralCorrelationKernelFunction.jl")
include("algo/approxCorrelationKernelMatrix.jl")

function demoConvergenceExactSolution()

    # width
    sigma = 1.

    # regularizer
    lambda =  1.# too large 2# too small 0.5; # good 1.

    # regularizer for K positive definite
    epsilon = 1e-6; 

    # max number of iteration
    it_max = 5000;

    # relative objective tolerance
    tol = 1e-6

    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    # load one DPP sample 
    i = 1;
    temp = CSV.File("data/dppy/samples/MultivariateJacobiOPE_sample"*string(i)*".csv"; header=false) |> Tables.matrix 

    # number of rows
    n = size(temp,1);

    id_temp = collect((n+1):(2*n));

    # indices of samples for approximating Fredholm
    FredholmSample = collect(1:n); # i.e., the set I

    # indices of DPP samples
    push!(idDppSamples,id_temp);

    # twice the same set
    totalSamples = [temp; temp];

    # construct Gaussian kernel
    k = SqExponentialKernel();

    x = (totalSamples)'/sigma;

    K = kernelmatrix(k, x) + epsilon *I ; # makes sure it is positive definite

    ################################
    # solve estimation problem
    ################################
    print("Picard iteration...")
    samples = idDppSamples;

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
    unifU = identity[:,FredholmSample];
    nb_unif = length(FredholmSample)

    # Exact solution

    X_exact = 0.5*(sqrt(m^2*I + 4*m*K/lambda)-m*I);
    norm_X_exact = norm(X_exact);

    # initialization
    obj = zeros(it_max,1);
    diff = zeros(it_max,1);

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

        diff[i] = norm(X-X_exact)/norm_X_exact;

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
            print("norm(X-X_exact)/norm(X_exact) = $(diff[i])\n")

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

    i_start = 10;
    plt_objective = plot(i_start:i_stop, obj[i_start:i_stop],legend = false);
    xlabel!("iterations")
    ylabel!("objective")
    display(plt_objective)
    savefig("figures/demo_MVJacobi_plot_1_dppSamples_50_points_objective.pdf")

    plt_error = plot(i_start:i_stop, diff[i_start:i_stop], yaxis=:log, legend = false);
    xlabel!("iterations")
    ylabel!("relative error")
    display(plt_error)
    savefig("figures/demo_MVJacobi_plot_1_dppSamples_50_points_error.pdf")



end



 
