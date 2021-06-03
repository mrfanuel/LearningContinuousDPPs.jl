include("algo/estimateGaussianB.jl")
include("algo/constructSquareGrid.jl")
include("algo/integralKernelFunction.jl")
include("algo/approxCorrelationKernelMatrix.jl")
include("algo/evaluateGramGrid.jl")
include("algo/correlationKernelGram.jl")
include("algo/likelihoodKernelGram.jl")
using JLD

# Here we use the square SqExponentialKernel

function demoEstimationGaussian(s::Int64=10,n::Int64=200,p::Int64=1000,sigma::Float64=0.05,lambda::Float64=0.1,epsilon::Float64=1e-12,it_max::Int64=10000,tol::Float64=1e-5)
    

    FredholmSample = rand(Uniform(0,1), n,2);

    ###################################
    # estimation of representer matrix
    ###################################
    merge = false;
    B, R , K, k, totalSamples, obj, i_stop = estimateGaussianB(s,n,sigma,lambda,epsilon,it_max,tol,FredholmSample,merge)

    print("\n")
    print("cond number of K : $(cond(K))")
    print("\n")

    R = cholesky(K).U;

    ###################################
    # estimation of likelihood kernel
    ###################################

    # construct grid of n_test points
    n_test = 30*30; a = 0.; b = 1.;
    print("\n")
    print("test points in [$(a), $(b)]")
    print("\n")

    testSamples = constructFlatSquareGrid(n_test, a, b);
    GramA = likelihoodKernelGram(B,R,totalSamples,testSamples,k,sigma);

    ###################################
    # estimation of correlation kernel
    ###################################
    print("\n")
    print("estimate  correlation kernel...")
    print("\n")

    c_1 = 0.; c_2 = 1.;
    d = 2;

    unifSamples = rand(Uniform(c_1,c_2), p,d);
    GramK = correlationKernelGram(B,R,unifSamples,totalSamples,testSamples,k,sigma);

    k0 = SqExponentialKernel();
    alpha0 = 0.05;
    rho0 = 100;
    x0 = (testSamples)'/(alpha0/sqrt(2));

    GramK0 = rho0*kernelmatrix(k0, x0) + epsilon *I ; # positive definite

    n_max = 30;
    GramA0 = zeros(size(GramA));
    for i = 1:n_max
        x0 = (testSamples)'/(sqrt(i)*alpha0/sqrt(2));
        factor = (1/i^(d/2))*rho0^i * (sqrt(pi)*alpha0)^((i-1)*d)
        GramA0 += factor * kernelmatrix(k0, x0)
    end


    filename = "results/result_s="*string(Int64(s))*"_n="*string(Int64(n))*"_p="*string(Int64(p))*"_sigma="*string(Int64(1000*sigma))*"_lambda="*string(Int64(1000000*lambda))*"divideBy1Million_tol="*string(Int64(1e6*tol))*".jld";

    save(filename, "B",B, "R",R, "n",n, "totalSamples",totalSamples, "GramK",GramK, "GramA",GramA, "GramK0",GramK0, "GramA0",GramA0, "obj",obj, "i_stop",i_stop);

end


