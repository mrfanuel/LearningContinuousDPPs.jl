include("algo/estimateGaussianB.jl")
include("algo/constructSquareGrid.jl")
include("algo/integralKernelFunction.jl")
include("algo/approxCorrelationKernelMatrix.jl")
include("algo/evaluateGramGrid.jl")
include("algo/correlationKernelGram.jl")
include("algo/likelihoodKernelGram.jl")


# Here we use the square SqExponentialKernel

function demoEstimationGaussian()
    # number of DPP samples
    s = 10; 

    # number of uniform samples for Fredholm
    n = 300; #300

    # number of uniform samples for correlation kernel
    p = 1000;

    #kernel = "MaternKernel";
    kernel = "SqExponentialKernel"; # exp(-d^2/2)

    nu = 5/2.;

    # kernel bw (Float64)
    sigma = 0.1; ;# last 0.1; # last 0.05

    # regularization (Float64)£
    lambda =  1e-2 # last 1e-4

    # regularizer for K positive definite
    epsilon = 1e-12; 

    # max number of iteration
    it_max = 10000;

    # relative objective tolerance
    tol = 1e-5; # last 1e-6

    # merge dpp and unif samples
    merge = false # true improves

    FredholmSample = rand(Uniform(0,1), n,2);

    ###################################
    # estimation of representer matrix
    ###################################

    B, R ,K,k,totalSamples,obj,i_stop = estimateGaussianB(s,n,kernel,nu,sigma,lambda,epsilon,it_max,tol,FredholmSample,merge)

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

    #savefig( "figures/intensityDiagGaussianMerge.pdf")

    # known result

    k0 = SqExponentialKernel();
    alpha0 = 0.05;
    rho0 = 50;
    x0 = (testSamples)'/(alpha0/sqrt(2));

    GramK0 = rho0*kernelmatrix(k0, x0) + epsilon *I ; # makes sure it is positive definite

    intensity = diag(GramK);
    display(plot(intensity,legend=false));
    display(plot!(50*ones(size(intensity)),legend=false))

    n_max = 30;
    likelihoodKernel0 = zeros(size(GramA));
    for i = 1:n_max
        x0 = (testSamples)'/(sqrt(i)*alpha0/sqrt(2));
        factor = (1/i^(d/2))*rho0^i * (sqrt(pi)*alpha0)^((i-1)*d)
        likelihoodKernel0 += factor * kernelmatrix(k0, x0)
    end

    #plot(real(eigvals(R'*B*R)), framestyle = :box, legend=false)
    #plot(real(diag(R'*B*R)), framestyle = :box, legend=false)
    return B, R, GramK, GramA, GramK0, obj, i_stop;

end


