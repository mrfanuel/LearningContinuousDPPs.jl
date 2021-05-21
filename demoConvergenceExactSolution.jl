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
include("algo/PicardObjectiveB.jl")

function demoConvergenceExactSolution()

    # width
    sigma = .5

    # regularizer
    lambda =  1.# too large 2# too small 0.5; # good 1.

    # regularizer for positive definiteness
    epsilon = 1e-14; 

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
    id_temp = collect(1:n);
    # indices of DPP samples
    push!(idDppSamples,id_temp);

    # indices of samples for approximating Fredholm
    FredholmSample = collect(1:n); # i.e., the set I

    # twice the same set
    totalSamples = temp;

    # construct Gaussian kernel
    k = SqExponentialKernel();

    x = (totalSamples)'/sigma;

    K = kernelmatrix(k, x) + 1e-10 *I ; # makes sure it is positive definite

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

    # Chol decomposition
    R = cholesky(K).U;
    Rinv = inv(R);

    # sampling matrix for uniformSample
    unifU = identity[:,FredholmSample];
    nb_unif = length(FredholmSample)

    # Exact solution

    X_exact = 0.5*(real(sqrt(m^2*I + 4*m*K/lambda))-m*I);
    B_exact = Rinv'*X_exact*Rinv;
    norm_B_exact = norm(B_exact);

    # initialization
    obj = zeros(it_max,1);
    obj_det = zeros(it_max,1);
    diff = zeros(it_max,1);

    i_stop = it_max;

    # initial positive definite iterate
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

        B = (0.5/lambda)*(real(sqrt(I+4*lambda*pB))-I);
        B = 0.5*(B+B');


        diff[i] = norm(B-B_exact)/norm_B_exact;

        # track the objective values

        obj_det0,ob_reg0 = PicardObjectiveB(B, samples, FredholmSample, R,lambda);
        obj[i] = obj_det0 + ob_reg0;
        obj_det[i] = obj_det0;
    
        if i%10 == 0
            print("---------------------------------------------------------------\n")
            print("$(i) / $it_max\n")
            print("relative objective variation $(abs(obj[i]-obj[i-1])/abs(obj[i]))\n")
            print("objective = $(obj[i]) \n")
            print("norm(B) = $(norm(B))\n")
            print("norm(B-B_exact)/norm(B_exact) = $(diff[i])\n")

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

    obj_exact_det, obj_exact_reg = PicardObjectiveB(B_exact, samples, FredholmSample, R,lambda);
    
    obj_exact = obj_exact_det + obj_exact_reg;
    print("norm(B_exact) = $(norm(B_exact))\n")
    print("obj_exact = $(norm(obj_exact))\n")


    # plotting range
    i_start = 1;
    r = i_start:i_stop;

    # relative error
    plt_error = plot(r, diff[r], yaxis=:log, legend = false);
    xlabel!("iterations")
    ylabel!("relative error")
    display(plt_error)
    savefig("figures/demo_MVJacobi_plot_1_dppSamples_50_points_error.pdf")


    # objective decay
    objective_optimal = ones(size(obj))*obj_exact;

    plt_objective = plot(r, obj[r],legend = false);
    plot!(r, objective_optimal[r],legend = false);

    xlabel!("iterations")
    ylabel!("objective")
    display(plt_objective)
    savefig("figures/demo_MVJacobi_plot_1_dppSamples_50_points_objective.pdf")

    # objective det 
    #objective_optimal_det = ones(size(obj))*obj_exact_det;
    #plt_objective_det = plot(r, obj_det[r],legend = false);
    #plot!(r, objective_optimal_det[r],legend = false);
    #xlabel!("iterations")
    #ylabel!("det objective")
    #display(plt_objective_det)
    #savefig("figures/demo_MVJacobi_plot_1_dppSamples_50_points_objective_det.pdf")

    return B, B_exact


end