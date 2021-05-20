# import utilities
using LinearAlgebra
using Plots
using CSV
using KernelFunctions
using Distributions
using DataFrames

# algo for regularized Picard
include("algo/regularizedPicard.jl")
include("algo/integralCorrelationKernelFunction.jl")
include("algo/approxCorrelationKernelMatrix.jl")

function demoEstimationGaussian()
    # number of DPP samples
    s = 1;

    # number of uniform samples for Fredholm
    n = 50 # 100 is better;

    # number of uniform samples for correlation kernel
    p = 500;

    # type of kernel
    # kernel = "MaternKernel"
    # nu = 5/2.;
    kernel = "SqExponentialKernel"


    # kernel bw (Float64)
    sigma = 0.5;

    # regularization (Float64)
    lambda =  1.

    # regularizer for K positive definite
    epsilon = 1e-6; 

    # max number of iteration
    it_max = 5000;

    # relative objective tolerance
    tol = 1e-6;

    # Plotting number of grid points along x-axis
    n_step_plot = 100; 

    # number of samples
    s = 1;

    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    # uniform sampling in the box [0,1]^2
    c = 1.;
    totalSamples = rand(Uniform(0,1), n,2);
    unifSample = collect(1:n); # i.e., the set I

    print("Loading samples...")
    print("\n")

    # loading DPP samples
    id_last_sample = n;
    # gather the samples
    for i = 0:(s-1)
        # read files and specify that they start from first row
        temp = CSV.File("data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_100_nb_"*string(i+1)*".csv"; header=true) |> Tables.matrix 
        temp = temp[:,2:3];
        # temp is a matrix with 2 cols
        # add it as a new entry in matSamples
        id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
        id_last_sample = id_last_sample + size(temp,1);
        push!(idDppSamples,id_temp);
        totalSamples = [totalSamples; temp];
    end
    print(idDppSamples)
    x = (totalSamples)'/sigma;

    # construct Gaussian kernel

    kernel="SqExponentialKernel"
    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; # makes sure it is positive definite

    ################################
    # solve estimation problem
    ################################
    print("max nb of iteration: $it_max \n")
    print("Picard iteration...")

    # iterations
    X, obj, i_stop = regularizedPicard(K, idDppSamples, unifSample, lambda, it_max ,tol)


    # recover matrix C
    invK = inv(K);
    X = 0.5*(X+X');# makes sure it is symmetric
    C = invK*X*invK; 
    C = 0.5*(C+C');# makes sure it is symmetric

    ###################################
    # estimation of correlation kernel
    ###################################
    print("\n")
    print("estimate  correlation kernel...")
    
    K_hat_mat = approxCorrelationKernelMatrix(C,p,c,totalSamples,k,sigma);

    print("\n")

    print("plotting...")

    intensity = zeros(Float64, n_step_plot,n_step_plot);

    t = 0
    for i in 1:n_step_plot
        for j in 1:n_step_plot
            t += 1
            x = (i-1)/(n_step_plot-1);
            y = (j-1)/(n_step_plot-1);
            v = [x y]';
            intensity[i,j] = integralCorrelationKernelFunction(v,v,K_hat_mat,totalSamples,k,sigma);
        end
    end   

    # plotting intensity of the estimated process

    x_tics = (0:(1/(n_step_plot-1)):1);
    y_tics = x_tics;
    display(heatmap(x_tics,
    y_tics,
    intensity,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="x",
    ylabel="y",
    title="estimated intensity"))

    return 1;

end

