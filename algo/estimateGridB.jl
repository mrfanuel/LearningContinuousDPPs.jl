# import utilities
using LinearAlgebra
using Plots
using CSV
using KernelFunctions
using Distributions
using DataFrames

# algo for regularized Picard
include("regularizedPicardB.jl")
include("integralCorrelationKernelFunction.jl")
include("approxCorrelationKernelMatrix.jl")

function estimateGridB(s,n,p,sigma,lambda,epsilon,it_max,tol,n_step_plot)

    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    # uniform sampling in the box [0,1]
    totalSamples = rand(Uniform(0,1), n,1);
    unifSample = collect(1:n); # i.e., the set I

    print("Loading samples...")
    print("\n")

    # loading DPP samples
    id_last_sample = n;
    # gather the samples
    #step = 0.1;
    #temp = collect(0:step:1);
    temp = rand(Uniform(0,1), 10,1);

    print("Number of 'dpp' samples $(length(temp))\n")
    id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
    id_last_sample = id_last_sample + size(temp,1);
    push!(idDppSamples,id_temp);
    totalSamples = [totalSamples; temp];

    x = (totalSamples)'/sigma;


    # construct Gaussian kernel

    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; # makes sure it is positive definite

    ################################
    # solve estimation problem
    ################################
    print("max nb of iteration: $it_max \n")
    print("Picard iteration... \n")

    # iterations
    B, obj, i_stop = regularizedPicardB(K, idDppSamples, unifSample, lambda, it_max ,tol)


    # recover matrix C
    R = cholesky(K).U;
    invR = inv(R);

    B = 0.5*(B+B');# makes sure it is symmetric
    C = invR'*B*invR; 
    C = 0.5*(C+C');# makes sure it is symmetric

    ###################################
    # estimation of correlation kernel
    ###################################
    print("\n")
    print("estimate  correlation kernel...")
    
    c_1 = 0;
    c_2 = 1;
    d = 1;
    K_hat_mat = approxCorrelationKernelMatrix(C,p,c_1,c_2,totalSamples,k,sigma,d);

    print("\n")

    print("plotting...")

    intensity = zeros(Float64, n_step_plot,1);

    t = 0
    v = zeros(Float64, 1,1);
    for i in 1:n_step_plot
        t += 1
        x = (i-1)/(n_step_plot-1);
        v[1] = x;
        intensity[i] = integralCorrelationKernelFunction(v,v,K_hat_mat,totalSamples,k,sigma);
    end   

    # plotting intensity of the estimated process

    x_tics = (0:(1/(n_step_plot-1)):1);
    display(plot(x_tics,
    intensity,
    title="estimated intensity"))

    # construct grid n_step [0, 1]
    n_step = 200;

    X = collect(0:(1/n_step):1);
        
    nb_pts_grid = length(X);
    GramMatrix = zeros(Float64, nb_pts_grid,nb_pts_grid);

    v_i = zeros(Float64, 1,1);
    v_j = zeros(Float64, 1,1);

    for i in 1:nb_pts_grid
        for j in 1:nb_pts_grid
            v_i[1] = X[i];
            v_j[1] = X[j];
            GramMatrix[i,j] = integralCorrelationKernelFunction(v_i,v_j,K_hat_mat,totalSamples,k,sigma);
        end
    end

    return GramMatrix;

end

