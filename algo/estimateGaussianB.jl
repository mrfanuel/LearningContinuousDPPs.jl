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

function estimateGaussianB(s,n,p,sigma,lambda,epsilon,it_max,tol,n_step_plot)
    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    # uniform sampling in the box [0,1]^2
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
    K_hat_mat = approxCorrelationKernelMatrix(C,p,c_1,c_2,totalSamples,k,sigma);

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

    # construct grid n_step x n_step within [eps, 1-eps]^2
    n_step = 10;

    X = Array{Float64,2}[];
    eps = 0.1;
    t = 0
    for i in 1:n_step
        for j in 1:n_step
            t += 1
            x =  eps + (1-eps)*(i-1)/(n_step-1);
            y =  eps + (1-eps)*(j-1)/(n_step-1);
            v = [x y];
            push!(X,v);
        end
    end
        
    nb_pts_grid = n_step*n_step;
    GramMatrix = zeros(Float64, nb_pts_grid,nb_pts_grid);

    for i in 1:nb_pts_grid
        for j in 1:nb_pts_grid
            v_i = X[i,:][1]';
            v_j = X[j,:][1]';
            GramMatrix[i,j] = integralCorrelationKernelFunction(v_i,v_j,K_hat_mat,totalSamples,k,sigma);
        end
    end

    return GramMatrix;

end

