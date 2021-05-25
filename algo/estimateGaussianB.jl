# import utilities
using LinearAlgebra
using Plots
using CSV
using KernelFunctions
using Distributions
using DataFrames

# algo for regularized Picard
include("regularizedPicardB.jl")


function estimateGaussianB(s,n,kernel,nu,sigma,lambda,epsilon,it_max,tol,FredholmSample, merge)
    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    FredholmSampleId = collect(1:n); # i.e., the set I

    totalSamples = FredholmSample;

    print("Loading samples...")
    print("\n")

    # loading DPP samples
    id_last_sample = n;
    # gather the samples
    for i = 0:(s-1)
        # read files and specify that they start from first row
        temp = CSV.File("data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_50_nb_"*string(i+1)*".csv"; header=true) |> Tables.matrix 
        temp = temp[:,2:3];
        # temp is a matrix with 2 cols
        # add it as a new entry in matSamples
        id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
        id_last_sample = id_last_sample + size(temp,1);
        push!(idDppSamples,id_temp);
        totalSamples = [totalSamples; temp];
    end

    if merge
        FredholmSampleId = collect(1:size(totalSamples,1));
    end

    x = (totalSamples)'/sigma;


    print("kernel type: ")
    if kernel=="MaternKernel"
        k = MaternKernel(;nu);
        print("Matern kernel \n")
    else kernel=="SqExponentialKernel"
        k = SqExponentialKernel();
        print("SqExponential kernel \n")
    end
    K = kernelmatrix(k, x) + epsilon *I ; # makes sure it is positive definite

    ################################
    # solve estimation problem
    ################################
    print("max nb of iteration: $it_max \n")
    print("Picard iteration... \n")

    # iterations
    B, obj, i_stop = regularizedPicardB(K, idDppSamples, FredholmSampleId, lambda, it_max ,tol)
    B = 0.5*(B+B');# makes sure it is symmetric


    return B,K, k, totalSamples;

end

