# import utilities
using LinearAlgebra
using CSV
using KernelFunctions
using Distributions
using DataFrames

# algo for regularized Picard
include("regularizedPicardB.jl")


function estimateGaussianB(s,n,sigma,lambda,epsilon,it_max,tol,FredholmSample, merge)
    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    FredholmSampleId = collect(1:n); # i.e., the set I

    totalSamples = FredholmSample;
    print("Loading samples...")
    print("\n")
    print("\nFredholmSample has $(size(FredholmSample,1)) points\n")
    print("DPP samples:\n")
    
    # loading DPP samples
    id_last_sample = n;
    # gather the samples
    for i = 0:(s-1)
        # read files and specify that they start from first row
        temp = CSV.File("data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_50_nb_"*string(i+1)*".csv"; header=true) |> Tables.matrix 
        temp = temp[:,2:3];
        print("\nsample $(i+1) has $(size(temp,1)) points\n")
        # temp is a matrix with 2 cols
        id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
        id_last_sample = id_last_sample + size(temp,1);
        push!(idDppSamples,id_temp);
        totalSamples = [totalSamples; temp];
    end
    print("\nIn total: $(size(totalSamples,1)) points\n")

    if merge
        FredholmSampleId = collect(1:size(totalSamples,1));
    end

    x = (totalSamples)'/sigma;

    #print("kernel type: ")
    #if kernel=="MaternKernel"
    #    k = MaternKernel(;nu);
    #    print("Matern kernel \n")
    #else kernel=="SqExponentialKernel"
    #    k = SqExponentialKernel();
    #    print("SqExponential kernel \n")
    #end

    k = SqExponentialKernel();

    K = kernelmatrix(k, x) + epsilon *I ; # makes sure it is positive definite

    print("size(K) = $(size(K))\n")

    print("Picard iteration... \n")

    ################################
    # solve estimation problem
    ################################
    print("max nb of iteration: $it_max \n")

    # iterations
    B, R, obj, i_stop = regularizedPicardB(K, idDppSamples, FredholmSampleId, lambda, it_max ,tol)

    B = 0.5*(B+B');# makes sure it is symmetric


    return B, R, K, k, totalSamples,obj,i_stop;

end

