using LinearAlgebra
using CSV
using KernelFunctions
using Distributions
using DataFrames

include("regularized_Picard.jl")

function estimate_B_from_samples(s,n,sigma,lambda,epsilon,it_max,tol,FredholmSample)

    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    FredholmSampleId = collect(1:n); # the set I

    totalSamples = FredholmSample;
    
    # loading DPP samples
    id_last_sample = n;
    for i = 0:(s-1)
        # read files and specify that they start from first row
        temp = CSV.File("data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_100_nb_"*string(i+1)*".csv"; header=true) |> Tables.matrix 
        temp = temp[:,2:3]; 
        id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
        id_last_sample = id_last_sample + size(temp,1);
        push!(idDppSamples,id_temp);
        totalSamples = [totalSamples; temp];
    end

    x = (totalSamples)'/sigma;
    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; 

    B, R, obj, i_stop = regularized_Picard(K, idDppSamples, FredholmSampleId, lambda, it_max ,tol)
    B = 0.5*(B+B');

    return B, R, k, totalSamples, obj, i_stop;

end

