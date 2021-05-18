# import utilities
using LinearAlgebra
using Plots
using CSV
using KernelFunctions
using Distributions
using DataFrames

# algo for regularized Picard
include("regularizedPicard.jl")
include("integralCorrelationKernelFunction.jl")
include("approxCorrelationKernelMatrix.jl")

function estimateDPP(s::Int64,n::Int64,p::Int64,kernel::String,nu::Float64,sigma::Float64,lambda::Float64,epsilon::Float64,it_max::Int64,tol::Float64,n_step_plot::Int64)
    ######################################################################
    # uniform sampling for approx. Fedholm det and loading DPP samples 
    ######################################################################

    # create an array of arrays
    idDppSamples = Array{Int64,1}[];

    # uniform sampling in the box [-1,1]^2
    c = 1.;
    totalSamples = rand(Uniform(-c,c), n,2);
    unifSample = collect(1:n); # i.e., the set I

    ## loading DPP samples
    id_last_sample = n;
    # gather the samples
    for i = 0:(s-1)
        # read files and specify that they start from first row
        temp = CSV.File("data/sample"*string(i)*".csv"; header=false) |> Tables.matrix 
        # temp is a matrix with 2 cols
        # add it as a new entry in matSamples
        id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
        id_last_sample = id_last_sample + size(temp,1);
        push!(idDppSamples,id_temp);
        totalSamples = [totalSamples; temp];
    end

    # construct Gaussian kernel

    print("kernel type: ")

    x = (totalSamples)'/sigma;
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
    # for plotting
    intensity = zeros(Float64, n_step_plot,n_step_plot);

    t = 0
    for i in 1:n_step_plot
        for j in 1:n_step_plot
            t += 1
            x = -1 + 2*(i-1)/(n_step_plot-1);
            y = -1 + 2*(j-1)/(n_step_plot-1);
            v = [x y]';
            #K_nv=kernelmatrix(k, x_n, v);
            #intensity[i,j] = ((K_nv)'*K_hat_mat*K_nv)[1];
            intensity[i,j] = integralCorrelationKernelFunction(v,v,K_hat_mat,totalSamples,k,sigma);
        end
    end

    # to check positive definiteness
    #eig_mat_intensity = eigvals(intensity);
    #print("eigenvalues empirical correlation matrix...")
    #print(eig_mat_intensity)
    

    # plotting intensity of the estimated process

    #gr()
    x_tics = (-1:2*(1/(n_step_plot-1)):1);
    y_tics = x_tics;
    display(heatmap(x_tics,
    y_tics,
    intensity,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="x",
    ylabel="y",
    title="estimated intensity"))

    
    for i = 0:(s-1)
        temp = CSV.File("data/sample"*string(i)*"lambda"*string(lambda)*".csv"; header=false) |> Tables.matrix 
        display(plot!(temp[:,1], temp[:,2], seriestype = :scatter, legend = false))
    end

    savefig("figures/MVJacobi_plot_"*string(s)*"_dppSamples_50_points.pdf")
    # plotting objectives
    # start plotting from it number 10 to have a nicer plot
    #display(plot(10:i_stop, obj[10:i_stop], xlabel = "iteration", ylabel = "objective value",
    #title="convergence regularized Picard"))

    return 1;
end 