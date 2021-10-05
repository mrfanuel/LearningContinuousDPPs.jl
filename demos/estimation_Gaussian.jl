using KernelFunctions
using Plots
using JLD

include("../algo/kernels.jl")
include("../algo/regularized_Picard.jl")
include("../algo/utils.jl")

"""
    estimate_Gaussian(s,n,sigma,lambda,tol,intensity,it_max,p)

# Arguments
- `s::Integer`: the number of Gaussian-DPP samples used for estimation (up to 10).
- `n:Integer`: the number of uniformly sampled point for approximating the Fredholm determinant
- `sigma:Float`: RKHS kernel bandwidth
- `lambda:Float`: regularization parameter
- `tol:Float`: relative objective variation desired for regularized Picard iteration
- `intensity:Integer=100`: choose Gaussian-DPP data-samples with intensity 100 (or 50)
- `it_max:Integer=100000`: largest number of iterations for Picard
- `p:Integer= 10000`: the number of uniformly sampled point for approximating correlation kernel

Estimate likelihood and correlation kernels and plots several figures
"""
function estimate_Gaussian(s::Int64,n::Int64,sigma::Float64,lambda::Float64,tol::Float64,intensity::Int64=100,it_max::Int64=100000,p::Int64 = 10000)

    # constant added on diagonal of matrices
    epsilon = 1e-10;

    # sample uniformly in [0,1]^2 n points
    Fredholm_sple = rand(Uniform(0,1), n,2);

    # load DPP samples
    total_sples, indices_Fredholm_sple, indices_DPP_sples = add_DPP_samples_to_Fredholm_samples(s,Fredholm_sple,intensity)

    # create full kernel matrix
    x = (total_sples)'/sigma;
    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; 

    #####################################################################################################
    ## estimate B: insample likelihood
    #####################################################################################################

    print("Solving discrete problem with regularized Picard ...\n")
    # initial positive definite B
    X = randn(size(K));
    B = X*X'+ UniformScaling(epsilon);

    use_inverse = false
    @time begin    
        B, R, obj, i_stop = regularized_Picard(B, K, indices_DPP_sples, indices_Fredholm_sple, lambda, it_max ,tol,use_inverse)
    end

    #####################################################################################################
    ## plot objectives
    #####################################################################################################
    print("Plotting ...\n")
    i_start = 5; # skip first iterates
    range = i_start:i_stop;

    # plotting
    plt_objectives = plot(range,obj[range],title="objective values",xlabel="number of iterations",ylabel="objective value",legend=false)
    display(plt_objectives)

    #####################################################################################################
    ## scatter plot of in-sample likelihood kernel
    #####################################################################################################
    # color is diagonal of likelihood matrix
    diagL = diag(R'*B*R);

    # identify uniform samples
    Fredholm_sples = total_sples[indices_Fredholm_sple,:];
    color_Fredholm_sples = diagL[indices_Fredholm_sple];

    # identify DPP samples
    indices_DPP_sples = setdiff(collect(1:size(total_sples,1)),indices_Fredholm_sple)
    DPP_sples = total_sples[indices_DPP_sples,:];
    color_DPP_sples = diagL[indices_DPP_sples]

    # plot in-sample likelihood
    plt_in_sple_likelihood = scatter(Fredholm_sples[:,1],Fredholm_sples[:,2],zcolor=color_Fredholm_sples,marker = :cross,markersize = 3, title="in-sample likelihood",label="unif",legend=true)
    scatter!(DPP_sples[:,1],DPP_sples[:,2],zcolor=color_DPP_sples,marker = :circle,markersize = 3,legend = false,colorbar = true,framestyle=:box,xtickfont = font(10),ytickfont = font(10),label="DPP")
    display(plt_in_sple_likelihood)

    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on grid
    #####################################################################################################

    # test samples on a grid n_side x n_side in [0,1]^2
    n_side = 30;
    x_tics = 0:(1/(n_side-1)):1;
    y_tics = x_tics;
    n_test = n_side^2; 

    a = 0.; b = 1.; # interval [a,b]
    test_sples = flat_square_2d_grid(n_test, a, b);

    # evaluate likelihood kernel on the grid
    GramA = likelihood_kernel_Gram(B,R,total_sples,test_sples,k,sigma);

    # plotting slice of Gram matrix
    id_slice = Int64(floor(n_side^2/2))+20;
    GramA_slice = GramA[:,id_slice];
    # reshaping
    GramA_slice_reshaped = reshape(GramA_slice,(n_side,n_side));
    plt = heatmap(x_tics,y_tics,GramA_slice_reshaped,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "learned likelihood kernel for one fixed argument");display(plt)

    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on the same grid
    #####################################################################################################

    # sampling uniformly p points in window [0,1]^2
    c_1 = 0.; c_2 = 1.;
    unif_sples_correlation = rand(Uniform(c_1,c_2), p,2);
    
    # use these samples to compute correlation kernel
    GramK = correlation_kernel_Gram(B,R,unif_sples_correlation,total_sples,test_sples,k,sigma);

    # for plotting heatmap of Gram matrix do:
    display(heatmap(GramK,title="Gram matrix of estimated correlation kernel"));
    # beware: it is expensive for large grids
    
    # plotting intensity
    IntensityGramK = reshape(diag(GramK),(n_side,n_side));
    plt_intensity_K = heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity of learned DPP ");display(plt_intensity_K)

    ## plotting slice of Gram matrix
    i = floor(n_side/2); 
    j = floor(n_side/2);
    # center of the grid
    id_slice = Int64(j + n_side*(i-1));
    GramK_slice = GramK[:,id_slice];
    # reshaping
    GramK_slice_reshaped = reshape(GramK_slice,(n_side,n_side));
    plt = heatmap(x_tics,y_tics,GramK_slice_reshaped,xtickfont = font(10),ytickfont = font(10),title= "correlation kernel for one fixed argument");display(plt)

    # exact correlation kernel
    alpha_0 = 0.05;
    rho_0 = 100;
    x_test_0 = (test_sples)'/(alpha_0/sqrt(2)); 
    # NB: srt(2) to match the kernel definition in genGaussianSpatstats.R
    k_0 = SqExponentialKernel();

    GramK_0 = rho_0 * kernelmatrix(k_0, x_test_0) + epsilon *I ; # positive definite

    # for plotting heatmap of Gram matrix do:
    display(heatmap(GramK_0,title="Gram matrix of exact correlation kernel"));
    # beware: it is expensive for large grids

    # take a slice
    GramK_0_slice = GramK_0[:,id_slice];
    # reshaping
    GramK_0_slice_reshaped = reshape(GramK_0_slice,(n_side,n_side));
    plt = heatmap(x_tics,y_tics,GramK_0_slice_reshaped,xtickfont = font(10),ytickfont = font(10),title= "exact correlation kernel for one fixed argument");display(plt)

    # plotting one g_2 (normalized pair correlation function)
    g_2 = (GramK_slice_reshaped./IntensityGramK)/IntensityGramK[id_slice];

    plt = heatmap(x_tics,y_tics,g_2,xtickfont = font(10),ytickfont = font(10),title= "estimated normalized pair correlation function");display(plt)

    #####################################################################################################
    ## save results
    #####################################################################################################

    filename = "results/result_s="*string(Int64(s))*"_n="*string(Int64(n))*"_p="*string(Int64(p))*"_divideby1e3sigma="*string(Int64(1e3*sigma))*"_divideBy1e6lambda="*string(Int64(1e6*lambda))*"divideBy1e6_tol="*string(Int64(1e6*tol))*".jld";

    save(filename, "B",B, "R",R, "n",n, "total_samples",total_sples, "GramK",GramK, "GramA",GramA, "GramK0",GramK_0, "obj",obj, "i_stop",i_stop);

    # for loading results do as follows:
    # dict = load("filename")
    # e.g. GramK = dict["GramK"]

end