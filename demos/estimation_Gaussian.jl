using KernelFunctions,Plots,JLD

include("../algo/kernels.jl")
include("../algo/regularized_Picard.jl")
include("../algo/utils.jl")

"""
    estimate_Gaussian(s,n,sigma,lambda,tol,intensity,it_max,p)

Estimate likelihood and correlation kernels and plots several figures. No output.

# Arguments
- `s::Integer`: the number of Gaussian-DPP samples used for estimation (up to 10).
- `n:Integer`: the number of uniformly sampled point for approximating the Fredholm determinant
- `sigma:Float`: RKHS kernel bandwidth
- `lambda:Float`: regularization parameter
- `tol:Float`: relative objective variation desired for regularized Picard iteration
- `intensity:Integer=100`: choose Gaussian-DPP data-samples with intensity 100 (or 50)
- `it_max:Integer=100000`: largest number of iterations for Picard
- `p:Integer= 10000`: the number of uniformly sampled point for approximating correlation kernel

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
    
    X = randn(size(K));
    B = X*X'+ UniformScaling(epsilon);# initial positive definite B

    use_inverse = false
    @time begin    
        B, R, obj, i_stop = regularized_Picard(B, K, indices_DPP_sples, indices_Fredholm_sple, lambda, it_max ,tol,use_inverse)
    end

    # create repo for storing results
    foldername = create_repo(s,n,p,sigma,lambda,tol);

    print("Plotting ...\n")

    #####################################################################################################
    ## plot objectives
    #####################################################################################################
    i_start = 5; # skip first iterates
    plot_objectives(i_start,i_stop, obj);
    savefig(foldername*"/objectives.pdf")

    #####################################################################################################
    ## scatter plot of in-sample likelihood kernel
    #####################################################################################################
    
    # color is diagonal of likelihood matrix
    plot_insample_likelihood(B, R, indices_Fredholm_sple, total_sples);
    savefig(foldername*"/in-sample_likelihood.pdf")

    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on grid
    #####################################################################################################

    # test samples on a grid n_side x n_side in [a,b]^2
    n_side = 30;
    a = 0.; b = 1.; 
    n_test = n_side^2; 

    test_sples = flat_square_2d_grid(n_test, a, b);

    # evaluate likelihood kernel on the grid
    GramA = likelihood_kernel_Gram(B,R,total_sples,test_sples,k,sigma);

    display(heatmap(GramA,title="Gram matrix of estimated likelihood kernel"));
    savefig(foldername*"/Gram_likelihood.pdf")

    #####################################################################################################
    ## estimate out-of-sample correlation kernel on the same grid
    #####################################################################################################

    ########## Approximate correlation kernel ##########

    # sampling uniformly p points in window [0,1]^2
    c_1 = 0.; c_2 = 1.;
    unif_sples_correlation = rand(Uniform(c_1,c_2), p,2);
    
    # use these samples to compute correlation kernel
    GramK = correlation_kernel_Gram(B,R,unif_sples_correlation,total_sples,test_sples,k,sigma);

    display(heatmap(GramK,title="Gram matrix of estimated correlation kernel"));
    savefig(foldername*"/Gram_correlation.pdf")

    heatmap_intensity_correlation_kernel(GramK,n_side)
    savefig(foldername*"/intensity_correlation.pdf")


    # select one point x_0 almost at center of grid and display heatmap of k(x,x_0)
    heatmap_slice_Gram_matrix_at_grid_center(GramK,n_side);
    savefig(foldername*"/slice_correlation.pdf")

    ########## Exact correlation kernel ##########

    alpha_0 = 0.05;
    rho_0 = 100;
    x_test_0 = (test_sples)'/(alpha_0/sqrt(2)); 
    # NB: srt(2) to match the kernel definition in genGaussianSpatstats.R

    k_0 = SqExponentialKernel();
    GramK_0 = rho_0 * kernelmatrix(k_0, x_test_0) + epsilon *I ; # positive definite

    # display heatmap of GramK_0
    display(heatmap(GramK_0,title="Gram matrix of exact correlation kernel"));
    savefig(foldername*"/Gram_correlation_exact.pdf")

    #####################################################################################################
    ## save outputs 
    #####################################################################################################

    save_results_to_jld(B, R, n, total_sples, GramK, GramA, GramK_0, obj, i_stop, foldername)    ;

end