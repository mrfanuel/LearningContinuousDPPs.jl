using KernelFunctions
using Plots

#include("../algo/estimate_B_from_samples.jl")
include("../algo/kernels.jl")
include("../algo/regularized_Picard.jl")
include("../algo/utils.jl")

function estimate_Gaussian(s,n,sigma,espilon,it_max,tol,p)
    #################### typical parameters #######################
    #### number of DPP samples used for the estimation (from 1 to 10)
    # s = 1;
    #### number of points for approximating Fredholm determinant (the larger the better)
    # n = 500;
    #### Gaussian kernel bandwidth
    # sigma = 0.1;
    #### regularization parameter
    # lambda = 1e-3;
    #### constant added on diagonal of matrices
    # epsilon = 1e-10;
    #### largest number of iterations for Picard
    # it_max = 1000;
    #### relative objective variation desired
    # tol = 1e-5;
    #### number of uniformly sampled points for correlation kernel approximation
    # p = 10000;

    # sample uniformly in [0,1]^2 n points
    Fredholm_sample = rand(Uniform(0,1), n,2);

    # load DPP samples
    total_samples, indices_Fredholm_sample, indices_DPP_samples = add_DPP_samples_to_Fredholm_samples(s,Fredholm_sample)

    # create full kernel matrix
    x = (total_samples)'/sigma;
    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; 

    #####################################################################################################
    ## estimate B: insample likelihood
    #####################################################################################################

    print("regularized Picard ...\n")


    # initial positive definite iterate
    X = randn(size(K));
    B = X*X'+ UniformScaling(1e-10);

    use_inverse = false
    @time begin    
        B, R, obj, i_stop = regularized_Picard(B, K, indices_DPP_samples, indices_Fredholm_sample, lambda, it_max ,tol,use_inverse)
    end

    ## plot objectives

    # skip first iterates
    i_start = 5;
    range = i_start:i_stop;

    # plotting
    plt_objectives = plot(range,obj[range],title="objective decay",xlabel="number of iterations",ylabel="objective value")
    display(plt_objectives)

    ## scatter plot of in-sample likelihood kernel

    # color is diagonal of likelihood matrix
    diagL = diag(R'*B*R);

    # identify uniform samples
    Fredholm_samples = total_samples[indices_Fredholm_sample,:];
    color_Fredholm_samples = diagL[indices_Fredholm_sample];

    #identify DPP samples
    indices_DPP_samples = setdiff(collect(1:size(total_samples,1)),indices_Fredholm_sample)
    DPP_samples = total_samples[indices_DPP_samples,:];
    color_DPP_samples = diagL[indices_DPP_samples]

    # plot in-sample likelihood
    plt_in_sample_likelihood = scatter(Fredholm_samples[:,1],Fredholm_samples[:,2],zcolor=color_Fredholm_samples,marker = :circle,markersize = 3, title="in-sample likelihood",label="unif",legend=true)
    scatter!(DPP_samples[:,1],DPP_samples[:,2],zcolor=color_DPP_samples,marker = :hexagon,markersize = 5,legend = false,colorbar = true,framestyle=:box,xtickfont = font(10),ytickfont = font(10),label="DPP")
    
    display(plt_in_sample_likelihood)

    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on grid
    #####################################################################################################

    # test samples on a grid n_side x n_side in [0,1]^2
    n_side = 30;
    n_test = n_side*n_side; a = 0.; b = 1.;
    test_samples = constructFlatSquareGrid(n_test, a, b);

    # evaluate likelihood kernel on the grid
    GramA = likelihoodKernelGram(B,R,total_samples,test_samples,k,sigma);

    # plotting diagonal of Gram matrix
    IntensityGramA = reshape(diag(GramA),(n_side,n_side));
    x_tics = 0:(1/(n_side-1)):1;
    y_tics = x_tics;
    plot_intensity_A = heatmap(x_tics,y_tics,IntensityGramA,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity A")

    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on the same grid
    #####################################################################################################

    # sampling uniformly p points in window [0,1]^2
    c_1 = 0.; c_2 = 1.;
    unif_samples_correlation = rand(Uniform(c_1,c_2), p,2);
    
    # use these samples to compute correlation kernel
    GramK = correlationKernelGram(B,R,unif_samples_correlation,total_samples,test_samples,k,sigma);

    # plotting
    IntensityGramK = reshape(diag(GramK),(n_side,n_side));
    plot_intensity_K = heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity K")

    display(plot_intensity_A)
    display(plot_intensity_K)

end