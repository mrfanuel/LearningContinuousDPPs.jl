using KernelFunctions
using Plots

include("../algo/kernels.jl")
include("../algo/regularized_Picard.jl")
include("../algo/utils.jl")

function estimate_Gaussian(intensity::Int64,s::Int64,n::Int64,sigma::Float64,lambda::Float64,it_max::Int64,tol::Float64,p::Int64)
    #################### typical parameters ######################
    #### number of DPP samples used for the estimation (from 1 to 10)
    # s = 1;
    #### number of points for approximating Fredholm determinant (the larger the better)
    # n = 1000;
    #### Gaussian kernel bandwidth
    # sigma = 0.1;
    #### regularization parameter
    # lambda = 1e-1;
    #### largest number of iterations for Picard
    # it_max = 100000;
    #### relative objective variation desired
    # tol = 1e-5;
    #### number of uniformly sampled points for correlation kernel approximation (the larger the better)
    # p = 10000

    # constant added on diagonal of matrices
    epsilon = 1e-10;

    # sample uniformly in [0,1]^2 n points
    Fredholm_sample = rand(Uniform(0,1), n,2);

    # load DPP samples
    total_samples, indices_Fredholm_sample, indices_DPP_samples = add_DPP_samples_to_Fredholm_samples(s,Fredholm_sample,intensity)

    # create full kernel matrix
    x = (total_samples)'/sigma;
    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; 

    # show value on a line
    center = [0 0];
    direction = [1 0];
    nb_points = 101; # odd number
    line_at_center, id_center = line(center,direction,nb_points);

    #####################################################################################################
    ## estimate B: insample likelihood
    #####################################################################################################

    print("Solving discrete problem with regularized Picard ...\n")


    # initial positive definite iterate
    X = randn(size(K));
    B = X*X'+ UniformScaling(epsilon);

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
    plt_in_sample_likelihood = scatter(Fredholm_samples[:,1],Fredholm_samples[:,2],zcolor=color_Fredholm_samples,marker = :cross,markersize = 3, title="in-sample likelihood",label="unif",legend=true)
    scatter!(DPP_samples[:,1],DPP_samples[:,2],zcolor=color_DPP_samples,marker = :circle,markersize = 3,legend = false,colorbar = true,framestyle=:box,xtickfont = font(10),ytickfont = font(10),label="DPP")
    
    display(plt_in_sample_likelihood)

    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on grid
    #####################################################################################################

    # test samples on a grid n_side x n_side in [0,1]^2
    n_side = 100;
    n_test = n_side*n_side; a = 0.; b = 1.;
    test_samples = flat_square_grid(n_test, a, b);

    # evaluate likelihood kernel on the grid
    GramA = likelihood_kernel_Gram(B,R,total_samples,test_samples,k,sigma);

    # plotting diagonal of Gram matrix
    IntensityGramA = reshape(diag(GramA),(n_side,n_side));
    x_tics = 0:(1/(n_side-1)):1;
    y_tics = x_tics;
    plt_intensity_A = heatmap(x_tics,y_tics,IntensityGramA,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity A")

    GramA_slice = GramA[:,5000];
    plt_GramA_slice = plot(1:(n_side*n_side),GramA_slice,title= "slice of GramA",marker = :circle,markersize = 1);

    display(plt_intensity_A)
    display(plt_GramA_slice)



    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on the same grid
    #####################################################################################################

    # sampling uniformly p points in window [0,1]^2
    c_1 = 0.; c_2 = 1.;
    unif_samples_correlation = rand(Uniform(c_1,c_2), p,2);
    
    # use these samples to compute correlation kernel
    GramK = correlation_kernel_Gram(B,R,unif_samples_correlation,total_samples,test_samples,k,sigma);

    # plotting
    IntensityGramK = reshape(diag(GramK),(n_side,n_side));
    plt_intensity_K = heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity K")

    display(plt_intensity_K)

    GramK_slice = GramK[:,5000];
    plt_GramK_slice = scatter(1:(n_side*n_side),GramK_slice,title= "slice of GramA",marker = :circle,markersize = 1);

    display(plt_GramK_slice)
    
    # show value on a line
    center = [0;0];
    direction = [1;0];
    nb_points = 101; # odd number
    line_at_center, id_center = line(center,direction,nb_points);

    GramK_line = correlation_kernel_Gram(B,R,unif_samples_correlation,total_samples,line,k,sigma);
    kernel_value = GramK_line[id_center,:];
    plt_line = plot(line_at_center, kernel_value,title= "slice of correlation kernel",marker = :circle,markersize = 1)
    display(plt_line)


end