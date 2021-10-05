using KernelFunctions
using Plots


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
    Fredholm_sample = rand(Uniform(0,1), n,2);

    # load DPP samples
    total_samples, indices_Fredholm_sample, indices_DPP_samples = add_DPP_samples_to_Fredholm_samples(s,Fredholm_sample,intensity)

    # create full kernel matrix
    x = (total_samples)'/sigma;
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
        B, R, obj, i_stop = regularized_Picard(B, K, indices_DPP_samples, indices_Fredholm_sample, lambda, it_max ,tol,use_inverse)
    end

    #####################################################################################################
    ## plot objectives
    #####################################################################################################
    print("Plotting ...\n")
    # skip first iterates
    i_start = 5;
    range = i_start:i_stop;

    # plotting
    plt_objectives = plot(range,obj[range],title="objective decay",xlabel="number of iterations",ylabel="objective value",legend=false)
    display(plt_objectives)

    #####################################################################################################
    ## scatter plot of in-sample likelihood kernel
    #####################################################################################################
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
    test_samples = flat_square_2d_grid(n_test, a, b);

    # evaluate likelihood kernel on the grid
    GramA = likelihood_kernel_Gram(B,R,total_samples,test_samples,k,sigma);

    # plotting diagonal of Gram matrix
    IntensityGramA = reshape(diag(GramA),(n_side,n_side));
    x_tics = 0:(1/(n_side-1)):1;
    y_tics = x_tics;
    plt_intensity_A = heatmap(x_tics,y_tics,IntensityGramA,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "likelihood intensity")

    # plotting slice of Gram matrix
    id_slice = Int64(floor(n_side^2/2))+20;
    GramA_slice = GramA[:,id_slice];
    plt_GramA_slice = plot(1:(n_side*n_side),GramA_slice,title= "slice of likelihood Gram matrix",marker = :circle,markersize = 1,legend = false);
    display(plt_intensity_A)
    display(plt_GramA_slice)

    GramA_slice_reshaped = reshape(GramA_slice,(n_side,n_side));

    plt = heatmap(x_tics,y_tics,GramA_slice_reshaped,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "likelihood kernel for one fixed argument")
    display(plt)
    plt = plot3d(x_tics,y_tics,GramA_slice_reshaped,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "likelihood kernel for one fixed argument")
    display(plt)



    #####################################################################################################
    ## estimate out-of-sample likelihood kernel on the same grid
    #####################################################################################################

    # sampling uniformly p points in window [0,1]^2
    c_1 = 0.; c_2 = 1.;
    unif_samples_correlation = rand(Uniform(c_1,c_2), p,2);
    
    # use these samples to compute correlation kernel
    GramK = correlation_kernel_Gram(B,R,unif_samples_correlation,total_samples,test_samples,k,sigma);

    # plotting intensity
    IntensityGramK = reshape(diag(GramK),(n_side,n_side));
    plt_intensity_K = heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity of learned DPP ")

    display(plt_intensity_K)

    # plotting slice of Gram matrix
    GramK_slice = GramK[:,id_slice];

    plt_GramK_slice = plot(1:(n_side*n_side),GramK_slice,title= "slice of correlation Gram matrix",marker = :circle,markersize = 1,legend = false);
    display(plt_GramK_slice)


    GramK_slice_reshaped = reshape(GramK_slice,(n_side,n_side));

    plt = heatmap(x_tics,y_tics,GramK_slice_reshaped,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "correlation kernel for one fixed argument")
    display(plt)

    plt = plot3d(x_tics,y_tics,GramK_slice_reshaped,xtickfont = font(10),ytickfont = font(10),title= "correlation kernel for one fixed argument")
    display(plt)

    # plotting one g_2 (normalized pair correlation function)

    g_2 = (GramK_slice_reshaped./IntensityGramK)/IntensityGramK[id_slice];

    plt = plot3d(x_tics,y_tics,g_2,xtickfont = font(10),ytickfont = font(10),title= "normalized pair correlation function")
    display(plt)

    plt = heatmap(x_tics,y_tics,g_2,xtickfont = font(10),ytickfont = font(10),title= "normalized pair correlation function")
    display(plt)

    # show value of correlation kernel on a line
    center = [0.5 0.5];
    dir = [1 1];
    dir = 0.5*dir/norm(dir);
    odd_number_pts = 401;

    line_at_center, pos_along_dir, id_center = line(center,dir,odd_number_pts);

    GramK_line = correlation_kernel_Gram(B,R,unif_samples_correlation,total_samples,line_at_center,k,sigma);
    kernel_value = vec(GramK_line[id_center,:]);

    plt_value_line = plot(pos_along_dir,kernel_value,title= "correlation kernel along segment",marker = :circle,markersize = 1,legend = false)
    display(plt_value_line)

end