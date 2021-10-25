using CSV, DelimitedFiles, DataFrames

"""
    flat_square_2d_grid(n, a, b)

constructs a regular grid in interval [a,b]^2

# Arguments
- `n:Integer`: total number of points sqrtn^2 with sqrtn = floor(sqrt(n))^2)
- `a:Float`: start point of interval [a,b].
- `b:Float`: end point of interval [a,b].

# Output
- `X::Array{Float64,2}`:  nx2 array with coordinates of n grid nodes with [a,b]
position (i,j) -> row = j + sqrtn (i-1) for i,j = 1, ..., sqrtn

"""
function flat_square_2d_grid(n, a, b)

    sqrtn = Int64(floor(sqrt(n)));
    X = zeros(sqrtn*sqrtn, 2);
    counter = 0;
    for i in 1:sqrtn
        for j in 1:sqrtn
            counter += 1;
            X[counter,1] =  a + (b-a)*(i-1)/(sqrtn-1);
            X[counter,2] =  a + (b-a)*(j-1)/(sqrtn-1);
        end
    end

    return X;
end

"""
    add_DPP_samples_to_Fredholm_samples(s,Fredholm_sample,intensity)

Loads DPP samples and concatenate them with Fredholm samples

# Arguments
- `s:Integer`: number of DPP samples.
- `Fredholm_sample:Array`: array containing samples to approximate Fredholm determinant.
- `intensity:Integer`: intensity of the DPP producing the samples (50 or 100).

# Outputs

- `all_sples`: array containing all the concatenated samples.
- `indices_Fredholm_sple`: array containing the indices of the Fredholm samples.
- `indices_DPP_sples`: array of arrays. The l-th entry of the array is an array containing the indices of the l-th DPP sample.

"""
function add_DPP_samples_to_Fredholm_samples(s,Fredholm_sple,intensity)

    # intensity of generating process
    strg_intensity = string(intensity)

    # create an array containing s arrays of DPP samples
    indices_DPP_sples = Array{Int64,1}[];
    n = size(Fredholm_sple,1)
    indices_Fredholm_sple = collect(1:n); # the set I
    all_sples = Fredholm_sple;

    print("Loading DPP samples from: \n")
    id_last_sple = n;
    for i = 0:(s-1)
        # read files and specify that they start from first row
        file_name = "data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_"*strg_intensity*"_nb_"*string(i+1)*".csv"
        print(file_name, " \n")
        sple = CSV.File(file_name; header=true) |> Tables.matrix 
        sple = sple[:,2:3]; # remove first column
        id_sple = collect((id_last_sple+1):(id_last_sple + size(sple,1)));
        id_last_sple = id_last_sple + size(sple,1);
        # add i-th dpp sample indices
        push!(indices_DPP_sples,id_sple);
        # array with all samples
        all_sples = [all_sples; sple];
    end
    print("Done \n")


    return all_sples, indices_Fredholm_sple, indices_DPP_sples
end


"""
    plot_objectives(i_start,i_stop, obj)

Plots the objective functions (obj) from time i_start to time i_stop. No output.

# Arguments
- `i_start::Integer`: index of the starting time
- `i_stop::Integer`: index of the stopping time
- `obj:Array`: array containing the objective values
"""
function plot_objectives(i_start,i_stop, obj)   

    range = i_start:i_stop;
    # plotting
    plt_objectives = plot(range,obj[range],title="objective values",xlabel="number of iterations",ylabel="objective value",legend=false);
    display(plt_objectives)

end


"""
    plot_insample_likelihood(B, R, indices_Fredholm_sple, total_sples)

Plots the in-sample likelihood.

# Arguments
- `B::Array{Float64,2}`: positive definite matrix B solution of the discrete optim problem.
- `R::Array{Float64,2}`: Cholesky factor such that K = R'R.
- `indices_Fredholm_sple:Array{Int64,1}`: indices of samples for approximating Fredholm determinant.
- `total_sples`: array containing all the samples used for representing the solution.

"""
function plot_insample_likelihood(B, R, indices_Fredholm_sple, total_sples)

    diagL = diag(R'*B*R);

    # identify uniform samples
    Fredholm_sples = total_sples[indices_Fredholm_sple,:];
    color_Fredholm_sples = diagL[indices_Fredholm_sple];

    # identify DPP samples
    indices_DPP_sples = setdiff(collect(1:size(total_sples,1)),indices_Fredholm_sple);
    DPP_sples = total_sples[indices_DPP_sples,:];
    color_DPP_sples = diagL[indices_DPP_sples];

    # plot in-sample likelihood
    plt_in_sple_likelihood = scatter(Fredholm_sples[:,1],Fredholm_sples[:,2],zcolor=color_Fredholm_sples,marker = :cross,markersize = 3, title="in-sample likelihood",label="unif",legend=true)
    scatter!(DPP_sples[:,1],DPP_sples[:,2],zcolor=color_DPP_sples,marker = :circle,markersize = 3,legend = false,colorbar = true,framestyle=:box,xtickfont = font(10),ytickfont = font(10),label="DPP")
    display(plt_in_sple_likelihood)
end


"""
    heatmap_intensity_correlation_kernel(GramK,n_side)

Plots a heatmap of the intensity (diagonal of Gram matrix) on a grid.

# Arguments
- `GramK::Array{Float64,2}`: Gram matrix of a kernel on a grid.
- `n_side::Integer`: number of points along each dim of the grid [0:(1/(n_side - 1)):1]^2.

"""
function heatmap_intensity_correlation_kernel(GramK,n_side)

    x_tics = 0:(1/(n_side - 1)):1;
    y_tics = x_tics;
    
    # plotting intensity
    IntensityGramK = reshape(diag(GramK),(n_side,n_side));
    plt_intensity_K = heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10),title= "Intensity of learned DPP ");
    display(plt_intensity_K)

end


"""
    heatmap_slice_Gram_matrix_at_grid_center(GramK,n_side)

Plots a heatmap of a slice of a Gram matrix on a grid.
The slice corresponds to a choice of point in the center of the grid (approximately).

# Arguments
- `GramK::Array{Float64,2}`: Gram matrix of a kernel on a grid.
- `n_side::Integer`: number of points along each dim of the grid [0:(1/(n_side - 1)):1]^2.

"""
function heatmap_slice_Gram_matrix_at_grid_center(GramK,n_side)

        x_tics = 0:(1/(n_side -1)):1;
        y_tics = x_tics;

        ## plotting slice of Gram matrix
        i = floor(n_side/2); 
        j = floor(n_side/2);
        # center of the grid
        id_slice = Int64(j + n_side*(i-1));
        GramK_slice = GramK[:,id_slice];
    
        # reshaping
        GramK_slice_reshaped = reshape(GramK_slice,(n_side,n_side));
        plt = heatmap(x_tics,y_tics,GramK_slice_reshaped,xtickfont = font(10),ytickfont = font(10),title= "correlation kernel for one fixed argument");
        display(plt)
end

"""
    folder_name(s,n,p,sigma,lambda,tol)

outputs a string with a folder name identifying the parameters.

# Arguments
- `s::Integer`: number of DPP samples.
- `n::Integer`: number of Fredholm samples.
- `sigma::Float64': kernel bandwidth.
- `lambda::Float64': regularization parameter.
- `tol::Float64': stopping criterion for optimization algo.

# Outputs
- `foldername::String`: a string with the foldername.

"""
function folder_name(s,n,p,sigma,lambda,tol)

    foldername = "results/result_s="*string(Int64(s))*"_n="*string(Int64(n))*"_p="*string(Int64(p))*"_divideby1e3sigma="*string(Int64(1e3*sigma))*"_divideBy1e6lambda="*string(Int64(1e6*lambda))*"divideBy1e6_tol="*string(Int64(1e6*tol));

    return foldername
end


"""
    save_results_to_jld(B, R, n, total_sples, GramK, GramA, GramK_0, obj, i_stop, foldername)

Save variables into jld file. No output.
"""
function save_results_to_jld(B, R, n, total_sples, GramK, GramA, GramK_0, obj, i_stop, foldername)    

    filename = foldername*"/outputs.jld";

    save(filename, "B",B, "R",R, "n",n, "total_samples",total_sples, "GramK",GramK, "GramA",GramA, "GramK0",GramK_0, "obj",obj, "i_stop",i_stop)

end

"""
    load_results_from_jld(foldername)

Loads variables from jld file.
"""
function load_results_from_jld(foldername)    

    ## load results
    filename = foldername*"/outputs.jld";
    dict = load(filename);

    B =  dict["B"];
    R = dict["R"];
    n = dict["n"];
    total_sples = dict["total_samples"];
    GramK = ["GramK"];
    GramA = dict["GramA"];
    GramK_0 = dict["GramK0"];
    obj =  dict["obj"];
    i_stop = dict["i_stop"];

    return B, R, n, total_sples, GramK, GramA, GramK_0, obj, i_stop
end


"""
    create_repo(s,n,p,sigma,lambda,tol)

creates a repo for storing output data.    

# Arguments
- `s::Integer`: number of DPP samples.
- `n::Integer`: number of Fredholm samples.
- `sigma::Float64': kernel bandwidth.
- `lambda::Float64': regularization parameter.
- `tol::Float64': stopping criterion for optimization algo.

# Outputs
- `foldername::String`: a string with the foldername.
"""
function create_repo(s,n,p,sigma,lambda,tol)
    foldername = folder_name(s,n,p,sigma,lambda,tol);
    if isdir(foldername)==false
        mkdir(foldername);
    end
    return foldername
end
