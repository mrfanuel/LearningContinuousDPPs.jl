using CSV
using DelimitedFiles
using DataFrames


"""
    flat_square_2d_grid(n, a, b)

# Arguments
- `n:Integer`: total number of points (perfect square, otherwise floor(sqrt(n))^2)
- `a:Float`: start point of interval [a,b].
- `b:Float`: end point of interval [a,b].

Gives an nx2 array with coordinates of n grid nodes with [a,b]
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

# Arguments
- `s:Integer`: number of DPP samples.
- `Fredholm_sample:Array`: array containing samples to approximate Fredholm determinant.
- `intensity:Integer`: intensity of the DPP producing the samples (50 or 100).

Loads DPP samples and 
"""
function add_DPP_samples_to_Fredholm_samples(s,Fredholm_sample,intensity)

    # intensity of generating process
    strg_intensity = string(intensity)

    # create an array containing s arrays of DPP samples
    indices_DPP_samples = Array{Int64,1}[];
    n = size(Fredholm_sample,1)
    indices_Fredholm_sample = collect(1:n); # the set I
    all_samples = Fredholm_sample;

    print("Loading DPP samples from: \n")
    id_last_sample = n;
    for i = 0:(s-1)
        # read files and specify that they start from first row
        file_name = "data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_"*strg_intensity*"_nb_"*string(i+1)*".csv"
        print(file_name, " \n")
        sample = CSV.File(file_name; header=true) |> Tables.matrix 
        sample = sample[:,2:3]; # remove first column
        id_sample = collect((id_last_sample+1):(id_last_sample + size(sample,1)));
        id_last_sample = id_last_sample + size(sample,1);
        # add i-th dpp sample indices
        push!(indices_DPP_samples,id_sample);
        # array with all samples
        all_samples = [all_samples; sample];
    end
    print("Done \n")


    return all_samples, indices_Fredholm_sample, indices_DPP_samples
end