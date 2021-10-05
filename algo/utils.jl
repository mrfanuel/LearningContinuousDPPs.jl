using CSV
using DelimitedFiles
using DataFrames


"""
    flat_square_2d_grid(n, a, b)

# Arguments
- `n:Integer`: total number of points sqrtn^2 with sqrtn = floor(sqrt(n))^2)
- `a:Float`: start point of interval [a,b].
- `b:Float`: end point of interval [a,b].

Gives an nx2 array with coordinates of n grid nodes with [a,b]
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

# Arguments
- `s:Integer`: number of DPP samples.
- `Fredholm_sample:Array`: array containing samples to approximate Fredholm determinant.
- `intensity:Integer`: intensity of the DPP producing the samples (50 or 100).

Loads DPP samples and 
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