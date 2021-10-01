

function constructFlatSquareGrid(n, a, b)

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


function add_DPP_samples_to_Fredholm_samples(s,Fredholm_sample)


    # create an array of arrays
    indices_DPP_samples = Array{Int64,1}[];
    n = size(Fredholm_sample,1)
    indices_Fredholm_sample = collect(1:n); # the set I

    total_samples = Fredholm_sample;

    # loading DPP samples
    id_last_sample = n;
    for i = 0:(s-1)
        # read files and specify that they start from first row
        temp = CSV.File("data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_100_nb_"*string(i+1)*".csv"; header=true) |> Tables.matrix 
        temp = temp[:,2:3]; 
        id_temp = collect((id_last_sample+1):(id_last_sample + size(temp,1)));
        id_last_sample = id_last_sample + size(temp,1);
        push!(indices_DPP_samples,id_temp);
        total_samples = [total_samples; temp];
    end

    return total_samples, indices_Fredholm_sample, indices_DPP_samples
end