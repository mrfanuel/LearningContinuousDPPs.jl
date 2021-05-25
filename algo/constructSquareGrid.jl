
function constructSquareGrid(n_step, a, b)

    # construct grid n_step x n_step within [a, b]^2
    X = Array{Float64,2}[];
    for i in 1:n_step
        for j in 1:n_step
            x =  a + (b-a)*(i-1)/(n_step-1);
            y =  a + (b-a)*(j-1)/(n_step-1);
            v = [x y];
            push!(X,v);
        end
    end

    return X;
end

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