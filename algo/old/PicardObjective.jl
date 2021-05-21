function PicardObjective(X, samples, FredholmSample, Rinv,lambda)

    # identity matrix
    m = size(invK,1);
    identity = Diagonal(vec(ones(m,1)));

    # number of dpp samples
    nb_samples = length(samples); 

    # samples Fredholm
    unifU = identity[:,FredholmSample];
    nb_unif = length(FredholmSample)


    ob = 0
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        ob = ob - logdet(U'*X*U);
        if ob==Inf
            error("singular determinant in objective")
        end
    end
    #print("ob $(ob) \n" )
    ob = ob/nb_samples;
    ob = ob+logdet(I + (1/nb_unif)*unifU'*X*unifU);
    ob_reg = lambda*tr(Rinv'*X*Rinv);

    return ob, ob_reg;
end