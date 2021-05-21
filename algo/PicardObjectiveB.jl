
function PicardObjectiveB(B, samples, FredholmSample, R,lambda)

    # identity matrix
    m = size(B,1);
    identity = Diagonal(vec(ones(m,1)));

    # number of dpp samples
    nb_samples = length(samples); 

    # samples Fredholm
    unifU = identity[:,FredholmSample];
    nb_unif = length(FredholmSample)

    PhiBPhi = R'*B*R;
    ob = 0
    for l = 1:nb_samples
        id = samples[l];
        U = identity[:,id];
        ob = ob - logdet(U'*PhiBPhi*U);
        if ob==Inf
            error("singular determinant in objective")
        end
    end
    #print("ob $(ob) \n" )
    ob = ob/nb_samples;
    ob = ob+logdet(I + (1/nb_unif)*unifU'*PhiBPhi*unifU);
    ob_reg = lambda*tr(B);

    return ob, ob_reg;
end