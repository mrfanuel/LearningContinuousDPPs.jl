
using SparseArrays, LinearAlgebra
function regularized_Picard(B::Array{Float64,2},K::Array{Float64,2}, dpp_samples::Array{Array{Int64,1},1}, unif_sample::Array{Int64,1}, lambda::Float64, it_max::Int64 ,tol::Float64,use_inverse::Bool)

# number of dpp_samples
nb_dpp_samples = length(dpp_samples); 

m = size(K,1);

# Chol decomposition
R = cholesky(K).U;

# sampling matrix for uniformSample
nb_unif = length(unif_sample)

# initialization
obj = zeros(it_max,1);
i_stop = it_max;


# iterations
for i in 1:it_max
    # construct  Delta
    BRDeltaRtB = zeros(m,m);
    RtBR = R'*B*R
    BR = B*R
    for l = 1:nb_dpp_samples
        id = dpp_samples[l];
        M = (RtBR + UniformScaling(epsilon))[id,id]
        BRid = BR[:,id]
        if use_inverse
            BRDeltaRtB +=  BRid * inv(M) * BRid';
        else
            BRDeltaRtB +=  BRid * (M\(BRid'));
        end
    end

    N = (RtBR + UniformScaling(nb_unif))[unif_sample,unif_sample]
    BRunif = BR[:,unif_sample]
    if use_inverse
        BRDeltaRtB = BRDeltaRtB/nb_dpp_samples - BRunif*inv(N)*BRunif';
    else
        BRDeltaRtB = BRDeltaRtB/nb_dpp_samples - BRunif*(N\(BRunif'));
    end

    BRDeltaRtB = 0.5*(BRDeltaRtB + BRDeltaRtB');

    # Picard iteration
    pB = B + BRDeltaRtB;

    # final expression
    B = (0.5/lambda)*(real(sqrt(I+4*lambda*pB))-I);
    B = 0.5*(B+B');

    # track the objective values
    obj_det,ob_reg = PicardObjectiveB(B, dpp_samples, unif_sample, R,lambda);
    obj[i] = obj_det + ob_reg;

    

    if i%100 == 0
        rel_variation = abs(obj[i]-obj[i-1])/abs(obj[i])
        print("---------------------------------------------------------------\n")
        print("$(i) / $it_max\n")
        print("relative objective variation $(rel_variation)\n")
        print("objective = $(obj[i]) \n")
        print("norm(B) = $(norm(B))\n")

    end
    # stopping criterion
    
    if i>1 && abs(obj[i]-obj[i-1])/abs(obj[i])< tol
        i_stop = i;
        print("---------------------------------------------------------------\n")
        print("Relative tolerance $(tol) attained after $(i) iterations.\n")
        print("Final objective= $(obj[i])\n")
        print("---------------------------------------------------------------\n")
        break
    end
    if i==it_max
        print("iteration has not yet converged.\n")
    end
end

return B, R, obj, i_stop

end

function PicardObjectiveB(B, dpp_samples, Fredholm_sample, R,lambda)

    # number of dpp samples
    nb_dpp_samples = length(dpp_samples); 

    # samples Fredholm
    nb_unif = length(Fredholm_sample)

    PhiBPhi = R'*B*R;
    ob_det = 0
    for l = 1:nb_dpp_samples
        id = dpp_samples[l];
        ob_det -= logdet(PhiBPhi[id,id]);
        if ob_det==Inf
            error("singular determinant in objective\n")
        end
    end
    ob_det = ob_det/nb_dpp_samples;
    ob_det += logdet(I + (1/nb_unif)*PhiBPhi[Fredholm_sample,Fredholm_sample]);
    ob_reg = lambda*tr(B);

    return ob_det, ob_reg;
end