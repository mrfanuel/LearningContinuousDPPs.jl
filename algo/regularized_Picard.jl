using LinearAlgebra

"""
    regularized_Picard(B, K, dpp_samples_ids, Fredholm_sample_ids, lambda, it_max, tol, use_inverse)

Run regularized Picard iteration starting from an initial positive definite matrix B.

# Arguments
- `B::Array{Float64,2}`: an initial positive definite matrix B.
- `K::Array{Float64,2}`: full kernel matrix.
- `dpp_samples_ids:Array{Array{Int64,1},1}`: array of arrays containing indices of s DPP samples.
- `Fredholm_sample_ids:Array{Int64,1}`: indices of samples for approximating Fredholm determinant.
- `lambda:Float`: regularization parameter.
- `it_max::Integer`: largest number of iterations.
- `tol:Float`: relative objective variation desired for regularized Picard iteration.
- `use_inverse:Bool`: if true, inverse matrices are computed, otherwise only linear systems.

# Output
- `B::Array{Float64,2}`: approximate solution.
- `R::Array{Float64,2}`: Cholesky s.t. K = R'R.
- `obj`: array containing the objective values at each iteration.
- `i_stop`: number of iterations runned before stopping.

# Example
import LinearAlgebra 
B, R, obj, i_stop = regularized_Picard(B,K,dpp_samples_ids,Fredholm_sample_ids,lambda,it_max,tol,use_inverse);
"""
function regularized_Picard(B::Array{Float64,2},K::Array{Float64,2}, dpp_samples_ids::Array{Array{Int64,1},1}, Fredholm_sample_ids::Array{Int64,1}, lambda::Float64, it_max::Int64 ,tol::Float64,use_inverse::Bool)

# number of dpp_samples_ids
nb_dpp_samples_ids = length(dpp_samples_ids); 

# total number of points
m = size(K,1);

# Chol decomposition
R = cholesky(K).U;

# sampling matrix for uniformSample
nb_unif = length(Fredholm_sample_ids)

# initialization
obj = zeros(it_max,1);
i_stop = it_max;

# regularization
epsilon = 1e-10;

# iterations
for i in 1:it_max
    # construct  Delta
    BRDeltaRtB = zeros(m,m);
    RtBR = R'*B*R
    BR = B*R
    for l = 1:nb_dpp_samples_ids
        id = dpp_samples_ids[l];
        M = (RtBR + UniformScaling(epsilon))[id,id]
        BRid = BR[:,id]
        if use_inverse
            BRDeltaRtB +=  BRid * inv(M) * BRid';
        else
            BRDeltaRtB +=  BRid * (M\(BRid'));
        end
    end

    N = (RtBR + UniformScaling(nb_unif))[Fredholm_sample_ids,Fredholm_sample_ids]
    BRunif = BR[:,Fredholm_sample_ids]
    if use_inverse
        BRDeltaRtB = BRDeltaRtB/nb_dpp_samples_ids - BRunif*inv(N)*BRunif';
    else
        BRDeltaRtB = BRDeltaRtB/nb_dpp_samples_ids - BRunif*(N\(BRunif'));
    end

    BRDeltaRtB = 0.5*(BRDeltaRtB + BRDeltaRtB');

    # Picard iteration
    pB = B + BRDeltaRtB;

    # final expression
    B = (0.5/lambda)*(real(sqrt(I+4*lambda*pB))-I);
    B = 0.5*(B+B');

    # track the objective values
    obj_det,ob_reg = Picard_objective(B, dpp_samples_ids, Fredholm_sample_ids, R,lambda);
    obj[i] = obj_det + ob_reg;
    # printing in repl
    if i%100 == 0
        rel_variation = abs(obj[i]-obj[i-1])/abs(obj[i])
        print("---------------------------------------------------------------\n")
        print("$(i) / $it_max\n")
        print("relative objective variation $(rel_variation)\n")
        print("objective = $(obj[i]) \n")

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


"""
    Picard_objective(B, dpp_samples_ids, Fredholm_samples_ids, R,lambda)

# Arguments
- `B::Array{Float64,2}`: positive definite matrix B.
- `dpp_samples_ids:Array{Array{Int64,1},1}`: array of arrays containing indices of s DPP samples.
- `Fredholm_sample_ids:Array{Int64,1}`: indices of samples for approximating Fredholm determinant.
- `R`: Cholesky decomposition such that K = R'R.
- `lambda:Float`: regularization parameter.

Computes Picard objective function.
"""
function Picard_objective(B, dpp_samples_ids, Fredholm_samples_ids, R,lambda)

    # number of dpp samples
    nb_dpp_samples_ids = length(dpp_samples_ids); 

    # samples Fredholm
    nb_unif = length(Fredholm_samples_ids)

    PhiBPhi = R'*B*R;
    ob_det = 0
    for l = 1:nb_dpp_samples_ids
        id = dpp_samples_ids[l];
        ob_det -= logdet(PhiBPhi[id,id]);
        if ob_det==Inf
            error("singular determinant in objective\n")
        end
    end
    ob_det = ob_det/nb_dpp_samples_ids;
    ob_det += logdet(I + (1/nb_unif)*PhiBPhi[Fredholm_samples_ids,Fredholm_samples_ids]);
    ob_reg = lambda*tr(B);

    # full objective = ob_det + ob_reg
    return ob_det, ob_reg;
end