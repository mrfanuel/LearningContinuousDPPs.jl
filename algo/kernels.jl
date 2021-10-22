using LinearAlgebra
using KernelFunctions


"""
    correlation_kernel_Gram(B,R,unif_samples,all_samples,test_samples,k,sigma)

Compute Gram matrix of correlation kernel on test data.

# Arguments
- `B::Array{Float64,2}`: B matrix with same size as K.
- `R:Array{Float64,2}`: Cholesky s.t. K = R'R
- `unif_samples:Array{Float64,2}`: number of uniformly sampled points for correlation kernel approximaton
- `all_samples:Array{Float64,2}`: array containing all points used to write the representer
- `test_samples:Array{Float64,2}`: test samples for evaluating the Gram matrix
- `k:kernel`: RKHS kernel function; see KernelFunctions.
- `sigma:Float`: RKHS kernel bandwidth.

# Output
- `GramK::Array{Float64,2}`: Gram matrix of size |test_samples| x |test_samples|

# Example
import LinearAlgebra, KernelFunctions 
GramK = correlation_kernel_Gram(B,R,unif_samples,all_samples,test_samples,k,sigma);

"""
function correlation_kernel_Gram(B,R,unif_samples,all_samples,test_samples,k,sigma)

    F = cholesky(B).U; # B = F'F

    # construct cross kernel matrices
    x_m = (all_samples)'/(sigma);
    x_p = (unif_samples)'/(sigma);
    K_mp = kernelmatrix(k, x_m, x_p);

    x_t = (test_samples)'/(sigma);
    K_mt = kernelmatrix(k, x_m, x_t);

    T_1 = F*((R')\K_mp);
    p = size(K_mp,2);
    M = (1/p)*T_1*T_1' + I;

    T_2 = F*((R')\K_mt);

    GramK = T_2'*(M\T_2); 

    GramK = 0.5*(GramK+GramK');

    return GramK;
end


"""
    likelihood_kernel_Gram(B,R,all_samples,test_samples,k,sigma)

Compute Gram matrix of likelihood kernel on test data.

# Arguments
- `B::Array`: B matrix with same size as K.
- `R:Array`: Cholesky s.t. K = R'R
- `all_samples:Array`: array containing all points used to write the representer
- `test_samples:Array`: test samples for evaluating the Gram matrix
- `k:kernel`: RKHS kernel function; see KernelFunctions.
- `sigma:Float`: RKHS kernel bandwidth.

# Output
- `GramA::Array{Float64,2}`: Gram matrix of size |test_samples| x |test_samples|

# Example
import LinearAlgebra, KernelFunctions 
GramK = correlation_kernel_Gram(B,R,unif_samples,all_samples,test_samples,k,sigma);

"""
function likelihood_kernel_Gram(B,R,all_sples,test_sples,k,sigma)

    F = cholesky(B).U; # B = F'F  
    
    # construct cross kernel matrices
    x_m = (all_sples)'/(sigma);
    x_t = (test_sples)'/(sigma);
    K_mt = kernelmatrix(k, x_m, x_t);

    T = F*((R')\K_mt);
    GramA = T'*T;

    return GramA;
end

