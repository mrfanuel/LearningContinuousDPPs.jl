using LinearAlgebra
using KernelFunctions
using Distributions


"""
    correlation_kernel_Gram(B,R,unif_samples,all_samples,test_samples,k,sigma)

# Arguments
- `B::Array{Float64,2}`: B matrix with same size as K.
- `R:Array{Float64,2}`: Cholesky s.t. K = R'R
- `unif_samples:Array{Float64,2}`: number of uniformly sampled points for correlation kernel approximaton
- `all_samples:Array{Float64,2}`: array containing all points used to write the representer
- `test_samples:Array{Float64,2}`: test samples for evaluating the Gram matrix
- `k:kernel`: RKHS kernel function; see KernelFunctions.
- `sigma:Float`: RKHS kernel bandwidth.

Compute Gram matrix of correlation kernel on test data.
"""
function correlation_kernel_Gram(B,R,unif_samples,all_samples,test_samples,k,sigma)

    F = cholesky(B).U;

    # construct kernel matrices
    x_m = (all_samples)'/(sigma);
    x_p = (unif_samples)'/(sigma);
    K_mp = kernelmatrix(k, x_m, x_p);

    p = size(K_mp,2);

    x_t = (test_samples)'/(sigma);
    K_mt = kernelmatrix(k, x_m, x_t);

    temp_1 = F*((R')\K_mp);

    M = temp_1*(1/p)*temp_1' +I;

    temp_2 = F*((R')\K_mt);

    GramK = temp_2'*(M\temp_2); 

    GramK = 0.5*(GramK+GramK');

    return GramK;
end


"""
    likelihood_kernel_Gram(B,R,all_samples,test_samples,k,sigma)

# Arguments
- `B::Array`: B matrix with same size as K.
- `R:Array`: Cholesky s.t. K = R'R
- `all_samples:Array`: array containing all points used to write the representer
- `test_samples:Array`: test samples for evaluating the Gram matrix
- `k:kernel`: RKHS kernel function; see KernelFunctions.
- `sigma:Float`: RKHS kernel bandwidth.

Compute Gram matrix of likelihood kernel on test data.
"""
function likelihood_kernel_Gram(B,R,all_samples,test_samples,k,sigma)

    F = cholesky(B).U;    
    
    x_m = (all_samples)'/(sigma);

    x_t = (test_samples)'/(sigma);

    K_mt = kernelmatrix(k, x_m, x_t);

    temp = F*((R')\K_mt);

    GramA = temp'*temp;

    return GramA;
end


function line(center,direction,odd_number_pts)
    x = zeros(Float64,odd_number_pts,2);
    l = zeros(Float64,odd_number_pts,1);
    L = norm(direction);
    for i in 1:odd_number_pts
        l[i] = (2*i/(odd_number_pts - 1) - 1)*L; # position along direction
        x[i,1] = center[1] + (2*i/(odd_number_pts - 1) - 1)*direction[1];
        x[i,2] = center[2] + (2*i/(odd_number_pts - 1) - 1)*direction[2];
    end
    id_center = Int64((odd_number_pts - 1)/2)
    return x, l, id_center;
end

