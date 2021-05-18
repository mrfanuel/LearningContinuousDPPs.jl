
using LinearAlgebra
using KernelFunctions

function integralCorrelationKernelFunction(v,w,K_hat_mat,totalSamples,k,sigma)
    
    x_n = (totalSamples)'/(sigma);

    v = v/(sigma);
    w = w/(sigma);

    K_nv = kernelmatrix(k, x_n, v);
    K_nw = kernelmatrix(k, x_n, w);

    k_vw = (K_nv'*K_hat_mat*K_nw)[1]

    return k_vw;
end

    