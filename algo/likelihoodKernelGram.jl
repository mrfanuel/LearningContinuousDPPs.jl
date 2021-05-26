using LinearAlgebra
using KernelFunctions
using Distributions

function likelihoodKernelGram(B,R,totalSamples,testSamples,k,sigma)

    F = cholesky(B).U;    
    
    x_m = (totalSamples)'/(sigma);

    x_t = (testSamples)'/(sigma);

    K_mt = kernelmatrix(k, x_m, x_t);

    temp =F*((R')\K_mt);

    GramA = temp'*temp;

    return GramA;
end