using LinearAlgebra
using KernelFunctions
using Distributions

function correlationKernelGram(B,R,unifSamples,totalSamples,testSamples,k,sigma)

    F = cholesky(B).U;

    # construct kernel matrices
    x_m = (totalSamples)'/(sigma);
    x_p = (unifSamples)'/(sigma);
    K_mp = kernelmatrix(k, x_m, x_p);

    p = size(K_mp,2);

    x_t = (testSamples)'/(sigma);
    K_mt = kernelmatrix(k, x_m, x_t);

    temp_1 = F*((R')\K_mp);

    M = temp_1*(1/p)*temp_1' +I;

    print("cond number of estimateK $(cond(M))\n")

    temp_2 = F*((R')\K_mt);

    GramK = temp_2'*(M\temp_2); # this system could be preconditionned.

    GramK = 0.5*(GramK+GramK');

    return GramK;
end