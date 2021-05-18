using LinearAlgebra
using KernelFunctions
using Distributions

function approxCorrelationKernelMatrix(C,p,c,totalSamples,k,sigma)

    L = cholesky(C).U;

    # number of points for approximating K
    unifSamples = rand(Uniform(-c,c), p,2);

    # construct kernel matrices
    x_n = (totalSamples)'/(sigma);
    x_p = (unifSamples)'/(sigma);
    K_np = kernelmatrix(k, x_n, x_p);

    # matrix of approximate correlation kernel
    # the formula is
    #K_hat_mat0 = L'*inv(L*(K_np*(1/p)*K_np')*L' + I)*L;
    # But to preserve positive definiteness we do as follows

    M = L*(K_np*(1/p)*K_np')*L' + I; # matrix to be 'inverted'
    M = 0.5*(M+M'); # makes sure it is symmetric
    T = cholesky(M).U;
    F = (T')\L; # solves system with Cholesky factorization
    K_hat_mat = F'*F;

    # to check correctness
    #print(norm(K_hat_mat-K_hat_mat0))

    return K_hat_mat;
end