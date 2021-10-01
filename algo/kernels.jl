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

    temp_2 = F*((R')\K_mt);

    GramK = temp_2'*(M\temp_2); 

    GramK = 0.5*(GramK+GramK');

    return GramK;
end


function likelihoodKernelGram(B,R,totalSamples,testSamples,k,sigma)

    F = cholesky(B).U;    
    
    x_m = (totalSamples)'/(sigma);

    x_t = (testSamples)'/(sigma);

    K_mt = kernelmatrix(k, x_m, x_t);

    temp =F*((R')\K_mt);

    GramA = temp'*temp;

    return GramA;
end

function evaluateGramGrid(X,K_hat_mat,totalSamples,k,sigma)

    nb_pts_grid = size(X,1);
    GramMatrix = zeros(Float64, nb_pts_grid,nb_pts_grid);
    for i in 1:nb_pts_grid
        for j in 1:nb_pts_grid
            v_i = X[i,:][1]';
            v_j = X[j,:][1]';
            GramMatrix[i,j] = integralKernelFunction(v_i,v_j,K_hat_mat,totalSamples,k,sigma);
        end
    end

    return GramMatrix;
end


function integralKernelFunction(v,w,K_hat_mat,totalSamples,k,sigma)
    
    x_n = (totalSamples)'/(sigma);

    v = v/(sigma);
    w = w/(sigma);

    K_nv = kernelmatrix(k, x_n, v);
    K_nw = kernelmatrix(k, x_n, w);

    k_vw = (K_nv'*K_hat_mat*K_nw)[1]

    return k_vw;
end