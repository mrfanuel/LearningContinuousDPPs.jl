
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