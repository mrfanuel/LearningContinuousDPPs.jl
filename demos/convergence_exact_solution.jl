using LinearAlgebra
using Plots
using CSV
using KernelFunctions
using Distributions
using DataFrames

include("../algo/kernels.jl")
include("../algo/regularized_Picard.jl")
include("../algo/utils.jl")

function convergence_exact_solution()

    # width
    sigma = .01
    # regularizer
    lambda =  .05
    # regularizer for positive definiteness
    epsilon = 1e-10; 
    # relative objective tolerance
    tol = 1e-6

    ############# load 1 DPP samples #############
    # intensity of the generating DPP
    intensity = 100; # 50 or 100
    strg_intensity = string(intensity)
    
    # create an array of arrays
    indices_DPP_samples = Array{Int64,1}[];
    
    print("Loading DPP samples from: \n")
    i = 1; # id of the DPP sample
    file_name = "data/statspats/samples/GaussDPPsample_alpha0_00p5_rho0_"*strg_intensity*"_nb_"*string(i)*".csv"
    print(file_name, " \n")
    temp = CSV.File(file_name; header=true) |> Tables.matrix 
    temp = temp[:,2:3]; 
    id_temp = collect(1:size(temp,1));
    push!(indices_DPP_samples,id_temp);
    total_samples = temp;

    # for approximating Fredholm determinant
    indices_Fredholm_sample = indices_DPP_samples[1]

    ############# exact solution #############

    # create full kernel matrix
    x = (total_samples)'/sigma;
    k = SqExponentialKernel();
    K = kernelmatrix(k, x) + epsilon *I ; 

    # Chol decomposition
    R = cholesky(K).U;
    Rinv = inv(R);

    # Exact solution
    m = size(K,1);
    X_exact = 0.5*(real(sqrt(m^2*I + 4*m*K/lambda))-m*I);
    B_exact = Rinv'*X_exact*Rinv;

    # objective exact solution
    obj_det_exact,obj_reg_exact = Picard_objective(B_exact, indices_DPP_samples, indices_Fredholm_sample, R,lambda)
    obj_exact = obj_det_exact + obj_reg_exact;

    ############# approximate solution #############

    # initial positive definite iterate
    X = randn(size(K));
    B = X*X'+ UniformScaling(epsilon);
    
    use_inverse = false

    n_steps = 50;
    it_max = 1;
    err = ones(n_steps,1);
    obj = ones(n_steps,1);

    for j=1:n_steps
        B, R, obj_j, i_stop = regularized_Picard(B, K, indices_DPP_samples, indices_Fredholm_sample, lambda, it_max ,tol,use_inverse)
        err[j] = norm(B-B_exact)/norm(B_exact);
        obj[j] = obj_j[end];
    end
    
    #############  plotting #############

    plt_error = plot(err, yaxis=:log,legend=false,ylabel = "relative error",xlabel = "iteration")
    display(plt_error)
    
    plt_obj = plot(obj,legend=false,ylabel = "objective",xlabel = "iteration")
    plot!(obj_exact*ones(size(obj)))
    display(plt_obj)
end
