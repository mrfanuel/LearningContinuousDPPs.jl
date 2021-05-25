include("algo/estimateGaussianB.jl")
include("algo/constructSquareGrid.jl")
include("algo/integralKernelFunction.jl")
include("algo/approxCorrelationKernelMatrix.jl")
include("algo/evaluateGramGrid.jl")

# Here we use the square SqExponentialKernel

# number of DPP samples
s = 1; #s = 3;

# number of uniform samples for Fredholm
n = 100; #n = 200 # 100 is good;

# number of uniform samples for correlation kernel
p = 1000;

#kernel = "MaternKernel";#"SqExponentialKernel"
kernel = "SqExponentialKernel";

nu = 1/2.;

# kernel bw (Float64)
sigma = .1; #sigma = 0.5; too large

# regularization (Float64)
#lambda =  1e-8#1e-6; #lambda =  .0001 
lambda =  1e-8 # 1e-10 # good ?

# regularizer for K positive definite
epsilon = 1e-7; 

# max number of iteration
it_max = 10000;

# relative objective tolerance
tol = 1e-5; #1e-6

# Plotting number of grid points along x-axis
n_step_plot = 100; 

# merge dpp and unif samples
merge = false

FredholmSampling = "uniform";

# For solving systems in Picard iteration
meth = "direct";

###################################
# estimation of likelihood kernel
###################################

# uniform sampling in the box [0,1]^2
if FredholmSampling == "uniform"
    FredholmSample = rand(Uniform(0,1), n,2);
else
    a = 0.; b = 1.;
    FredholmSample = constructFlatSquareGrid(n, a, b)
end


B,K,k,totalSamples = estimateGaussianB(s,n,kernel,nu,sigma,lambda,epsilon,it_max,tol,FredholmSample,merge)

R = cholesky(K).U;
invR = inv(R);

C = invR'*B*invR; 
C = 0.5*(C+C');# makes sure it is symmetric

# construct grid n_step x n_step within [eps, 1-eps]^2
n_step = 20;
a = 0.;
b = 1.;
X = constructSquareGrid(n_step, a, b);

# evaluate on the grid
GramMatrixA = evaluateGramGrid(X,C,totalSamples,k,sigma);

###################################
# estimation of correlation kernel
###################################
print("\n")
print("estimate  correlation kernel...")

c_1 = 0;
c_2 = 1;
d = 2;

# number of points for approximating K
unifSamples = rand(Uniform(c_1,c_2), p,d);
K_hat_mat = approxCorrelationKernelMatrix(C,unifSamples,totalSamples,k,sigma);
print("\n")

###################################
# Plotting
###################################

print("plotting...")
intensity = zeros(Float64, n_step_plot,n_step_plot);

for i in 1:n_step_plot
    for j in 1:n_step_plot
        x = (i-1)/(n_step_plot-1);
        y = (j-1)/(n_step_plot-1);
        v = [x y]';
        intensity[i,j] = integralKernelFunction(v,v,K_hat_mat,totalSamples,k,sigma);
    end
end   

# plotting intensity of the estimated process

x_tics = (0:(1/(n_step_plot-1)):1);
y_tics = x_tics;
display(heatmap(x_tics,y_tics,intensity,c=cgrad([:blue, :white,:red, :yellow]),xlabel="x",ylabel="y",title="estimated intensity"))

for i = 1:s
    v0 = totalSamples[(n+1):end,:];
    display(scatter!(v0[:,1],v0[:,2],color =:red, legend = false))
end
display(scatter!(FredholmSample[:,1],FredholmSample[:,2],color = :blue, legend = false))
display(scatter!(unifSamples[:,1],unifSamples[:,2],color = :black, legend = false))

###################################
# evaluate GramMatrix
###################################

# construct grid n_step x n_step within [eps, 1-eps]^2
n_step = 20;
a = 0.1;
b = 1-0.1;
X = constructSquareGrid(n_step, a, b)

# evaluate on the grid
GramMatrix = evaluateGramGrid(X,K_hat_mat,totalSamples,k,sigma);


d = diag(GramMatrix);



