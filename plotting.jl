using Base: load_InteractiveUtils
using Plots

include("demoEstimationGaussian.jl")

#############  Main paper ##########################

####### rho = 50 
# with sigma = 0.1 lambda = 0.1 PAPER
#D = load("results/results50/result_s=1_n=1000_p=1000_sigma=100_lambda=100_tol=10.jld");sigma = 0.1
# In the jld file name lambda and sigma have to be divided by 1000


####### rho = 100 
# with sigma = 0.1 and lambda = 0.1 
#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=100_tol=10.jld");sigma = 0.1;

# with sigma = 0.1 and lambda = 0.01 PAPER
#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=10_tol=10.jld");sigma = 0.1;

#############  SM  ##########################

# 3 DPP samples
#D = load("results/results100LambdaSmall/results/result_s=3_n=1000_p=1000_sigma=100_lambda=100divideBy1Million_tol=10.jld"); sigma = 0.1;
# lambda = 1e-4

# 10 DPP samples
#D = load("results/results100LambdaSmall/results/result_s=10_n=1000_p=1000_sigma=100_lambda=100divideBy1Million_tol=10.jld");sigma = 0.1;
# lambda = 1e-4;

##

# 10 DPP samples
#D = load("results/results100LambdaSmall/results/result_s=10_n=1000_p=1000_sigma=150_lambda=100divideBy1Million_tol=10.jld");sigma = 0.15;
# lambda = 1e-4

# 10 DPP samples
D = load("results/results100LambdaSmall/results/result_s=10_n=1000_p=1000_sigma=50_lambda=100divideBy1Million_tol=10.jld");sigma = 0.05;
# lambda = 1e-4


# loading content of dictionary
B = D["B"];
R = D["R"];
GramA = D["GramA"];
GramA0 = D["GramA0"];
GramK = D["GramK"];
GramK0 = D["GramK0"];
n = D["n"]
totalSamples = D["totalSamples"];
i_stop = D["i_stop"];
obj = D["obj"];

## objectives
plot(obj[30:i_stop])

## scatter plot
diagL = diag(R'*B*R);
scatter(totalSamples[:,1],totalSamples[:,2],zcolor=diagL,marker = :+)
scatter!(totalSamples[(n+1):end,1],totalSamples[(n+1):end,2],zcolor=diagL[(n+1):end],marker = :hexagon,legend = false,colorbar = true,framestyle=:box,xtickfont = font(10),ytickfont = font(10))

# decay eigenvalues
l = sort(real(eigvals(R'*B*R)), rev=true)
plot(l,legend = false,framestyle=:box,xtickfont = font(10),ytickfont = font(10),linewidth = 3)

# heatmap estimated correlation kernel 
c_1 = 0.; c_2 = 1.;d = 2;p = 15000;

#n_test = 100*100;
side = 30;
n_test = side*side;
a = 0.2; b = .8; print("Smaller grid")
testSamplesDense = constructFlatSquareGrid(n_test, a, b);

unifSamples = rand(Uniform(c_1,c_2), p,d);
k = SqExponentialKernel();

GramKDense = correlationKernelGram(B,R,unifSamples,totalSamples,testSamplesDense,k,sigma);
IntensityGramK = reshape(diag(GramKDense),(side,side));

#x_tics = a:(1/(side-1)):b;
#y_tics = x_tics;
#display(heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10)))

# heatmap exact correlation kernel 

alpha0 = 0.05;
rho0 = 100;
xtest0 = (testSamplesDense)'/(alpha0/sqrt(2));
k0 = SqExponentialKernel();

GramKDense0 = rho0*kernelmatrix(k0, xtest0) + 1e-10 *I ; # positive definite

plot(diag(GramKDense),xtickfont = font(10),ytickfont = font(10),legend=false);plot!(100*ones(size(diag(GramKDense))),xtickfont = font(10),ytickfont = font(10),legend=false,linewidth=3)

id = 400;
plot(GramKDense[id,:],legend=false,linewidth=2)
ylims!((-15,110))
plot(GramKDense0[id,:],legend=false,linewidth=2)
ylims!((-15,110))