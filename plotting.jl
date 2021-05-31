include("demoEstimationGaussian.jl")
using Plots
# In the jld file name lambda and sigma have to be divided by 1000

####### results for rho = 50 #####################

# with sigma = 0.1 lambda = 0.1 PAPER
#D = load("results/results50/result_s=1_n=1000_p=1000_sigma=100_lambda=100_tol=10.jld");
#sigma = 0.1

## with sigma = 0.05 and n = 1000
#D = load("results/results50/result_s=10_n=1000_p=1000_sigma=50_lambda=10_tol=1.jld");
#sigma = 0.05

## with sigma = 0.05 and n = 500
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=50_lambda=1_tol=1.jld");
#sigma = 0.05;

#D = load("results/results50/result_s=10_n=500_p=1000_sigma=50_lambda=10_tol=1.jld");
#sigma = 0.05;

#D = load("results/results50/result_s=10_n=500_p=1000_sigma=50_lambda=100_tol=1.jld");
#sigma = 0.05;

## with sigma = 0.1 and n= 500
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=100_lambda=100_tol=1.jld");
#sigma = 0.1;

#D = load("results/results50/result_s=10_n=500_p=1000_sigma=100_lambda=100_tol=1.jld");
#sigma = 0.1;

## with sigma = 0.01 and n= 500
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=10_lambda=100_tol=10.jld")
#sigma = 0.01;

####### results for rho = 100 #####################

### s = 1 ##############

## with sigma = 0.1 and n= 500
#D = load("results/results100/results/result_s=1_n=500_p=1000_sigma=100_lambda=100_tol=10.jld"); sigma = 0.1;

#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=50_lambda=100_tol=10.jld");sigma = 0.05;

# with sigma = 0.1 and lambda = 0.1 PAPER
#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=100_tol=10.jld");sigma = 0.1;

# with sigma = 0.1 and lambda = 0.01 PAPER
#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=10_tol=10.jld");sigma = 0.1;

#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=1_tol=10.jld")
#sigma = 0.1
### s = 3  ##############

# lambda=1
#D = load("results/results100/results/result_s=3_n=1000_p=1000_sigma=100_lambda=1000_tol=10.jld");sigma = 0.1; # very uniform


#D = load("results/results100/results/result_s=3_n=1000_p=1000_sigma=50_lambda=10000_tol=10.jld");sigma = 0.05;

#D = load("results/results100/results/result_s=3_n=1000_p=1000_sigma=50_lambda=1000_tol=10.jld");sigma = 0.05;

### s =5  ##############

#D = load("results/results100/results/result_s=5_n=1000_p=1000_sigma=100_lambda=1_tol=10.jld")
#sigma = 0.1;#lambda = 0.001

#D = load("results/results100/results/result_s=5_n=1000_p=1000_sigma=50_lambda=100_tol=10.jld")
#sigma = 0.05; #lambda = 0.1

#D = load("results/results100/results/result_s=5_n=1000_p=1000_sigma=50_lambda=1_tol=10.jld")
#sigma = 0.05; #lambda = 0.001

#D = load("results/results100/results/result_s=5_n=1000_p=1000_sigma=50_lambda=10_tol=10.jld")
#sigma = 0.05; #lambda = 0.01


### s = 10 ##############

#D = load("results/results100/results/result_s=10_n=500_p=1000_sigma=50_lambda=1_tol=10.jld")
#sigma = 0.05; # not too bad sigma too small

#D = load("results/results100/results/result_s=10_n=500_p=1000_sigma=100_lambda=1_tol=10.jld")
#sigma = 0.1; #Good one

#D = load("results/results100/results/result_s=10_n=500_p=1000_sigma=100_lambda=10_tol=10.jld")
#sigma = 0.1;

#D = load("results/results100/results/result_s=10_n=1000_p=1000_sigma=50_lambda=5_tol=10.jld")
#sigma = 0.05;

### small lambda %%% s = 3 (divide lambda by 10^6)

#
#D = load("results/results100LambdaSmall/results/result_s=3_n=1000_p=1000_sigma=100_lambda=100divideBy1Million_tol=10.jld"); sigma = 0.1;
# lambda = 1e-4

#D = load("results/results100LambdaSmall/results/result_s=3_n=1000_p=1000_sigma=100_lambda=10divideBy1Million_tol=10.jld"); sigma = 0.1;
# also good ! lambda = 1e-5

#
#D = load("results/results100LambdaSmall/results/result_s=3_n=1000_p=1000_sigma=80_lambda=10divideBy1Million_tol=10.jld"); sigma = 0.08; #lambda = 1e-5
# Very good intensity to put in SM ???? lambda = 1e-5

#D = load("results/results100LambdaSmall/results/result_s=3_n=1000_p=1000_sigma=80_lambda=1divideBy1Million_tol=10.jld"); sigma = 0.08; lambda = 1e-6;

# with 10 dpp samples


D = load("results/results100LambdaSmall/results/result_s=10_n=1000_p=1000_sigma=50_lambda=10divideBy1Million_tol=10.jld"); sigma = 0.05; # lambda = 1e-5;

#D = load("results/results100LambdaSmall/results/result_s=10_n=1000_p=1000_sigma=80_lambda=10divideBy1Million_tol=10.jld");sigma = 0.08; # lambda = 1e-5;

#D = load("results/results100LambdaSmall/results/result_s=10_n=1000_p=1000_sigma=100_lambda=100divideBy1Million_tol=10.jld");sigma = 0.1; # lambda = 1e-4;
# not bad


###############################################
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
#savefig("figures/DiagDiscreteIntensitySigma005Lambda01.pdf")

# decay eigenvalues
l = sort(real(eigvals(R'*B*R)), rev=true)
plot(l,legend = false,framestyle=:box,xtickfont = font(10),ytickfont = font(10),linewidth = 3)
#savefig("figures/EigenDecay.pdf")

# heatmap correlation kernel 
c_1 = 0.; c_2 = 1.;d = 2;p = 15000;

n_test = 100*100; a = 0.; b = 1.;
testSamplesDense = constructFlatSquareGrid(n_test, a, b);

unifSamples = rand(Uniform(c_1,c_2), p,d);
k = SqExponentialKernel();

GramKDense = correlationKernelGram(B,R,unifSamples,totalSamples,testSamplesDense,k,sigma);
IntensityGramK = reshape(diag(GramKDense),(100,100));

x_tics = 0:(1/99):1;
y_tics = x_tics;
display(heatmap(x_tics,y_tics,IntensityGramK,colorbar = true,xtickfont = font(10),ytickfont = font(10)))


# smaller grid to remove boundary
# heatmap correlation kernel 

#c_1 = 0.; c_2 = 1.;d = 2;p = 2000; # domain: do not modify

#n_test = 100*100; a = 0.1; b = 0.9;
#testSamplesDenseSmall = constructFlatSquareGrid(n_test, a, b);

#unifSamples = rand(Uniform(c_1,c_2), p,d);
#k = SqExponentialKernel();

#GramKDenseSmall = correlationKernelGram(B,R,unifSamples,totalSamples,testSamplesDenseSmall,k,sigma);
#IntensityGramKSmall = reshape(diag(GramKDenseSmall),(100,100));

#x_tics = a:((b-a)/99):b;
#y_tics = x_tics;
#display(heatmap(x_tics,y_tics,IntensityGramKSmall,colorbar = true,xtickfont = font(10),ytickfont = font(10)))
#display(scatter!(unifSamples[:,1],unifSamples[:,2]))


# likelihood kernel
#A_insample = likelihoodKernelGram(B,R,totalSamples,totalSamples,k,sigma);
#GramADense = likelihoodKernelGram(B,R,totalSamples,testSamplesDense,k,sigma);
#scatter(testSamplesDense[:,1],testSamplesDense[:,2],zcolor=diag(GramADense),marker = :dot,legend = false,colorbar = true,xtickfont = font(10),ytickfont = font(10))
#eigGramA = eigvals(GramA/size(GramA,1));
#eigGramK = eigvals(GramK/size(GramK,1));
