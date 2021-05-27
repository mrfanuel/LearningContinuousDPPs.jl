# number of DPP samples
#s = 1; 

# number of uniform samples for Fredholm
#n = 700; #300

# number of uniform samples for correlation kernel
#p = 1000;

# kernel bw (Float64)
#sigma = 0.1; ;# last 0.1; # last 0.05

# regularization (Float64)£
#lambda =  1e-1 # last 1e-4

# regularizer for K positive definite
#epsilon = 1e-12; 

# max number of iteration
#it_max = 10000;

# relative objective tolerance
#tol = 1e-5;

# merge dpp and unif samples


####### results for rho = 50
# 1DPP sample with sigma = 0.1 and n= 1000
#D = load("results/results50/result_s=1_n=1000_p=1000_sigma=100_lambda=100_tol=10.jld");

## with sigma = 0.05 and n = 1000
#D = load("results/results50/result_s=10_n=1000_p=1000_sigma=50_lambda=10_tol=1.jld");

## with sigma = 0.05 and n = 500
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=50_lambda=1_tol=1.jld");
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=50_lambda=10_tol=1.jld");
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=50_lambda=100_tol=1.jld");
#sigma = 0.05;

## with sigma = 0.1 and n= 500
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=100_lambda=100_tol=1.jld");
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=100_lambda=100_tol=1.jld");
#sigma = 0.1;

## with sigma = 0.01 and n= 500
#D = load("results/results50/result_s=10_n=500_p=1000_sigma=10_lambda=100_tol=10.jld")
#sigma = 0.01;

####### results for rho = 100
## with sigma = 0.1 and n= 500
#D = load("results/results100/results/result_s=1_n=500_p=1000_sigma=100_lambda=100_tol=10.jld"); sigma = 0.1;

#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=50_lambda=100_tol=10.jld");sigma = 0.05;

#D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=100_tol=10.jld");sigma = 0.1;

D = load("results/results100/results/result_s=1_n=1000_p=1000_sigma=100_lambda=10_tol=10.jld");sigma = 0.1;


B = D["B"];
R = D["R"];
GramA = D["GramA"];
GramA0 = D["GramA0"];

GramK = D["GramK"];
GramK0 = D["GramK0"];

n = D["n"]

totalSamples = D["totalSamples"];

#plot(diag(R'*B*R))
#heatmap(GramK)
plot(obj[30:i_stop])
obj = D["obj"];
i_stop = D["i_stop"];

#plot(diag(GramA))
plot(diag(GramK))

diagL = diag(R'*B*R);
#L = R'*B*R;
#L = 0.5*(L+L');
#d_discrete = diag(L*inv(L+I));
scatter(totalSamples[:,1],totalSamples[:,2],zcolor=diagL,marker = :+)
scatter!(totalSamples[(n+1):end,1],totalSamples[(n+1):end,2],zcolor=diagL[(n+1):end],marker = :hexagon,legend = false,colorbar = true,framestyle=:box,xtickfont = font(10),ytickfont = font(10))
#savefig("figures/DiagDiscreteIntensitySigma005Lambda01.pdf")



# out of sample ###########
n_test = 30*30; a = 0.; b = 1.;
print("\n")
print("test points in [$(a), $(b)]")
print("\n")
d_test = diag(GramK);
testSamples = constructFlatSquareGrid(n_test, a, b);
scatter(testSamples[:,1],testSamples[:,2],zcolor=d_test,marker = :dot)



c_1 = 0.; c_2 = 1.;
d = 2;
p = 1000;

n_test = 100*100; a = 0.; b = 1.;
testSamplesDense = constructFlatSquareGrid(n_test, a, b);

unifSamples = rand(Uniform(c_1,c_2), p,d);

k = SqExponentialKernel();

#sigma = 0.05;
#sigma = 0.01;

GramKDense = correlationKernelGram(B,R,unifSamples,totalSamples,testSamplesDense,k,sigma);

scatter(testSamplesDense[:,1],testSamplesDense[:,2],zcolor=diag(GramKDense),marker = :dot,legend = false,colorbar = true,xtickfont = font(10),ytickfont = font(10))
#savefig("figures/DenseIntensitySigma005Lambda01.pdf")

#,xtickfont = font(10),ytickfont = font(10)
l = sort(real(eigvals(R'*B*R)), rev=true)
plot(l,legend = false,framestyle=:box,xtickfont = font(10),ytickfont = font(10),linewidth = 3)
#savefig("figures/EigenDecay.pdf")


A_insample = likelihoodKernelGram(B,R,totalSamples,totalSamples,k,sigma);

GramADense = likelihoodKernelGram(B,R,totalSamples,testSamplesDense,k,sigma);
scatter(testSamplesDense[:,1],testSamplesDense[:,2],zcolor=diag(GramADense),marker = :dot,legend = false,colorbar = true,xtickfont = font(10),ytickfont = font(10))


# 
#eigGramA = eigvals(GramA/size(GramA,1));
#eigGramK = eigvals(GramK/size(GramK,1));