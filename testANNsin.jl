
# test simple ANN to predict sin function
# DL rev 29-Dec-2018


using Knet
using PyPlot
using AutoGrad
using Compat.Statistics
## using mymod

predict(w,x) = w.*x
loss(w,x,y) = mean(abs2,y-predict(w,x))
lossgradient =grad(loss)

const	lr_= 0.1;
const beta1_ = 0.9;
const beta2_ = 0.95;
const eps_ =1e-6;




function train(P,prms,x,y,maxiters = 200)
	prm = Adam(lr=lr_, beta1 = beta1_, beta2 = beta2_, eps = eps_);
	for iter = 1:maxiters
		for i = length(x)
			g = lossgradient(P,x,y);
			update!(P,g,prm);
		end
	end
	return P
end

function test(P,timepoints,y)
	sumloss = numloss = 0;
	for t in timepoints
		sumloss += loss(P,t,y)
		numloss += 1
	end
	return sumloss/numloss
end

function mainTestANNsin()
	
	
	nPoints = 1000;
	y = zeros(Float64,nPoints);
	w = zeros(Float64,nPoints);

	x = rand(nPoints)*pi*2.0;

	for i=1:nPoints
		y[i] = sin(x[i]) + rand()*0.05; 
	end

	
	#init the ADAM algorithm
	prms = Any[];
	prms = Adam(lr=lr_, beta1 = beta1_, beta2 = beta2_, eps = eps_)

	_maxIters_ = 50;

	

	@time for iters=1:_maxIters_
		train(w,prms,x,y,100)
		loss = test(w,x,y)
		if mod(iters,100) == 0
			println(:iteration,iters, :loss,loss)
		end
	end

	figure(1)
	clf();
	plot(x,y,"og",color=:blue,markersize=6, label="sin")
	plot(x,w.*x,"or", markersize = 2, label = "approximation")
	xlabel("x")
	ylabel("y")
	legend();
	
	
end


mainTestANNsin();
##foo()
