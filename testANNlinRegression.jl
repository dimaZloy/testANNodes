
# test simple ANN to make linear regression on random data 
# DL rev 29-April-2019


	using Knet
	using PyPlot
	using AutoGrad
	using Compat.Statistics


	predict(w,x)=(w[1]*x.+w[2])
	loss(w,x,y)=(sum(abs2,y-predict(w,x))/size(x,2) )

	lossgradient =grad(loss)

	function train(w, x, y; lr=.1, epochs=20)
    	for epoch=1:epochs
        	g = lossgradient(w, x, y)
        	update!(w, g; lr=lr)
    	end
    	return w
	end
	

	n = 1000;

	xtrn = zeros(Float64,1,n);
	ytrn = zeros(Float64,1,n);
	xtst = zeros(Float64,1,n);
	ytst = zeros(Float64,1,n);


	for i = 1:n
		xtrn[i] = rand();
		ytrn[i] = rand();
	
		xtst[i] = xtrn[i];
		ytst[i] = ytrn[i];
	end
	
    
    w = map(Array{Float64}, [ 0.0*randn(1,1), 0.0*randn(1,1) ])


    report(epoch)=println((:epoch,epoch,:trn,loss(w,xtrn,ytrn),:tst,loss(w,xtst,ytst)))
	
	Nepochs = 10;
	
    
    report(0)
    @time for epoch=1:Nepochs
       train(w, xtrn, ytrn; lr=0.1, epochs=1)
       report(epoch)
    end
    
	f = w[1]*xtrn .+ w[2]; 
    
	figure(1);
	clf();
	plot(xtrn[1:end], ytrn[1:end],"or",label = "raw data");
	plot(xtrn[1:end], f[1:end],"sk", label = "predicted regression")
 	xlabel("x");
 	ylabel("y");
	legend();	
	
