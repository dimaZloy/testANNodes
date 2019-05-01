
# test simple ANN to predict sin function using the non-linear regression 
# DL rev 1-May-2019


using Knet;
using PyPlot;
using DataFrames;


function predict(ω, x) 
    x = mat(x)
    for i=1:2:length(ω)-2
        x = relu.(ω[i]*x .+ ω[i+1]) ## - fast 
		##x = tanh.(ω[i]*x .+ ω[i+1]) ## - very slow 
		##x = sigm.(ω[i]*x .+ ω[i+1]) ## - nooo
    end
    return ω[end-1]*x .+ ω[end]
end

loss(ω, x, y) = mean(abs2, predict(ω, x)-y);

lossgradient = grad(loss);


function train(ω, data; lr=0.1) 
    for (x,y) in data
        dω = lossgradient(ω, x, y)
        for i in 1:length(ω)
            ω[i] -= dω[i]*lr
        end
    end
    return ω
end;



N1 = 100; ##90
N2 = 100; ##60


ω = Any[xavier(Float64,N1,1), zeros(Float64,N1,1), ## One input layer of size 1
        xavier(Float64,N2,N1), zeros(Float64,N2,1), ## A hidden layer of size N1
        xavier(Float64,1,N2),  zeros(Float64,1,1)]; ## Another hidden layer of size N2



nPoints = 1000;	
y = zeros(Float64,1,nPoints);
x = zeros(Float64,1,nPoints);

	

for i=1:nPoints
	x[i] = rand()*pi*2.0;	
	y[i] = sin(x[i]) + rand()*0.05; 
end


Niters = 20000;
div = 100;
Nout = Int(round(Niters/div));


errdf = DataFrame(Epoch=1:Nout, Error=0.0)
cntr = 1
for i=1:Niters
	global cntr;
    train(ω, [(x,y)], lr=0.1)
    if mod(i, div) == 0
        errdf[cntr, :Epoch]=i
		err = loss(ω,x,y);
        errdf[cntr, :Error]= err
        cntr+=1
		println(:iteration, "\t",  i, "\t",  :loss, "\t", err)
    end
end



figure(1)
clf();
subplot(1,3,1)
plot(x,y, "og", markersize=6);
plot(x,predict(ω, x),"sw", markersize=2, markeredgecolor="gray", );
xlabel("x");
ylabel("y");

subplot(1,3,2)
plot(predict(ω, x), y,"sw", markersize=6,markeredgecolor="gray",); 
xlabel("Predicted");
ylabel("Observed");

subplot(1,3,3)
semilogy(errdf[:,:Epoch], errdf[:,:Error], "--k")
xlabel("Error");
ylabel("Epoch");

