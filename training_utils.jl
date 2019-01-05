
function predict(P,x)
	w1,b1,w2 = P
	h = sigm(w1 * x .+ b1)
	return w2 + h
end


function loss_trial(P,timepoints,f,phi,hl_width)
	w,b,v = P
	t0 = timepoints[1]
	dN_dt(t) = sum([v[i]*w[i]*sig_der(w[i]*t .+ b[i]) for i = 1:hl_width])
	sumabs2([gradient(x->phi(P,x),t) - f(t,phi(P,t)) for t in timepoints][1])
end


lossgradient = grad(loss_trial)

function train(P,prms, timepoints, f, phi, hl_width; maxiters = 100)
	for iter = 1:maxiters
		for x in timepoints
			g = lossgradient(P,timepoints,f,phi,hl_width)
			update!(P,g,prms)
		end
	end
	return P
end


function test(P,timepoints,f,phi,hl_width)
	sumloss = numloss = 0;
	for t in timepoints
		sumloss += loss_trial(P,timpoints,f,phi,hl_width)
		numloss += 1
	end
end


function init_params(ftype, hl_width;atype = KnetArray{Float64})
	P = Array{Any}(3)
	P[1] = randn(ftype,hl_width,1);
	P[2] = zeros(ftype,hl_width,1);
	P[3] = randn(ftype,hl_width,1);
end


function generate_data(low, high, dt; atype = KnetArray{Float64})
	num_points = 1/dt
	x = linspace(low,high,num_points+1)
	return x
end
