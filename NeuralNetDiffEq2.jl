
using Reexport
@reexport using DiffEqBase
using Knet
using Compat
using ForwardDiff

import DiffEqBase: solve

@compat abstract type NeuralNetDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct nnode <: NeuralNetDiffEqAlgorithm
	hl_width::Int
end


nnode(;hl_width=10) = nnode(hl_width)

sig_der(x) = sigm(x)*(1-sigm(x))

function solve(
		prob::DiffEqBase.AbstractODEProblem,
		alg::NeuralNetDiffEqAlgorithm,
		dt = nothing,
		timeseries_errors = true,
		iterations = 50)
	
	u0 = prob.u0
	tspan = prob.tspan
	f = prob.f
	t0 = tspan[1]
	
	if dt == nothing
		error("dt must be set")
	end
	
	uElType = eltype(u0)
	tType =typeof(tspan[1])
	outdim = length(u0)
	
	hl_width = alg.hl_width
	
	phi(P,t) = u0 + (t-t0)*predict(P,t)
	dtrn = generate_data(tspan[1],tspan[2]mdt,atype = tType)
	_maxiters = iterations
	
	w = init_params(uElType, hl_width)
	lr_ = 0.1
	beta1_ = 0.9
	beta2_ = 0.95
	eps_ =1.0e-6
	prms = Any[]
	
	for i=1:length(w)
		prms = Adam(lr = lr_, beta1 = beta1_, beta2 = beta2_, eps = eps_)
		push!(prms,prm)
	end
	
	@time for iters = 1:_maxiters
		train(w,prms,dtrn,f,phi,hl_width; maxiters = 1)
		loss = test(w,dtrn,f,phi,hl_width)
		if mod(iters,100) == 0
			println((:iterations,iters,:loss,loss))
		end
		
		if loss < 1.0e-8
			break
		end
	end
	
	
	u = [phi(w,x) for x in dtrn]
	
	build_solution(prob,alg,dtrn,u, timeseries_errors = timeseries_errors, retcode = :Success)
	
end

include("training_utils.jl")