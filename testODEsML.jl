
using PyPlot
using DiffEqBase
using DifferentialEquations
using Knet

# Robert1966  problem
# This example calculates the stiff system of ODES using the native JULIA solver Rosenbrock23  and
# the simple neural network based on the ADAM algorithm. 
# Rober1966 problem deals with a system of ODEs that describes the kinetics of an auto-catalytic reaction [1]. The structure of the reactions is
# 	A →  B (k1)
# 	B + B →  B + C (k2)
# 	B + C →  A + C (k3). 
# Under some idealized assumptions [2], the following mathematical model can be set up as a set of three ODEs
# 	du[1] = -k1*u[1]                + k3*u[2]*u[3]
# 	du[2] =  k1*u[1] - k2*u[2]*u[2] - k3*u[2]*u[3]
# 	du[3] =  		   k2*u[2]*u[2]
# where u1, u2, u3 are the concentrations of species A,B,C respectively. 
# The system has the following initial conditions at time, t = 0
# 	u[1] = 1; u[2] = u[3] = 0;  
# The numerical values of the rate constants were k1 = 0.04, k2 = 3e+7 and k3 = 1e+4. 
# The large differences among the reaction rate constants provide the reason for stiffness. Originally the problem
# was proposed on the time interval 0 < t ≤ 40, but it is convenient to extend the integration
# of solution on much longer intervals due to that many codes fail if t becomes very large.

# References:
# [1] Robertson, H.H.: The Solution of a set of reaction rate equations. In:Walsh, J. (ed.) Numerical Analysis: an Introduction, pp. 178-182. Academic Press, London (1966)
# [2] Gobbert, M.K.: Robertson’s example for stiff differential equations. Arizona State University, Technical report (1996)
# [3] Lysenko, D.A., Ertesvåg, I.S., Reynolds-Averaged, Scale-Adaptive and Large-Eddy Simulations of Premixed Bluff-Body Combustion Using the Eddy Dissipation Concept, 
# Flow Turbulence Combust, 100, 721-768 (2018) https://doi.org/10.1007/s10494-017-9880-4

# Dmitry Lysenko (c) 2019 


function Robertson1966(du,u,p,t)
	
	k1 = 0.04;   
	k2 = 3.0e+7;
	k3 = 1.0e+4;
	du[1] = -k1*u[1]                + k3*u[2]*u[3];
	du[2] =  k1*u[1] - k2*u[2]*u[2] - k3*u[2]*u[3];
	du[3] =  		   k2*u[2]*u[2];
end


include("NeuralNetDiffEq2.jl")  

u0 = [1.0; 0.0; 0.0]; #initial conditions, t = 0
tspan = (0.0, 1.0e+12); #time interval 
prob = ODEProblem(Robertson1966,u0,tspan); # setup ODE problem
sol = solve(prob, Rosenbrock23(), reltol=1.0e-8,abstol=1.0e-8); #solve ODEs using Rosenbrock23 method 
solML = solve(prob,nnode(10),dt=1.0e-5,iterations =1000); #predict solution with ADAM algorithm 

# plot results: compare Rosenbrock23 vs ML (ADAM) 

figure(1)
clf;
subplot(3,1,1);
plot(sol.t, sol[1,:],solML.t,solML[1,:]);
xscale("log")
ylabel("u[1]")
grid(true)
legend(["Rosenbrock23", "ML"])

subplot(3,1,2);
plot(sol.t, sol[2,:],solML.t,solML[2,:]);
xscale("log")
ylabel("u[2]")
grid(true)
legend(["Rosenbrock23", "ML"])

subplot(3,1,3);
plot(sol.t, sol[3,:],solML.t,solML[3,:]);
xscale("log")
xlabel("time")
ylabel("u[3]")
grid(true)
legend(["Rosenbrock23", "ML"])


subplots_adjust(left=0.2)


