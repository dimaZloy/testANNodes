
Solve stiff systems of ODEs using simple machine learning algorithms (Knet / Julia HP lang)

Robertson1966 problem deals with a system of ODEs that describes the kinetics of an auto-catalytic
reaction [1]. The structure of the reactions is

A →  B (k1)
B + B →  B + C (k2)
B + C →  A + C (k3). 

Under some idealized assumptions [2], the following mathematical model can be set up
as a set of three ODEs

du[1] = -k1*u[1]                + k3*u[2]*u[3]
du[2] =  k1*u[1] - k2*u[2]*u[2] - k3*u[2]*u[3]
du[3] =  		   k2*u[2]*u[2]

where u1, u2, u3 are the concentrations of species A,B,C respectively. 
The system has the following initial conditions at time, t = 0
 u[1] = 1; 
 u[2] = u[3] = 0;  

The numerical values of the rate constants were k1 = 0.04, k2 = 3e+7 and k3 = 1e+4. 
The large differences among the reaction rate constants provide the reason for stiffness. Originally the problem
was proposed on the time interval 0 < t ≤ 40, but it is convenient to extend the integration
of solution on much longer intervals due to that many codes fail if t becomes very large.

The provided code calculates the solutions by the Rosenbrock23 method and by the ADAM algorithm using a simple
neural network.  Results caluclated using Matlab (ode23tb), and RADAU5, LSODA are provided in [3].   

References:
[1] Robertson, H.H.: The Solution of a set of reaction rate equations. In:Walsh, J. (ed.) Numerical Analysis:
an Introduction, pp. 178-182. Academic Press, London (1966)
[2] Gobbert, M.K.: Robertson’s example for stiff differential equations. Arizona State University, Technical
report (1996)
[3] Lysenko, D.A., Ertesvåg, I.S., Reynolds-Averaged, Scale-Adaptive and Large-Eddy Simulations of Premixed Bluff-Body Combustion
Using the Eddy Dissipation Concept, Flow Turbulence Combust, 100, 721-768 (2018) https://doi.org/10.1007/s10494-017-9880-4
