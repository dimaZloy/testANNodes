
# DL
# Rev 13-Feb-2020

using PyPlot; 
using DifferentialEquations;
using Flux;
using Optim;
using DiffEqFlux;

p = [0.0,0.0,0.0];

function lorenz!(du,u,p,t)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8.0/3.0)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob, Tsit5(), reltol = 1e-8, atol = 1e-8)


##n_ode = NeuralODE(lorentz!,tspan,Tsit5(),saveat=1.0)


function predict_adjoint(p) # Our 1-layer neural network
  Array(concrete_solve(prob,Tsit5(),u0,p,saveat=0.0:1:100.0))
end

function loss_adjoint(p)
  prediction = predict_adjoint(p)
  loss = sum(abs2,x-1 for x in prediction)
  loss,prediction
end

cb = function (p,l,pred) #callback function to observe training
  #display(l)
  # using `remake` to re-create our `prob` with current parameters `p`
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.0:0.1:10.0),ylim=(0,6)))

  solve(remake(prob,p=p),Tsit5(),saveat=0.0:0.1:10.0);
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end

# Display the ODE with the initial parameter values.
#cb(p,loss_adjoint(p)...)
#res = DiffEqFlux.sciml_train(loss_adjoint, p, BFGS(initial_stepnorm = 0.0001), cb = cb)

a  = solve(remake(prob,p=p),Tsit5(),saveat=0.0:0.01:100.0);

N = length(a.t);
xn = zeros(N);
yn = zeros(N);
zn = zeros(N);

for i=1:N
   xn[i] = a.u[i][1];
   yn[i] = a.u[i][2];
   zn[i] = a.u[i][3];
end


figure(3)
clf();
plot3D(sol[1,:],sol[2,:],sol[3,:],"-b",linewidth=1.5);
title("Lorentz attractor: ODE");

figure(4)
clf();
plot3D(xn,yn,zn,"-g",linewidth=1.5);
#xlabel("x");
#ylabel("y");
#zlabel("z");
title("Lorentz attractor: ML");

