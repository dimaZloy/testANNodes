
## DL 
## rev 29 Feb 2020
## EulerSI method 

using LinearAlgebra
using PyPlot
using ForwardDiff

mu = 1000.0;

vanderpol(x,y) = [ y, 1.0 * (1.0 - x*x)*y - x]; 
jac(x)=ForwardDiff.jacobian(x->vanderpol(x[1],x[2]), x);


function SIEderivatives(t,u,du)
	
	
##	k1 = 0.04;   
##	k2 = 3.0e+7;
##	k3 = 1.0e+4;
##	du[1] = -k1*u[1]                + k3*u[2]*u[3];
##	du[2] =  k1*u[1] - k2*u[2]*u[2] - k3*u[2]*u[3];
##	du[3] =  		   k2*u[2]*u[2];
	
    ##mu = 1.0;
    global mu; 
    
	du[1] = u[2];
	du[2] = mu * (1.0 - u[1]*u[1])*u[2] - u[1]; 
	
end

function SIEjacobianAnalitical(x0, y0)

  n = length(y0);
  J = zeros(n,n); 	
	
  ##mu = 1.0;	
  global mu;
	
  J[1,1] = 0.0;
  J[1,2] = 1.0;
  J[2,1] = -2.0*mu*y0[1]*y0[2]-1.0;
  J[2,2] = ( mu * (1.0-y0[1]*y0[1]));
	
 return J; 	

 ## println("J: $J");

 ##       Van der pol equations: Jacobian analytical: 	
 ##       dfdy = [ 0                       1
 ##		         (-2*mu*y(1)*y(2) - 1)  (mu*(1-y(1)^2)) ];	
	
end


function SIEjacobianNum(x0, y0)


  n = length(y0);
  yTmp1_ = zeros(n);

  uround = 1.0e-16;
  dfdx0_ = zeros(n);
  dfdx_ = zeros(n);
	
  J =  zeros(n,n);

  for i = 1:n
  
		  yTmp1_[:] = y0[:];
   
          delt = sqrt(uround*max(1.0e-5, abs(y0[i])));
		  ##delt = 1e-6; 
		  
          yTmp1_[i] = y0[i] + delt;

          SIEderivatives(x0, y0, dfdx0_);
          SIEderivatives(x0, yTmp1_, dfdx_);	

          for j = 1:n
              J[j,i] = ( dfdx_[j]  - dfdx0_[j])/delt;
			
	      end
    
  end
	
  return J; 	

end

function EEsolve(x0, y0, dydx0_, dx)
	
	## Calculate error estimate from the change in state:

	n = length(y0);
	err_ = zeros(n);
	y = zeros(n);
	
    for i=1:n
        err_[i] = dx*dydx0_[i];
    end

	
    ## Update the state
    for i=1:n
        y[i] = y0[i] + err_[i];
	end

    
	maxErr_  = SIEnormalizeError(y0, y, err_);
	
	return y, maxErr_; 
	
end


function Trapezoidsolve(x0, y0, dydx0_, dx)
	
	## Calculate error estimate from the change in state:

	n = length(y0);
	err_ = zeros(n);
	y = zeros(n);
	
    for i=1:n
        y[i] = y0[i] + dx*dydx0_[i];
    end

    SIEderivatives(x0 +dx, y0, err_);
	
    ## Update the state
    for i=1:n
        y[i] = y0[i] + 0.5*dx*(dydx0_[i] + err_[i]);
        err_[i] = 0.5*dx*(err_[i] - dydx0_[i]);
	end

    
	maxErr_  = SIEnormalizeError(y0, y, err_);
	
	return y, maxErr_; 
	
end


function SIEsolve(x0, y0, dydx0_, dx )

	
	n = length(y0);
	err_ = zeros(n);
	y = zeros(n)
	##dfdx0_ = zeros(n);
	dfdx_ = zeros(n);
	dfdy_ = zeros(n,n);
	
	a_ = zeros(n,n);

 
	###############################################
	## dfdy_ = SIEjacobianAnalitical(x0, y0);
	## dfdy_ = jac(y0);
	dfdy_ = SIEjacobianNum(x0, y0);
	
	
	
	SIEderivatives(x0+dx,y0,dfdx_);
	
    for i=1:n

        for j=1:n
            a_[i,j] = -dfdy_[i,j];
        end
        a_[i,i] += 1.0/dx;

    end

    ##LUDecompose(a_, pivotIndices_);

    ## Calculate error estimate from the change in state:
    
    for i=1:n
        err_[i] = dydx0_[i] + dx*dfdx_[i];
    end


    ##LUBacksubstitute(a_, pivotIndices_, err_);
 
    k_ = a_\err_;

	
    
    for i=1:n	
        ##y[i] = y0[i] + err_[i];
		y[i] = y0[i] + k_[i];
		
    end
   	

    ##return SIEnormalizeError(y0, y, err_);
	maxErr_  = SIEnormalizeError(y0, y, err_);
	
	##println("maxError = $maxErr_");
	
	return y, maxErr_;
	
	

end


function Rodas23solve(x0, y0, dydx0_, dx)
    
    c3 = 1.0;
    d1 = 1.0/2.0;
    d2 = 3.0/2.0;
    a31 = 2.0;
    a41 = 2.0;
    c21 = 4.0;
    c31 = 1.0;
    c32 = -1.0;
    c41 = 1.0;
    c42 = -1.0;
    c43 = -8.0/3.0;
    gamma = 1.0/2.0;
    
     n = length(y0);
    
    err_ = zeros(n);
	y = zeros(n);
    dy_ = zeros(n);
	##dfdx0_ = zeros(n);
	dfdx_ = zeros(n);
    dydx_ = zeros(n);
  
    k1_ = zeros(n);
    k2_ = zeros(n);
    k3_ = zeros(n);
    
    k1err_ = zeros(n);
	dfdy_ = zeros(n,n);
	
	a_ = zeros(n,n);
    
    
    ###############################################
	dfdy_ = SIEjacobianAnalitical(x0, y0);
	## dfdy_ = jac(y0);
	## dfdy_ = SIEjacobianNum(x0, y0);
    
    SIEderivatives(x0+dx,y0,dfdx_);
	
    for i=1:n

        for j=1:n
            a_[i,j] = -dfdy_[i,j];
        end
        a_[i,i] += 1.0/(gamma*dx);

    end
        

    ##// Calculate k1:
    for i=1:n,        
        k1_[i] = dydx0_[i] + dx*d1*dfdx_[i];
    end
    
    k1_ = a_\k1_;
    
    ## Calculate k2:
    for i=1:n,        
        k2_[i] = dydx0_[i] + dx*d2*dfdx_[i] + c21*k1_[i]/dx;
    end
    
    k2_ = a_\k2_;
    
    ## Calculate k3:
    for i=1:n,        
        dy_[i] = a31*k1_[i];
        y[i] = y0[i] + dy_[i];
    end

    ## odes_.derivatives(x0 + dx, y, dydx_);
    SIEderivatives(x0+dx,y,dydx_);
    for i=1:n,        
        k3_[i] = dydx_[i] + (c31*k1_[i] + c32*k2_[i])/dx;
    end
    
    k3_ = a_\k3_;
    
    ## Calculate new state and error
    for i=1:n,        
        dy_[i] = dy_[i] +  k3_[i];
        y[i] = y0[i] + dy_[i];
    end

    ##odes_.derivatives(x0 + dx, y, dydx_);
    SIEderivatives(x0+dx,y,dydx_);

    for i=1:n,        
        err_[i] = dydx_[i] + (c41*k1_[i] + c42*k2_[i] + c43*k3_[i])/dx;
    end
    
    err_ = a_\err_;

    for i=1:n,        
    
        y[i] = y0[i] + dy_[i] + err_[i];
        
    end

    maxErr_  = SIEnormalizeError(y0, y, err_);
	
	##println("maxError = $maxErr_");
	
	return y, maxErr_;
    
end


function Rosenbrock12solve(x0, y0, dydx0_, dx)
    
    
    gamma = 1.0 + 1.0/sqrt(2.0);
    a21 = 1.0/gamma;
    c2 = 1.0;
    c21 = -2.0/gamma;
    b1 = (3.0/2.0)/gamma;
    b2 = (1.0/2.0)/gamma;
    e1 = b1 - 1.0/gamma;
    e2 = b2;
    d1 = gamma;
    d2 = -gamma;
    
    n = length(y0);
    
    err_ = zeros(n);
	y = zeros(n);
	##dfdx0_ = zeros(n);
	dfdx_ = zeros(n);
    dydx_ = zeros(n);
  
    k1_ = zeros(n);
    k2_ = zeros(n);
    
    k1err_ = zeros(n);
	dfdy_ = zeros(n,n);
	
	a_ = zeros(n,n);
    
    
    
    ###############################################
	 dfdy_ = SIEjacobianAnalitical(x0, y0);
	## dfdy_ = jac(y0);
	## dfdy_ = SIEjacobianNum(x0, y0);
    
    
    
    SIEderivatives(x0+dx,y0,dfdx_);
	
    for i=1:n

        for j=1:n
            a_[i,j] = -dfdy_[i,j];
        end
        a_[i,i] += 1.0/(gamma*dx);

    end
        

    ##// Calculate k1:
    for i=1:n,        
        k1_[i] = dydx0_[i] + dx*d1*dfdx_[i];
    end
    
    k1_ = a_\k1_;
    

    ##// Calculate k2:
    for i=1:n,
        y[i] = y0[i] + a21*k1_[i];
    end

    
    SIEderivatives(x0+c2*dx,y0,dydx_);

    for i=1:n,
        k2_[i] = dydx_[i] + dx*d2*dfdx_[i] + c21*k1_[i]/dx;            
    end

    k2_= a_\k2_;        

    ##// Calculate error and update state:
    for i=1:n,
        y[i] = y0[i] + b1*k1_[i] + b2*k2_[i];
        err_[i] = e1*k1_[i] + e2*k2_[i];
    end

    maxErr_  = SIEnormalizeError(y0, y, err_);    

    return y, maxErr_;
end


function SIEnormalizeError( y0, y, err_ )
    
    ## Calculate the maximum error
	
	## global atol;
	## global rtol;
	
	atol = 1e-6;
	rtol = 1e-6;

	
	n = length(y0);
	
    maxErr = 0.0;

    for i=1:n	
        tol = atol  + rtol *max(abs(y0[i]), abs(y[i]));
        maxErr = max(maxErr, abs(err_[i])/tol);
    end	

    return maxErr;
end


function SIEadaptivesolve(x, y, dxTry)

    safeScale_ =  0.9;
    alphaInc_  =  0.2;
    alphaDec_ =  0.25;
    minScale_ = 0.2;
    maxScale_ = 20.0;	

    dx = dxTry;
    err = 0.0;
	n = length(y);

	## global dydx0_;
	## global yTemp_;
	
	dydx0_ = zeros(n);
	yTemp_ = zeros(n);
	
	
    ##SIEderivatives(x,y,dydx0_);
	##SIEsolve(x, y, dydx0_, dx, yTemp_,err);
	##println("t=$x, y = $y, dx = $dx, maxError = $err");
    ## Solve step and provide error estimate
   
	
	EEsolve(x, y, dydx0_, dx, yTemp_,err);
	
	return x+dx,yTemp_,dxTry;
	
end






function main()



    ## Van der Pol problem:
    ## non stiff: mu = 0; t =[0 20];
    ## stiff: mu = 1000; t =[0 3000];

	y0 = [2; 0]; #initial conditions, t = 0
	x0  = 0.0;
	xEnd = 0.0;
    
	if (floor(Int32,mu) == 1.0)
	    xEnd  =   20.0;
    elseif (floor(Int32,mu) == 1000.0)
        xEnd = 3000.0;
    end
        
	
	n_ = 2;
	maxIters = 1000; 
	dx = (xEnd-x0)/maxIters;

    t = zeros(maxIters+1);
    sol = zeros(maxIters+1,n_);

    t[1] = x0;
    sol[1,:] = y0;

	x = 0.0;
	y = 0.0;
	err = 0.0;
	
    println("iteration i=0, t=$x, y = $y0");

	for z =1:maxIters

	
	      dydx0_ = zeros(n_);
	      yTemp_ = zeros(n_);		  
	
		  x = x0;	
          y = y0;
         
	      SIEderivatives(x,y,dydx0_);
		  ##y, err = SIEsolve(x,y,dydx0_, dx);
          ## y, err = Trapezoidsolve(x0, y0, dydx0_, dx);
          y, err = Rodas23solve(x0, y0, dydx0_, dx);
          ## y, err = EEsolve(x0, y0, dydx0_, dx)
          ## y, err = Rosenbrock12solve(x,y,dydx0_, dx);
        
	
	      x0 = x + dx ;
	      y0 = y; 
	
	      t[z+1] = x0;
	      sol[z+1,:] = y0;	
	
		  if (x>=xEnd)
				return;
	      end	
	  
		  ##println("iteration i=$z, t=$x, y = $y");
		  

	end


##end


figure(1)
clf();
##subplot(3,1,1);
plot(t, sol[:,1],"or");
plot(t, sol[:,2],"ob");


##xscale("log")
ylabel("u")
grid(true)
	
end


function testJacobianComputation()
	
	x0 = 0;
	y0 = [2;0];
	n = 2;
	
	JN = zeros(n,n);
	JA = zeros(n,n);
	dfdx0 = zeros(n);
	
	
	SIEderivatives(x0,y0,dfdx0);
	
	JA = SIEjacobianAnalitical(x0, y0);
	JN = SIEjacobianNumTest(x0, y0);
	
	
	JJ = jac(y0);
	
	println("JA=$JA");
	println("JJ=$JJ");
	println("JN=$JN");
	
	##JA-JJ
	##JN - JJ
	
	
	
	##julia> f(x,y)=[x^2+y^3-1,x^4-y^4+x*y]
	##julia> j(x)=ForwardDiff.jacobian(x->f(x[1],x[2]), x)
	##julia> j([0.5;0.5])
	
end


