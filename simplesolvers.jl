
## DL 
## rev 29 Feb 2020

## list of the ODE solvers: 
## Euler Semi-Implicit method 
## Euler method 
## Trapezoid method 
## Rodas23 method 

function EEsolve(x0, y0, dydx0_, dx)
	
    ## Simple explicit Euler method
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

    # Semi-explicit Euler method
	
	n = length(y0);
	err_ = zeros(n);
	y = zeros(n)
	##dfdx0_ = zeros(n);
	dfdx_ = zeros(n);
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
        a_[i,i] += 1.0/dx;

    end

    ## Calculate error estimate from the change in state:
    
    for i=1:n
        err_[i] = dydx0_[i] + dx*dfdx_[i];
    end
 
    err_ = a_\err_;	
    
    for i=1:n	
        y[i] = y0[i] + err_[i];
    end
   	
	maxErr_  = SIEnormalizeError(y0, y, err_);
	
	
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

