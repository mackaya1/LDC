"""Compressible Lid Driven Cavity Problem for an COMPRESSIBLE Oldroyd-B Fluid"""
"""Solution Method: Finite Element Method using DOLFIN (FEniCS)"""

"""COMPRESSIBLE TAYLOR GALERKIN METHOD"""
"""DISCRETE Elastic Viscous Split Stress (DEVSS) Stabilisation used """


from LDC_base import *  # Import Base Code for LDC Problem



"""START LOOP that runs the simulation for a range of parameters"""
"""Ensure that break control parameters are re-adjusted if solution diverges"""
"""ADJUSTIBLE PARAMETERS"""


dt = 0.002  #Time Stepping  
Tf=5.0
loopend=5
j=0
jj=0
tol=10E-6
defpar=1.0



convdef=1


# FEM Solution Convergence/Energy Plot
x1=list()
x2=list()
x3=list()
x4=list()
x5=list()
y=list()
z=list()
zz=list()
zzz=list()
zl=list()
ek1=list()
ek2=list()
ek3=list()
ek4=list()
ee1=list()
ee2=list()
ee3=list()
ee4=list()
ek5=list()
ee5=list()

#Start Solution Loop
while j < loopend:
    j+=1


    """The Following to are routines for comparing non-inertail flow with inertial flow OR Different Weissenberg Numbers at Re=0"""

    # Comparing different Speed of Sound NUMBERS Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=0, We=0.1
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=0.2
    Re=50
    if j==1:
        c0=1500
    elif j==2:
       c0=1250
    elif j==3:
       c0=1000
    elif j==4:
       c0=750
    elif j==5:
       c0=500"""

    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    conv=10E-8                                     # Non-inertial Flow Parameter (Re=0)
    Re=1.0
    if j==1:
       We=0.5
    elif j==2:
       We=0.2
    elif j==3:
       We=0.3
    elif j==4:
       We=0.4
    elif j==5:
       We=0.5
    #We=0.01             # Continuation in Weissenberg number

    # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=1.0        # We=0.1-->0.5 using continuation in We 
    if j==1:
       conv=10E-8
       Re=1
    elif j==2:
       Re=5
    elif j==3:
       Re=10
    elif j==4:
       Re=25
    elif j==5:
       Re=50"""

    # Continuation in Reynolds/Weissenberg Number Number (Re-->10Re)
    Ret=Expression('Re*(1.0+0.5*(1.0+tanh(0.7*t-4.0))*19.0)', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(0.1+(We-0.1)*0.5*(1.0+tanh(500*(t-2.5))))', t=0.0, We=We, degree=2)

    # Stabilisation Parameters
    th = (1-betav)*5*10E-2           # DEVSS Stabilisation Terms
    c1=0.1
    c2=0.5
    c3=0.05



    print '############# Fluid Characteristics ############'
    print 'Density', rho_0
    print 'Solvent Viscosity (Pa.s)', mu_1
    print 'Polymeric Viscosity (Pa.s)', mu_2
    print 'Total Viscosity (Pa.s)', mu_0
    print 'Relaxation Time (s)', lambda1
    print 'Heat Capacity', Cv
    print 'Thermal Conductivity', kappa

    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', U
    print 'Lid velocity:', (U*0.5*(1.0+tanh(e*t-3.0)),0)
    print 'Speed of sound (m/s):', c0
    print 'Mach Number', Ma
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    Np= len(p0.vector().array())
    Nv= len(w0.vector().array())   
    Ntau= len(tau0.vector().array())
    dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', d
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity/DEVSS Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh1.num_cells()
    print 'Number of Vertices:', mesh1.num_vertices()
    print 'Minimum Cell Diamter:', mesh1.hmin()
    print 'Maximum Cell Diamter:', mesh1.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', th

    #quit()

    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = 1.0
    rho0.vector()[:] = rho_array 

    # Initial Temperature Field
    T_array = T0.vector().array()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array

    # Identity Tensor   
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)


    #Define Variable Parameters, Strain Rate and other tensors
    sr0 = 0.5*(grad(u0) + transpose(grad(u0)))
    sr1 = (grad(u1) + transpose(grad(u1)))
    sr12 = 0.5*(grad(u12) + transpose(grad(u12)))
    sr = 0.5*(grad(u) + transpose(grad(u)))-1.0/3*div(u)*I
    srv = 0.5*(grad(v) + transpose(grad(v)))
    F0 = (grad(u0)*tau0 + tau0*transpose(grad(u0)))
    F12 = (grad(u12)*tau + tau*transpose(grad(u12)))
    F12R = (grad(u12)*tau12 + tau12*transpose(grad(u12)))
    F1 = (grad(u1)*tau + tau*transpose(grad(u1)))
    F1R = (grad(u1)*tau1 + tau1*transpose(grad(u1)))
    gamdots = inner(sr1,grad(u1))
    gamdots12 = inner(sr12,grad(u12))
    gamdotp = inner(tau1,grad(u1))
    gamdotp12 = inner(tau12,grad(u12))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,Qt)
    theta0 = (T0-T_0)/(T_h-T_0)
    alpha = 1.0/(rho*Cv)

    weta = We/dt                                                  #Ratio of Weissenberg number to time step

    # Artificial Diffusion Term
    #o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
    #h= p1.vector()-p0.vector()
    #m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
    #l=T1.vector()-T0.vector()
    #alt=norm(o)/(norm(tau1.vector())+10E-10)
    #alp=norm(h)/(norm(p1.vector())+10E-10)
    #alu=norm(m)/(norm(u1.vector())+10E-10)
    #alT=norm(l, 'linf')/(norm(T1.vector(),'linf')+10E-10)
    #epstau = alt*betav+10E-8                                    #Stabilisation Parameter (Stress)
    #epsp = alp*betav+10E-8                                      #Stabilisation Parameter (Pressure)
    #epsu = alu*betav+10E-8                                      #Stabilisation Parameter (Stress)
    #epsT = 0.1*alT*kappa+10E-8                                  #Stabilisation Parameter (Temperature)




    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)


    #Temperature Equation Stabilisation DEVSS
    Dt=grad(T0)
    Dt=project(Dt,V) 


    #Half Step
    a1=(Re/(dt/2.0))*inner(rho0*u,v)*dx+betav*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)\
        +th*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx-inner(D,grad(v))*dx)+(inner(D,R)*dx-inner(sr,R)*dx)     # Compressible DEVSS Term
    L1=(Re/(dt/2.0))*inner(rho0*u0,v)*dx+inner(p0,div(v))*dx-inner(tau0,grad(v))*dx-Re*conv*inner(rho0*grad(u0)*u0,v)*dx 


    #Predicted U* Equation
    a2=(Re/dt)*inner(rho0*u,v)*dx +th*(inner(grad(u),grad(v))*dx-inner(D,grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)+(inner(D,R)*dx-inner(sr,R)*dx)     # Compressible DEVSS Term
    L2=(Re/dt)*inner(rho0*u0,v)*dx-0.5*betav*(inner(grad(u0),grad(v))*dx+1.0/3*inner(div(u0),div(v))*dx) \
        +inner(p0,div(v))*dx-inner(tau0,grad(v))*dx-Re*conv*inner(rho0*grad(u12)*u12,v)*dx 
        #+ theta*inner(D,grad(v))*dx


    
    # Stress Half Step
    a3 = (2.0*We/dt)*inner(tau,R)*dx + We*(inner(dot(u12,grad(tau)),R)*dx - inner(F12, R)*dx+inner(div(u12)*tau,R)*dx)
    L3 = (2.0*We/dt)*inner(tau0,R)*dx-inner(tau0,R)*dx + 2*(1.0-betav)*(inner(D12,R)*dx) 

    # Temperature Update (Half Step)
    #a8 = (2.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u12,grad(thetal)),r)*dx 
    #L8 = (2.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u12,grad(thetar)),r)*dx \
    #      + (2.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots,r)*dx + inner(gamdotp,r)*dx - inner(p0*div(u0),r)*dx) - Di*Bi*inner(theta0,r)*ds(1) \
    # #     + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation


    #Continuity Equation 1
    a5=(Ma*Ma/(dt))*inner(p,q)*dx+0.5*dt*inner(grad(p),grad(q))*dx   #Using Dynamic Speed of Sound (c=c(x,t))
    L5=(Ma*Ma/(dt))*inner(p0,q)*dx+0.5*dt*inner(grad(p0),grad(q))*dx-(inner(rho0*div(us),q)*dx+inner(dot(grad(rho0),us),q)*dx)

    #Continuity Equation 2 
    a6=inner(rho,q)*dx 
    L6=inner(rho0,q)*dx + Ma*Ma*inner(p1-p0,q)*dx 

    #Velocity Update
    a7=(1.0/dt)*inner(Rey*rho0*u,v)*dx+0.5*betav*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)+(inner(D,R)*dx-inner(sr,R)*dx) 
    L7=(1.0/dt)*inner(Rey*rho0*us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx) #+ theta*inner(D,grad(v))*dx


    F1=dot(grad(u1),tau) + dot(tau,transpose(grad(u1)))

    # Stress Full Step ()
    a4 = (We/dt+1.0)*inner(tau,Rt)*dx+We*(inner(dot(u1,grad(tau)),Rt)*dx - inner(F1, Rt)*dx + inner(div(u1)*tau,Rt)*dx)
    L4 = (We/dt)*inner(tau0,Rt)*dx + 2*(1.0-betav)*(inner(D1,Rt)*dx)


    # Temperature Update (Full Step)
    #a9 = (1.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u1,grad(thetal)),r)*dx 
    #L9 = (1.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u1,grad(thetar)),r)*dx \
     #     + (1.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots12,r)*dx + inner(gamdotp12,r)*dx-inner(p1*div(u1),r)*dx) - Di*Bi*inner(theta0,r)*ds(1) \
      #    + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
       #   #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation



    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    A4 = assemble(a4)
    A5 = assemble(a5)
    A6 = assemble(a6)
    A7 = assemble(a7)
    #A8 = assemble(a8)
    #A9 = assemble(a9)


    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 
    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()
    z=list()

    conerr=list()
    deferr=list()
    tauerr=list()


    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 10000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        (u0, D0)=w0.split()        

        # Velocity Half Step
        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()
        
        (u12, D12)=w12.split()
        
        #Compute Predicted U* Equation
        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()


        (us, Ds) = ws.split()

        # Stress Half STEP
        #A3=assemble(a3)
        #b3=assemble(L3)
        #[bc.apply(A3, b3) for bc in bctau]
        #solve(A3, tau12.vector(), b3, "bicgstab", "default")
        #end()


        #Temperature Half Step
        #A8 = assemble(a8)
        #b8 = assemble(L8)
        #[bc.apply(A8, b8) for bc in bcT]
        #solve(A8, T12.vector(), b8, "bicgstab", "default")
        #end()

        #Continuity Equation 1
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()


        #Continuity Equation 2
        rho1=rho0+(p1-p0)/(c0*c0)
        rho1=interpolate(rho1,Q)




        #Velocity Update
        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7, "bicgstab", "default")
        end()

        (u1, D1) = w1.split()

        F1=dot(grad(u1),tau) + dot(tau,transpose(grad(u1)))

        a4 = (We/dt+1.0)*inner(tau,Rt+h_ka*dot(u1,grad(Rt)))*dx+We*(inner(dot(u1,grad(tau)),Rt+h_ka*dot(u1,grad(Rt)))*dx - inner(F1, Rt+h_ka*dot(u1,grad(Rt)))*dx\
             + inner(div(u1)*tau,Rt+h_ka*dot(u1,grad(Rt)))*dx)+inner(0.2*h_ka*grad(tau),grad(Rt))*dx
        L4 = (We/dt)*inner(tau0,Rt+h_ka*dot(u1,grad(Rt)))*dx + 2*(1.0-betav)*inner(D1,Rt+h_ka*dot(u1,grad(Rt)))*dx

        # Stress Full Step
        A4=assemble(a4)
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1.vector(), b4, "bicgstab", "default")
        end()



        #Temperature Full Step
        #A9 = assemble(a9)
        #b9 = assemble(L9)
        #[bc.apply(A9, b9) for bc in bcT]
        #solve(A9, T1.vector(), b9, "bicgstab", "default")
        #end()

        # First Normal Stress Difference
        #tau_xx=project(tau1[0,0],Q)
        #tau_xy=project(tau1[1,0],Q)
        #tau_yy=project(tau1[1,1],Q)

        #print 'Stress Norm:', norm(tau1.vector(),'linf')
        #print '12 Stress Norm:', norm(tau12.vector(),'linf')
        #print 'Velocity Norm:', norm(u1.vector(),'linf')

        # Energy Calculations
        E_k=assemble(0.5*dot(u1,u1)*dx)
        E_e=assemble(tr(tau1)*dx)

        DEFERR = max(assemble(inner(F1R,R)*dx))
        CONERR = max(assemble(inner(dot(u1,grad(tau1)),R)*dx))
        TAUERR = max(assemble(inner(tau1,R)*dx))


        # Calculate Size of Artificial Term
        #o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
        #h= p1.vector()-p0.vector()
        #m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
        #l=T1.vector()-T0.vector()



        # Record Error Data 
        
        #x.append(t)
        #y.append(norm(h,'linf')/norm(p1.vector()))
        #z.append(norm(o,'linf')/(norm(tau1.vector())+0.00000000001))
        #zz.append(norm(m,'linf')/norm(u1.vector()))
        #zzz.append(norm(l,'linf')/(norm(u1.vector())+0.0001))

        # Record Elastic & Kinetic Energy Values (Method 1)
        if j==1:
           x1.append(t)
           ek1.append(E_k)
           ee1.append(E_e)
        if j==2:
           x2.append(t)
           ek2.append(E_k)
           ee2.append(E_e)
        if j==3:
           x3.append(t)
           ek3.append(E_k)
           ee3.append(E_e)
        if j==4:
           x4.append(t)
           ek4.append(E_k)
           ee4.append(E_e)
        if j==5:
           x5.append(t)
           ek5.append(E_k)
           ee5.append(E_e)

        # Record Elastic & Kinetic Energy Values (Method 2)
        x.append(t)
        deferr.append(DEFERR)
        conerr.append(CONERR)
        tauerr.append(TAUERR)
        

        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.02/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(tau1.vector(), 'linf'),norm(p1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1.vector().array())):
            print 'FE Solution Diverging'   #Print message 
            #with open("DEVSS Weissenberg Compressible Stability.txt", "a") as text_file:
                 #text_file.write("Iteration:"+str(j)+"--- Re="+str(Rey)+", We="+str(We)+", t="+str(t)+", dt="+str(dt)+'\n')
            if j==1:           # Clear Lists
               x1=list()
               ek1=list()
               ee1=list()
            if j==2:
               x2=list()
               ek2=list()
               ee2=list()
            if j==3:
               x3=list()
               ek3=list()
               ee3=list()
            if j==4:
               x4=list()
               ek4=list()
               ee4=list()
            if j==5:
               x5=list()
               ek5=list()
               ee5=list() 
            dt=dt/2                        # Use Smaller timestep 
            j-=1                            # Extend loop
            jj+= 1                          # Convergence Failures
            Tf= (iter-10)*dt
            #c2=c2*10.0
            plt.figure(0)
            plt.plot(x, tauerr, 'r-', label=r'$\tau$')
            plt.plot(x, deferr, 'b-', label=r'Deformation Terms')
            plt.plot(x, conerr, 'g-', label=r'Convection Term')
            plt.legend(loc='best')
            plt.xlabel('time(s)')
            plt.ylabel('$||A||$')
            plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/stresserrorRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
            plt.clf()
            plt.close()
            #quit()
            break


        # Plot solution
        #if t>0.1:
            #plot(tau1[0,1], title="tau_xy Stress", rescale=True, interactive=False)
            #plot(tau1[0,0], title="tau_xx Stress", rescale=True, interactive=False)
            #plot(p1, title="Pressure", rescale=True)
            #plot(rho1, title="Density", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
           



        # Move to next time step (Continuation in Reynolds Number)
        ulid.t=t
        ulidreg.t=t
        ulidregsmall.t=t
        rampd.t=t
        #Ret.t=t
        Wet.t=t
        t += dt
        w0.assign(w1)
        T0.assign(T1)
        rho0.assign(rho1)
        p0.assign(p1)
        tau0.assign(tau1)


    # PLOTS


    # Plot Convergence Data 
    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 or jj>0:
        fig1=plt.figure()
        plt.plot(x, z, 'r-', label='Stress Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||S1-S0||/||S1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/StressCovergenceRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()"""


        #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==1 or j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$Re=0$')
        plt.plot(x2, ek2, 'b-', label=r'$Re=5$')
        plt.plot(x3, ek3, 'c-', label=r'$Re=10$')
        plt.plot(x4, ek4, 'm-', label=r'$Re=25$')
        plt.plot(x5, ek5, 'g-', label=r'$Re=30$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/We0p5KineticEnergyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$Re=0$')
        plt.plot(x2, ee2, 'b-', label=r'$Re=5$')
        plt.plot(x3, ee3, 'c-', label=r'$Re=10$')
        plt.plot(x4, ee4, 'm-', label=r'$Re=25$')
        plt.plot(x5, ee5, 'g-', label=r'$Re=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/We0p5ElasticEnergyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Speed of sound numbers at constant Weissenberg & Reynolds Numbers    
    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$c_0=1500$')
        plt.plot(x2, ek2, 'b-', label=r'$c_0=1250$')
        plt.plot(x3, ek3, 'c-', label=r'$c_0=1000$')
        plt.plot(x4, ek4, 'm-', label=r'$c_0=750$')
        plt.plot(x5, ek5, 'g-', label=r'$c_0=500$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/c0KineticEnergyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$c_0=1500$')
        plt.plot(x2, ee2, 'b-', label=r'$c_0=1250$')
        plt.plot(x3, ee3, 'c-', label=r'$c_0=1000$')
        plt.plot(x4, ee4, 'm-', label=r'$c_0=750$')
        plt.plot(x5, ee5, 'g-', label=r'$c_0=500$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/c0ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 1)  
    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6:
        if j==1:
           col='r-'
        if j==2:
           col='b-'
        if j==3:
           col='c-'
        if j==4:
           col='m-'
        if j==5:
           col='g-'
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x, ek, col, label=r'$We=%s'%We)
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/KineticEnergyTf="+str(Tf)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x, ee, col, label=r'$We=%s'%We)
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/ElasticEnergyTf="+str(Tf)+"b="+str(betav)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
    if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ek2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ek3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ek4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ek5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_{kinetic}$')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/KineticEnergyTf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ee2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ee3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ee4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ee5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_{elastic}$')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/ElasticEnergyTf="+str(Tf)+"b="+str(betav)+"mesh="+str(mm)+"c0="+str(c0)+"dt="+str(dt)+".png")
        plt.clf()




    if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==1 or j==5:

        # Plot First Normal Stress Difference
        tau_xx=project((1/small)*tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        tau_xy=project((1/small)*tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        tau_yy=project((1/small)*tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        #N1=project((1/small)*(tau1[0,0]-tau1[1,1]),Q)
        #mplot(N1)
        #plt.colorbar()
        #plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        #plt.clf()

    if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==1 or j==5:
 
       # Plot Velocity Components
        ux=project((1/small)*u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_xRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()
        uy=project((1/small)*u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_yRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()

    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6:


        # Matlab Plot of the Solution at t=Tf
        rho1=rho_0*rho1
        rho1=project(rho1,Q)
        #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
        #p1=project(p1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf() 
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()"""



    """if max(norm(tau1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6:
        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', d=d, degree=d)  #GET X-COORDINATES LIST
        y = Expression('x[1]', d=d, degree=d)  #GET Y-COORDINATES LIST
        pvals = p1.vector().array() # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().array() # GET SOLUTION T= T(x,y) list
        rhovals = rho1.vector().array() # GET SOLUTION p= p(x,y) list
        tauxxvals=tauxx.vector().array()
        xyvals = meshex.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, W)#xyvals[:,0]
        yvalsw= interpolate(y, W)#xyvals[:,1]

        xvals = xvalsq.vector().array()
        yvals = yvalsq.vector().array()


        xx = np.linspace(x0,x1)
        yy = np.linspace(y0,y1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 
        dd = mlab.griddata(xvals, yvals, rhovals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, dd, 25)
        plt.title('Density Contours')   # DENSITY CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()

        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()

        xvals = xvalsw.vector().array()
        yvals = yvalsw.vector().array()

        TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') 
        plt.contour(XX, YY, TT, 20) 
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()


        normstress = mlab.griddata(xvals, yvals, tauxxvals, xx, yy, interp='nn')

        plt.contour(XX, YY, normstress, 20) 
        plt.title('Stress Contours')   # NORMAL STRESS CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/StressContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")
        plt.clf()


        #Plot Contours USING MATPLOTLIB
        # Vector Function code

        u1=U*u1  # DIMENSIONALISED VELOCITY
        u1=project(u1,V)
        g=list()
        h=list()
        n= meshex.num_vertices()
        print(u1.vector().array())   # u is the FEM SOLUTION VECTOR IN FUNCTION SPACE 
        for i in range(len(u1.vector().array())/2-1):
            g.append(u1.vector().array()[2*i+1])
            h.append(u1.vector().array()[2*i])

        uvals = np.asarray(h) # GET SOLUTION (u,v) -> u= u(x,y) list
        vvals = np.asarray(g) # GET SOLUTION (u,v) -> v= v(x,y) list


        xy = Expression(('x[0]','x[1]'), d=d, degree=d)  #GET MESH COORDINATES LIST
        xyvalsv = interpolate(xy, V)

        q=list()
        r=list()

        for i in range(len(u1.vector().array())/2-1):
           q.append(xyvalsv.vector().array()[2*i+1])
           r.append(xyvalsv.vector().array()[2*i])

        xvals = np.asarray(r)
        yvals = np.asarray(q)

        # Interpoltate velocity field data onto matlab grid
        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

        # Interpoltate velocity field data onto matlab grid
        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

        #Determine Speed 
        speed = np.sqrt(uu*uu+ vv*vv)

        plot3 = plt.figure()
        plt.streamplot(XX, YY, uu, vv,  
                       density=3,              
                       color=speed,  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=1.0)                           # line thickness
                                                                # arrow size
        plt.colorbar()                                          # add colour bar on the right
        plt.title('Lid Driven Cavity Flow')
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"t="+str(t)+".png")   
        plt.clf()                                             # display the plot"""


    #plt.close()


    if jj==30:
       j=loopend+1
       break








