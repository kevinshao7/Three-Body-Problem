#Kevin Shao Jan 30, 2020
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations Keigo Nitadori, Junichiro Makino

using Quadmath
using LinearAlgebra
#Setup (Three Dimensional Rotating Figure 8)
intr = [1.08066966433283384729277098058181084e+00 -1.55416110399993636626738281562853938e-02 4.50000000000000008012254054667877767e-04; -5.39006847423408148822462134658328736e-01 3.46431693188283000967269808362258843e-01 2.10000000000000010402377743912172292e-04; -5.40558088505425865480001016566413696e-01 -3.45324810552283650813174768062774334e-01 -3.00000000000000007600257229123386082e-05]
intv = [-1.44224756704366929443994222587166476e-02 4.68929878061247363481728886794308586e-01 -3.20000000000000007203439233993691460e-03; 1.09616414564358520570151104937817177e+00 -2.33489804567645798970612885242514878e-01 9.92000000000000055398827886188328762e-02; -1.09719166997314859330155860719924199e+00 -2.35990073493601609965543983507552106e-01 -9.74500000000000054966773320452855245e-02]

m = [1 1 1]
dt = 1e-3
t_end = 100
sum_mass = 3
r = zeros(Float128,(3,3)) #initialize positions and vectors as Float128
v = zeros(Float128,(3,3))
for i in 1:3,j in 1:3 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end


function initialize(r,v,m) #calculate initial energy and momentum
    m0 =  zeros(Float128,(3,1))  #initialize linear momentum
    e0 = 0.  #energy
    a0 =  zeros(Float128,(3,1)) #angular momentum
    com = (m[1]*r[1,:]+m[2]*r[2,:]+m[3]*r[3,:])/(m[1]+m[2]+m[3])
    for x in 1:3#normalize to center of mass
        r[x,:] -= com
    end
    for x in 1:3
        e0 += 0.5*m[x]*v[x,:]'*v[x,:] #calculate kinetic energy
        m0 += m[x]*v[x,:] #linear momentum
        a0 += cross(r[x,:],(m[x]*v[x,:])) #angular momentum
        for y in x+1:3 #we obtain all xy pairs (1,2), (1,3), and (2,3)
            xy = r[x,:]-r[y,:] #calculate relative distance
            e0 -= m[x]*m[y]/sqrt(xy'*xy) #potential energy
        end
    end
    return e0, m0, a0
end



function InertialError(intr,intv,r,v,m,e0,m0,a0)
    energy = 0 #initialize values
    linear_m = zeros(Float128,(3,1))
    angular_m = zeros(Float128,(3,1))
    perror = zeros(Float128, (1,6))
    for i in 1:3
        energy += 0.5*m[i]*v[i,:]'*v[i,:] #kinetic energy
        linear_m += m[i]*v[i,:] #linear momentum
        angular_m += cross(r[i,:],(m[i]*v[i,:])) #angular momentum
        perror[i] = sqrt((intr[i,:]-r[i,:])'*(intr[i,:]-r[i,:])) #difference from original position
        perror[i+3] = sqrt((intv[i,:]-v[i,:])'*(intv[i,:]-v[i,:])) #difference from original velocity
        for j in i+1:3
            ij = r[j,:]-r[i,:]
            energy -= m[j]*m[i]/sqrt(ij'*ij)
        end
    end
    return hcat(hcat(reshape(r,(1,9)),reshape(v,(1,9))),[1e18*energy/e0-1e18 1e18*sqrt(((linear_m-m0)'*(linear_m-m0))[1]) 1e18*sqrt((((angular_m-a0)'*(angular_m-a0)))[1]) maximum(perror)])
end


function Inertial(r, v, m, dt, t_end)
    intr = copy(r) #save initial positions and velocities
    intv = copy(v)
    e0, m0, a0 = initialize(r,v,m) #calculate initial quantities
    results=hcat(hcat([0],hcat(reshape(r,(1,9))),hcat(reshape(v,(1,9))),zeros((1,4))))
    resolution = convert(Int64, round((t_end/dt)/100, digits=0))#100 datapoints per sim
    local a = zeros(Float128,(3,3)) #initialize variables
    local jk = zeros(Float128,(3,3))
    local s = zeros(Float128,(3,3))
    local c = zeros(Float128,(3,3))
    for i in 1:3 #loop through pairs of bodies (1,2), (1,3), (2,3)
        for j in i+1:3 
            rij = r[j,:]-r[i,:] #relative positions
            vij = v[j,:]-v[i,:] #relative velocities
            r2 = rij'*rij 
            r3 = r2*sqrt(r2)
            aij = m[j] * rij / r3 #acceleration of i to j
            a[i,:] += aij #calculate acceleration of body i
            a[j,:] -= m[i]*aij/m[j] #body j
            alpha = (rij'*vij)/r2 #see paper for coefficients alpha, beta, and gamma
            jk[i,:] += m[j] * vij / r3 - 3*alpha*aij  #calculate jerk of body i
            jk[j,:] -= m[i] * vij / r3 - 3*alpha*aij #body j
        end
    end
    

    #break out of loop (acceleration and jerk must be totalled before calculating higher order derivatives)
    for i in 1:3
        for j in i+1:3 
            rij = r[j,:]-r[i,:] 
            vij = v[j,:]-v[i,:]
            r2 = rij'*rij
            r3 = r2*sqrt(r2)
            taij = a[j,:]-a[i,:] #relative acceleration
            tjkij = jk[j,:]-jk[i,:] #relative jerk
            aij = m[j] * rij / r3 #acceleration i to j
            alpha = (rij'*vij)/r2
            jkij= m[j] * vij / r3 - 3*alpha*aij #jerk i to j
            beta = (vij'*vij + rij'*taij)/r2 + alpha^2
            sij = m[j] * taij / r3 - 6*alpha*jkij - 3*beta*aij #snap i to j
            s[i,:] += sij #calculate snape of body i
            s[j,:] -= sij #body j
            gamma = (3*vij'*taij + rij'*tjkij)/r2 + alpha*(3*beta-4*alpha^2)
            c[i,:] += m[j] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij #crackle of body i
            c[j,:] -= m[i] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij #body j
        end
    end

    #main loop
    step = 0 #initialize step counter
    for t in 0:dt:t_end
        old_r = copy(r) #save old values
        old_v = copy(v)
        old_a = copy(a)
        old_jk = copy(jk)
        old_s = copy(s)
        old_c = copy(c)
        #predictor (Taylor series)
        r += v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120
        v += a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24
        # pa = a + jk*dt + s*(dt^2)/2 + c*(dt^3)/6
        # pjk = jk + s*dt + c*(dt^2)/2
        
        #calculate new acceleration etc. at new predicted position
        a = zeros(Float128,(3,3))
        jk = zeros(Float128,(3,3))
        s = zeros(Float128,(3,3))
        c = zeros(Float128,(3,3))
       
        for i in 1:3 #loop through pairs of bodies (1,2), (1,3), (2,3)
            for j in i+1:3 
                rij = r[j,:]-r[i,:] #relative positions
                vij = v[j,:]-v[i,:] #relative velocities
                r2 = rij'*rij 
                
                r3 = r2*sqrt(r2)
                aij = m[j] * rij / r3 #acceleration of i to j
                
                a[i,:] += aij #calculate acceleration of body i
                a[j,:] -= m[i]*aij/m[j] #body j
                alpha = (rij'*vij)/r2 #see paper for coefficients alpha, beta, and gamma
                jk[i,:] += m[j] * vij / r3 - 3*alpha*aij  #calculate jerk of body i
                jk[j,:] -= m[i] * vij / r3 - 3*alpha*aij #body j
            end
        end
        
        
        #break out of loop (acceleration and jerk must be totalled before calculating higher order derivatives)
        for i in 1:3
            for j in i+1:3 
                rij = r[j,:]-r[i,:] 
                vij = v[j,:]-v[i,:]
                r2 = rij'*rij
                r3 = r2*sqrt(r2)
                taij = a[j,:]-a[i,:] #relative acceleration
                tjkij = jk[j,:]-jk[i,:] #relative jerk
                aij = m[j] * rij / r3 #acceleration i to j
                alpha = (rij'*vij)/r2
                jkij= m[j] * vij / r3 - 3*alpha*aij #jerk i to j
                beta = (vij'*vij + rij'*taij)/r2 + alpha^2
                sij = m[j] * taij / r3 - 6*alpha*jkij - 3*beta*aij #snap i to j
                s[i,:] += sij #calculate snape of body i
                s[j,:] -= sij #body j
                gamma = (3*vij'*taij + rij'*tjkij)/r2 + alpha*(3*beta-4*alpha^2)
                c[i,:] += m[j] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij #crackle of body i
                c[j,:] -= m[i] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij #body j
            end
        end
        
        #corrector (see paper for more details)
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        
        
        if step % 1 == 0
            
            #conversion to inertial frame
            new = hcat([t],InertialError(intr,intv,r,v,m,e0,m0,a0))
            results = vcat(results,new)
            println("t=",t)
        end
        step +=1
    end
    return results
end

arr = zeros(Float128,(1002,8)) #1000 steps, 8 sizes of timesteps
for i in 1:8 #dt = 1e-1, 1e-2, ..... 1e-8
    dt = 1/(10^i)
    t_end = dt*1000 #integrate 1000 steps
    results = run(r,v,m,dt,t_end)
    arr[:,i] = results[:,20] #save the energy error of each step of the sim
end

using CSV
using DataFrames
df = convert(DataFrame,arr)
CSV.write("InertialConvergence.csv",df)
