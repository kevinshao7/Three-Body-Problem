#Kevin Shao Jan 30, 2020
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations Keigo Nitadori, Junichiro Makino

using Quadmath
#Setup

p1 = -0.93240737 #setup for Figure 8 periodic system
p2 = -0.86473146
global m = [1. 1. 1.] #Masses
dt = 1e-5 #timestep of integration
t_end = 1 #integration end
period = 6.32591398 #calculated period

r = zeros(Float128,(3,2)) #initialize arrays for positions and velocities as Float128
v = zeros(Float128,(3,2))
intr = [0.970040	-0.24309; #prepare data for positions and velocities
-0.97004	0.24309;
0.00000	0.00000]
intv = [0.46620	0.43237;
0.46620	0.43237;
-0.93241	-0.86473]
for i in 1:3,j in 1:2 #read position and velocity data into Float128 arrays
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end
intr = r #save initial positions and velocities
intv = v
results=hcat(hcat(reshape(r,(1,6)),reshape(v,(1,6))),zeros(Float128,(1,5))) #initialize array for results

function cross(x,y) #cross product given 2d vectors (only considering 2d space)
    return x[1]*y[2]-x[2]*y[1]
end

function initialize(r,v,m) #calculate initial energy and momentum
    m0 = [0,0]  #initialize linear momentum
    e0 = 0  #energy
    a0 = 0 #angular momentum
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
    linear_m = [0,0]
    angular_m = 0
    perror = [0. 0. 0. 0. 0. 0.] #periodicity error
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
    return [1e18*energy/e0-1e18 1e18*sqrt((linear_m-m0)'*(linear_m-m0)) 1e18*sqrt((angular_m-a0)'*(angular_m-a0)) maximum(perror)] #return error
end




function Inertial(r, v, m, dt, t_end)
    intr = r
    intv = v
    e0, m0, a0 = initialize(r,v,m) #calculate initial quantities
    results=hcat(hcat(reshape(r,(1,6)),reshape(v,(1,6))),zeros(Float128,(1,5))) #initialize array for results
    resolution = convert(Int64, round((t_end/dt)/1000, digits=0))#1000 datapoints per sim
    local a = zeros(Float128,(3,2)) #initialize variables
    local jk = zeros(Float128,(3,2))
    local s = zeros(Float128,(3,2))
    local c = zeros(Float128,(3,2))
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
        old_r = r #save old values
        old_v = v
        old_a = a
        old_jk = jk
        old_s = s
        old_c = c
        #predictor (Taylor series)
        pr = r + v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120
        pv = v + a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24
        pa = a + jk*dt + s*(dt^2)/2 + c*(dt^3)/6
        pjk = jk + s*dt + c*(dt^2)/2
        
        #calculate new acceleration etc. at new predicted position
        a = zeros(Float128,(3,2))
        jk = zeros(Float128,(3,2))
        s = zeros(Float128,(3,2))
        c = zeros(Float128,(3,2))
        for i in 1:3
            for j in i+1:3 
                rij = pr[j,:]-pr[i,:] 
                vij = pv[j,:]-pv[i,:]
                r2 = rij'*rij
                r3 = r2*sqrt(r2)
                a[i,:] += m[j] * rij / r3
                a[j,:] -= m[i] * rij / r3
                alpha = (rij'*vij)/r2
                aij = m[j] * rij / r3
                jk[i,:] += m[j] * vij / r3 - 3*alpha*aij
                jk[j,:] -= m[i] * vij / r3 - 3*alpha*aij
                taij = pa[j,:]-pa[i,:]
                tjkij = pjk[j,:]-pjk[i,:]
                jkij= m[j] * vij / r3 - 3*alpha*aij
                beta = (vij'*vij + rij'*taij)/r2 + alpha^2
                sij = m[j] * taij / r3 - 6*alpha*jkij - 3*beta*aij
                s[i,:] += sij
                s[j,:] -= sij * m[i] / m[j]
                gamma = (3*vij'*taij + rij'*tjkij)/r2 + alpha*(3*beta-4*alpha^2)
                c[i,:] += m[j] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij
                c[j,:] -= m[i] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij
            end
        end

        #corrector (see paper for more details)
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        
        if step % resolution == 0 #record results once every 100 timesteps
            new = hcat(hcat(reshape(r,(1,6)),reshape(v,(1,6))),hcat(t,InertialError(intr,intv,r,v,m,e0,m0,a0)))
            results = vcat(results,new)
            println("t=",t)
        end
        step += 1
    end
    return results
end


using Plots
s = 1
e = 12
title = plot(title=string("6 Order Hermite, dt =",dt),ticks=false, labels=false, grid = false, showaxis = false, bottom_margin = -100Plots.px)
results = Inertial(r, v, m, dt, t_end)
bodies = plot(results[s:e,1:3],results[s:e,4:6],title="System",linewidth = 3)
velocities = plot(results[s:e,7:9],results[s:e,10:12],title="Velocities",linewidth = 3)
energy = plot(results[:,13],results[:,14],title="Energy Error (1e18)",legend=false,linewidth = 3)
linear_m = plot(results[:,13],results[:,15],title="Linear Momentum Error (1e18)",legend=false,linewidth = 3)
angular_m = plot(results[:,13],results[:,16],title="Angular Momentum Error (1e18)",legend=false,linewidth = 3)
periodicity = plot(results[:,13],results[:,17],title="Periodicity Error",legend=false,linewidth = 3)
plot(title,bodies,velocities,energy,linear_m,angular_m,periodicity,layout=(7,1),size=(500,1000))
savefig("6Order.png")

using CSV
using DataFrames
df = convert(DataFrame,results)
CSV.write("6OrderInertial.csv",df)