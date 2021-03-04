#Kevin Shao Jan 30, 2020
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations Keigo Nitadori, Junichiro Makino

using Quadmath
using LinearAlgebra
#Setup
intr = [1.08105966433283395241374390321269010e+00 -1.61103999936333666101824156054682023e-06 0.;
-5.40556847423408105134957741609652478e-01 3.45281693188283016303154284469911822e-01 0.;
-5.40508088505425823287375981275225727e-01 -3.45274810552283676957903446556133749e-01 0.]
intv =[2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.;
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 0.;
 -1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 0.]
m = [1 1 1]
dt = 1e-4
t_end = 1e-4
sum_mass = 3
#period ~ 6.325913985
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
    intr = copy(r)
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
        pr = r + v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120
        pv = v + a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24
        pa = a + jk*dt + s*(dt^2)/2 + c*(dt^3)/6
        pjk = jk + s*dt + c*(dt^2)/2
        
        #calculate new acceleration etc. at new predicted position
        a = zeros(Float128,(3,3))
        jk = zeros(Float128,(3,3))
        s = zeros(Float128,(3,3))
        c = zeros(Float128,(3,3))
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
        println(s[1,:]-s[3,:])
        #corrector (see paper for more details)
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        
        
        if step % 100 == 0
            
            #conversion to inertial frame
            new = hcat([t],InertialError(intr,intv,r,v,m,e0,m0,a0))
            results = vcat(results,new)
            println("t=",t)
        end
        step +=1
    end
    return results
end


using Plots
s = 1
e = 1
title = plot(title=string("6 Order Hermite, dt =",dt),ticks=false, labels=false, grid = false, showaxis = false, bottom_margin = -100Plots.px)
results = Inertial(r, v, m, dt, t_end)
bodies = plot(results[s:e,2:4],results[s:e,5:7],results[s:e,8:10],title="System",linewidth = 3)
velocities = plot(results[s:e,11:13],results[s:e,14:16],results[s:e,17:19],title="Velocities",linewidth = 3)
energy = plot(results[:,1],results[:,20],title="Energy Error (1e18)",legend=false,linewidth = 3)
linear_m = plot(results[:,1],results[:,21],title="Linear Momentum Error (1e18)",legend=false,linewidth = 3)
angular_m = plot(results[:,1],results[:,22],title="Angular Momentum Error (1e18)",legend=false,linewidth = 3)
periodicity_error = plot(results[:,1],results[:,23],title="Periodicity Error",legend=false,linewidth = 3)
plot(title,bodies,velocities,energy,linear_m,angular_m,periodicity_error,layout=(7,1),size=(500,1000))
savefig("6OrderInertial3D.png")

using CSV
using DataFrames
df = convert(DataFrame,results)
CSV.write("6OrderInertial.csv",df)