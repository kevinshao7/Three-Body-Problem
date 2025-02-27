#Kevin Shao Jan 30, 2020
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations Keigo Nitadori, Junichiro Makino

using Quadmath
using LinearAlgebra
#Setup (Rotating Figure-Eight)

# Case 5:
# On the continuation of periodic orbits from the planar to the three-dimensional general three-body problem



#BEST:


ra = 0.533486
rb = 0.339818
intr = [ra*2 0. 0.;
-ra rb 0.;
-ra -rb 0.]

va = 1.09699
vb = 0.235714
vz = 0.158752
intv =[0. vb*2 0.;
va -vb vz ;
 -va -vb -vz]

m = [1. 1. 1.]

dt = 1e-3
t_end =120
sum_mass = m[1]+m[2]+m[3]
r = zeros(Float128,(3,3)) #initialize positions and vectors as Float128
v = zeros(Float128,(3,3))
for i in 1:3,j in 1:3 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end
momentum = v[1,:]+v[2,:]+v[3,:]
intv = copy(v)
for i in 1:3
    intv[i,:] -= momentum/3
end
v =copy(intv)
print(v[1,:]+v[2,:]+v[3,:])

com = (r[1,:]+r[2,:]+r[3,:])/3
for i in 1:3
    r[i,:] -= com
end
print((r[1,:]+r[2,:]+r[3,:])/3)

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
    return hcat(hcat(reshape(r,(1,9)),reshape(v,(1,9))),hcat([1e18*energy/e0-1e18 1e18*sqrt(((linear_m-m0)'*(linear_m-m0))[1]) 1e18*sqrt((((angular_m-a0)'*(angular_m-a0)))[1]) maximum(perror)],perror))
end


function Inertial(r, v, m, dt, t_end)
    intr = copy(r) #save initial positions and velocities
    intv = copy(v)
    e0, m0, a0 = initialize(r,v,m) #calculate initial quantities
    results=hcat(hcat([0],hcat(reshape(r,(1,9))),hcat(reshape(v,(1,9))),zeros((1,10))))
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
#plot system, various errors
s = 1 #start 
e = 1124 #end
title = plot(title=string("6 Order Hermite Inertial, dt =",dt),ticks=false, labels=false, grid = false, showaxis = false, bottom_margin = -100Plots.px)
results = Inertial(r, v, m, dt, t_end)
names = ["r1" "r2" "r3"]
bodies = plot(results[s:e,2:4],results[s:e,5:7],results[s:e,8:10],title="System",label =names,linewidth = 3)
velocities = plot(results[s:e,11:13],results[s:e,14:16],results[s:e,17:19],title="Velocities",label = ["v1" "v2" "v3"],linewidth = 3)
energy = plot(results[:,1],results[:,20],title="Energy Error (1e18)",legend=false,linewidth = 3)
linear_m = plot(results[:,1],results[:,21],title="Linear Momentum Error (1e18)",legend=false,linewidth = 3)
angular_m = plot(results[:,1],results[:,22],title="Angular Momentum Error (1e18)",legend=false,linewidth = 3)
periodicity_error = plot(results[:,1],results[:,23],title="Periodicity Error",legend=false,linewidth = 3)
plot(title,bodies,velocities,energy,linear_m,angular_m,periodicity_error,layout=(7,1),size=(500,1000))
savefig("6OrderInertial3D.png")

bodies = plot(results[s:e,2:4],results[s:e,5:7],title="System",label =names,linewidth = 3)


s = 1
e = 1002
title = plot(title=string("Periodicity Error, dt =",dt),ticks=false, labels=false, grid = false, showaxis = false, bottom_margin = -100Plots.px)
r1 = plot(results[s:e,1],results[s:e,24],title="Error R1",legend=false,linewidth = 3)
r2 = plot(results[s:e,1],results[s:e,25],title="Error R2",legend=false,linewidth = 3)
r3 = plot(results[s:e,1],results[s:e,26],title="Error R3",legend=false,linewidth = 3)
v1 = plot(results[s:e,1],results[s:e,27],title="Error V1",legend=false,linewidth = 3)
v2 = plot(results[s:e,1],results[s:e,28],title="Error V2",legend=false,linewidth = 3)
v3 = plot(results[s:e,1],results[s:e,29],title="Error V3",legend=false,linewidth = 3)
plot(title,r1,r2,r3,v1,v2,v3,layout=(7,1),size=(500,1000))
savefig("6OrderInertialPeriodicity.png")

using CSV
using DataFrames
df = convert(DataFrame,results)
CSV.write("6OrderInertial.csv",df) #save integration results
