#Kevin Shao Feb 2, 2021
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations

using Quadmath

#starting conditions
p1 = -0.93240737
p2 = -0.86473146
m = [1. 1. 1.] #masses
sum_mass = m[1]+m[2]+m[3]
dt = 1e-5 #timestep
t_end = 1 #time end
r = zeros(Float128,(3,2)) #initialize positions and vectors as Float128
v = zeros(Float128,(3,2))
intr = [0.970040	-0.24309; #data
-0.97004	0.24309;
0 0 ]
intv = [0.46620	0.43237;
0.46620	0.43237;
-0.93241 -0.86473]
for i in 1:3,j in 1:2 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end


function cross(x,y) #cross product given 2d vectors (only considering 2d space), used to calculate angular momentum
    return x[1]*y[2]-x[2]*y[1]
end

function initialize(r,v,m) #calculate initial energy and momentum
    e0 = 0  #initialize energy
    a0 = 0 #angular momentum
    m0 = m[1]*v[1,:]+m[2]*v[2,:]+m[3]*v[3,:]#initial momentum
    for x in 1:3
        e0 += 0.5*m[x]*v[x,:]'*v[x,:] #calculate kinetic energy
        a0 += cross(r[x,:],(m[x]*v[x,:])) #angular momentum
        for y in x+1:3 #we obtain all xy pairs (1,2), (1,3), and (2,3)
            xy = r[x,:]-r[y,:] #calculate relative distance
            e0 -= m[x]*m[y]/sqrt(xy'*xy) #potential energy
        end
    end
    return e0,m0,a0
end




function error(r,v,m,m0,sum_mass,e0,a0)
    energy = 0
    angular_m = 0
    inertial_v = zeros(Float128,(3,2)) #velocity in inertial frame
    inertial_r = zeros(Float128,(3,2)) #positions in inertial frame
    perror = [0. 0. 0. 0. 0. 0.] #periodicity error
    #conversion to inertial frame
    inertial_v[3,:] = (m0 - m[2]*v[2,:] - m[1]*v[1,:])/sum_mass #derived from conservation of momentum
    inertial_v[2,:] = v[2,:] + inertial_v[3,:]
    inertial_v[1,:] = v[1,:] + inertial_v[3,:]
    inertial_r[3,:] = -(m[1]*r[1,:]+m[2]*r[2,:])/sum_mass #find centre of mass
    inertial_r[2,:] = r[2,:] + inertial_r[3,:]
    inertial_r[1,:] = r[1,:] + inertial_r[3,:]
    for i in 1:3
        energy += 0.5*m[i]*inertial_v[i,:]'*inertial_v[i,:] 
        angular_m += cross(inertial_r[i,:],(m[i]*inertial_v[i,:]))
        perror[i] = sqrt((intr[i,:]-inertial_r[i,:])'*(intr[i,:]-inertial_r[i,:])) #calculate distance from original state
        perror[i+3] = sqrt((intv[i,:]-inertial_v[i,:])'*(intv[i,:]-inertial_v[i,:]))
    end
    rij = r[1,:]-r[2,:] #distance 1 to 2
    energy -= m[1]*m[3]/sqrt(r[1,:]'*r[1,:]) #potential energy
    energy -= m[2]*m[3]/sqrt(r[2,:]'*r[2,:])
    energy -= m[1]*m[2]/sqrt(rij'*rij)
    return hcat(hcat(reshape(inertial_r,(1,6)),reshape(inertial_v,(1,6))),[1e18*energy/e0-1e18 1e18*sqrt((angular_m-a0)'*(angular_m-a0)) maximum(perror)])
end




function run(r, v, dt, t_end)
    e0,m0,a0 = initialize(r,v,m)
    results=hcat(hcat([0],hcat(reshape(r,(1,6))),hcat(reshape(v,(1,6))),zeros((1,3)))) #initialize results array
    for i in 1:3,j in 1:2 #convert positions and velocities into relative perspective of body 3
        r[i,j]-=r[3,j]
        v[i,j]-=v[3,j]
    end 
    r = r[1:2,:] #discard data of body 3 (should be zero anyway)
    v = v[1:2,:]
    

    local a = zeros(Float128,(2,2))
    local jk = zeros(Float128,(2,2))
    local s = zeros(Float128,(2,2))
    local c = zeros(Float128,(2,2))

    #The following code looks far more complicated than the inertial version simply because
    #for loops don't work as well
    #Remember, body 3 is the  stationary one
    rij = r[2,:]-r[1,:] #position of body 2 from 1
    vij = v[2,:]-v[1,:]
    #organize values into arrays 31, 32, 12
    # to convert from aij to aji, aji = -aij*mi/mj
    r2 = [r[1,:]'*r[1,:], r[2,:]'*r[2,:], rij'*rij] #distances squared
    r3 = r2.*[sqrt(r2[1]), sqrt(r2[2]), sqrt(r2[3])] #distances cubed
    aij = [m[1]*r[1,:]/r3[1], m[2]*r[2,:]/r3[2], m[2]*rij/r3[3]] #31, 32, 12
    a[1,:] = aij[3] #a12
    a[1,:] -= aij[1]*m[3]/m[1] #a13
    a[1,:] -= aij[1] #-a31
    a[1,:] -= aij[2] #-a32
    a[2,:] = -aij[3]*m[1]/m[2] #a21
    a[2,:] -= aij[2]*m[3]/m[2] #a23
    a[2,:] -= aij[1] #-a31
    a[2,:] -= aij[2] #-a32
    alpha = [(r[1,:]'*v[1,:])/r2[1], (r[2,:]'*v[2,:])/r2[2], (rij'*vij)/r2[3]] #see paper for purpose of alpha, beta, gamma
    #an important property is that alphaij = alphaji
    #calculate jerk 31, 32, 12
    jkij = [m[1]*v[1,:]/r3[1]-3*alpha[1]*aij[1], m[2]*v[2,:]/r3[2]-3*alpha[2]*aij[2], m[2]*vij/r3[3]-3*alpha[3]*aij[3]]
    jk[1,:] = jkij[3] #jk12 
    jk[1,:] -= jkij[1]*m[3]/m[1] #+jk13 = -jk31*m3/m1
    jk[1,:] -= jkij[1] #-jk31
    jk[1,:] -= jkij[2] #-jk32
    jk[2,:] = -jkij[3]*m[1]/m[2] #jk21 = -jk12*m1/m2
    jk[2,:] -= jkij[2]*m[3]/m[2] #jk23 = -jk32*m3/m2
    jk[2,:] -= jkij[1] #-jk31
    jk[2,:] -= jkij[2] #-jk32

    
    #next phase: calculating snap and crackle
    taij = a[2,:] - a[1,:] #total acceleration difference 12
    tjkij = jk[2,:] - jk[1,:] #total jerk difference 12
    beta = [(v[1,:]'*v[1,:]+r[1,:]'*a[1,:])/r2[1]+alpha[1]^2, #beta coefficients
    (v[2,:]'*v[2,:]+r[2,:]'*a[2,:])/r2[2]+alpha[2]^2,
    (vij'*vij+rij'*taij)/r2[3]+alpha[3]^2]
    sij = [m[1]*a[1,:]/r3[1] - 6*alpha[1]*jkij[1] - 3*beta[1]*aij[1], #snap 31, 32, 12
    m[2]*a[2,:]/r3[2] - 6*alpha[2]*jkij[2] - 3*beta[2]*aij[2],
    m[2]*taij/r3[3] - 6*alpha[3]*jkij[3] - 3*beta[3]*aij[3]]
    s[1,:] = sij[3] #snap12
    s[1,:] -= sij[1]*m[3]/m[1] #snap13
    s[1,:] -= sij[1] #-snap31
    s[1,:] -= sij[2] #-snap32
    s[2,:] = -sij[3]*m[1]/m[2] #snap21
    s[2,:] -= sij[2]*m[3]/m[2] #snap23
    s[2,:] -= sij[1] #-snap31
    s[2,:] -= sij[2] #-snap32
    gamma = [(3*v[1,:]'*a[1,:] + r[1,:]'*jk[1,:])/r2[1] + alpha[1]*(3*beta[1]-4*alpha[1]^2),
    (3*v[2,:]'*a[2,:] + r[2,:]'*jk[2,:])/r2[2] + alpha[2]*(3*beta[2]-4*alpha[2]^2),
    (3*vij'*taij + rij'*tjkij)/r2[3] + alpha[3]*(3*beta[3]-4*alpha[3]^2)] #gamma coefficients 31, 32, 12
    cij = [m[1]*jk[1,:]/r3[1] - 9*alpha[1]*sij[1] - 9*beta[1]*jkij[1] - 3*gamma[1]*aij[1],
    m[2]*jk[2,:]/r3[2] - 9*alpha[2]*sij[2] - 9*beta[2]*jkij[2] - 3*gamma[2]*aij[2],
    m[2]*tjkij/r3[3] - 9*alpha[3]*sij[3] - 9*beta[3]*jkij[3] - 3*gamma[3]*aij[3]] #crackle 31, 32, 12
    c[1,:] = cij[3] #crackle12
    c[1,:] -= cij[1]*m[3]/m[1] #crackle13
    c[1,:] -= cij[1] #-crackle31
    c[1,:] -= cij[2] #-crackle32
    c[2,:] = -cij[3]*m[1]/m[2] #crackle21
    c[2,:] -= cij[2]*m[3]/m[2] #crackle23
    c[2,:] -= cij[1] #-crackle31
    c[2,:] -= cij[2] #-crackle32
    
    #main loop
    step = 0
    for t in 0:dt:t_end
        #save old values
        old_r = r
        old_v = v
        old_a = a
        old_jk = jk
        old_s = s
        old_c = c

        #predictor, taylor series
        r += v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120 
        v += a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24

        #calculate acceleration, jerk, snap, and crackle
        #initialize values
        a = zeros(Float128,(2,2))
        jk = zeros(Float128,(2,2))
        s = zeros(Float128,(2,2))
        c = zeros(Float128,(2,2))
        #Remember, body 3 is the  stationary one
        rij = r[2,:]-r[1,:] #position of body 2 from 1
        vij = v[2,:]-v[1,:]
        #organize values into arrays 31, 32, 12
        # to convert from aij to aji, aji = -aij*mi/mj
        r2 = [r[1,:]'*r[1,:], r[2,:]'*r[2,:], rij'*rij] #distances squared
        r3 = r2.*[sqrt(r2[1]), sqrt(r2[2]), sqrt(r2[3])] #distances cubed
        aij = [m[1]*r[1,:]/r3[1], m[2]*r[2,:]/r3[2], m[2]*rij/r3[3]] #31, 32, 12
        a[1,:] = aij[3] #a12
        a[1,:] -= aij[1]*m[3]/m[1] #a13
        a[1,:] -= aij[1] #-a31
        a[1,:] -= aij[2] #-a32
        a[2,:] = -aij[3]*m[1]/m[2] #a21
        a[2,:] -= aij[2]*m[3]/m[2] #a23
        a[2,:] -= aij[1] #-a31
        a[2,:] -= aij[2] #-a32
        alpha = [(r[1,:]'*v[1,:])/r2[1], (r[2,:]'*v[2,:])/r2[2], (rij'*vij)/r2[3]] #see paper for purpose of alpha, beta, gamma
        #an important property is that alphaij = alphaji
        #calculate jerk 31, 32, 12
        jkij = [m[1]*v[1,:]/r3[1]-3*alpha[1]*aij[1], m[2]*v[2,:]/r3[2]-3*alpha[2]*aij[2], m[2]*vij/r3[3]-3*alpha[3]*aij[3]]
        jk[1,:] = jkij[3] #jk12 
        jk[1,:] -= jkij[1]*m[3]/m[1] #+jk13 = -jk31*m3/m1
        jk[1,:] -= jkij[1] #-jk31
        jk[1,:] -= jkij[2] #-jk32
        jk[2,:] = -jkij[3]*m[1]/m[2] #jk21 = -jk12*m1/m2
        jk[2,:] -= jkij[2]*m[3]/m[2] #jk23 = -jk32*m3/m2
        jk[2,:] -= jkij[1] #-jk31
        jk[2,:] -= jkij[2] #-jk32
    
        
        #next phase: calculating snap and crackle
        taij = a[2,:] - a[1,:] #total acceleration difference 12
        tjkij = jk[2,:] - jk[1,:] #total jerk difference 12
        beta = [(v[1,:]'*v[1,:]+r[1,:]'*a[1,:])/r2[1]+alpha[1]^2, #beta coefficients
        (v[2,:]'*v[2,:]+r[2,:]'*a[2,:])/r2[2]+alpha[2]^2,
        (vij'*vij+rij'*taij)/r2[3]+alpha[3]^2]
        sij = [m[1]*a[1,:]/r3[1] - 6*alpha[1]*jkij[1] - 3*beta[1]*aij[1], #snap 31, 32, 12
        m[2]*a[2,:]/r3[2] - 6*alpha[2]*jkij[2] - 3*beta[2]*aij[2],
        m[2]*taij/r3[3] - 6*alpha[3]*jkij[3] - 3*beta[3]*aij[3]]
        s[1,:] = sij[3] #snap12
        s[1,:] -= sij[1]*m[3]/m[1] #snap13
        s[1,:] -= sij[1] #-snap31
        s[1,:] -= sij[2] #-snap32
        s[2,:] = -sij[3]*m[1]/m[2] #snap21
        s[2,:] -= sij[2]*m[3]/m[2] #snap23
        s[2,:] -= sij[1] #-snap31
        s[2,:] -= sij[2] #-snap32
        gamma = [(3*v[1,:]'*a[1,:] + r[1,:]'*jk[1,:])/r2[1] + alpha[1]*(3*beta[1]-4*alpha[1]^2),
        (3*v[2,:]'*a[2,:] + r[2,:]'*jk[2,:])/r2[2] + alpha[2]*(3*beta[2]-4*alpha[2]^2),
        (3*vij'*taij + rij'*tjkij)/r2[3] + alpha[3]*(3*beta[3]-4*alpha[3]^2)] #gamma coefficients 31, 32, 12
        cij = [m[1]*jk[1,:]/r3[1] - 9*alpha[1]*sij[1] - 9*beta[1]*jkij[1] - 3*gamma[1]*aij[1],
        m[2]*jk[2,:]/r3[2] - 9*alpha[2]*sij[2] - 9*beta[2]*jkij[2] - 3*gamma[2]*aij[2],
        m[2]*tjkij/r3[3] - 9*alpha[3]*sij[3] - 9*beta[3]*jkij[3] - 3*gamma[3]*aij[3]] #crackle 31, 32, 12
        c[1,:] = cij[3] #crackle12
        c[1,:] -= cij[1]*m[3]/m[1] #crackle13
        c[1,:] -= cij[1] #-crackle31
        c[1,:] -= cij[2] #-crackle32
        c[2,:] = -cij[3]*m[1]/m[2] #crackle21
        c[2,:] -= cij[2]*m[3]/m[2] #crackle23
        c[2,:] -= cij[1] #-crackle31
        c[2,:] -= cij[2] #-crackle32


        #corrector
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        step +=1
        if step % 100 == 1
            #conversion to inertial frame
            new = hcat(t,error(r,v,m,m0,sum_mass,e0,a0))
            results = vcat(results,new)
            println("t=",t)
        end

        
    end
    return results
end
using Plots


results = run(r,v,dt,t_end)
s = 1
e = 1000
title = plot(title=string("6 Order Hermite Relative, dt =",dt),ticks=false, labels=false,grid = false, showaxis = false, bottom_margin = -100Plots.px)
system = plot(results[s:e,2:4],results[s:e,5:7],title="System",linewidth = 3)
velocities = plot(results[s:e,8:10],results[s:e,11:13],title="Velocities",linewidth = 3)
energy = plot(results[:,1],results[:,14],title="Energy Error (1e18)",linewidth = 3)
angular_m = plot(results[:,1],results[:,15],title="Angular Momentum Error (1e18)",linewidth = 3)
periodicity = plot(results[:,1],results[:,16],title="Max Periodicity Error",linewidth = 3)
plot(title,system,velocities,energy,angular_m,periodicity,layout=(6,1),size=(500,1000))
savefig("6OrderRelative.png")

using CSV
using DataFrames
df = convert(DataFrame,results)
CSV.write("6OrderRelative.csv",df)
