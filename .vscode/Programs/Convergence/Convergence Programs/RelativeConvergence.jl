#Kevin Shao Feb 2, 2021
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations

using Quadmath
using LinearAlgebra
#Setup (Three Dimensional Rotating Figure 8)
intr = [1.08066966433283384729277098058181084e+00 -1.55416110399993636626738281562853938e-02 4.50000000000000008012254054667877767e-04; -5.39006847423408148822462134658328736e-01 3.46431693188283000967269808362258843e-01 2.10000000000000010402377743912172292e-04; -5.40558088505425865480001016566413696e-01 -3.45324810552283650813174768062774334e-01 -3.00000000000000007600257229123386082e-05]
intv = [-1.44224756704366929443994222587166476e-02 4.68929878061247363481728886794308586e-01 -3.20000000000000007203439233993691460e-03; 1.09616414564358520570151104937817177e+00 -2.33489804567645798970612885242514878e-01 9.92000000000000055398827886188328762e-02; -1.09719166997314859330155860719924199e+00 -2.35990073493601609965543983507552106e-01 -9.74500000000000054966773320452855245e-02]

m = [1 1 1]
dt = 1e-3
t_end = 1
sum_mass = 3
r = zeros(Float128,(3,3)) #initialize positions and vectors as Float128
v = zeros(Float128,(3,3))
for i in 1:3,j in 1:3 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end


function initialize(r,v,m) #calculate initial energy and momentum
    e0 = 0  #initialize energy
    a0 = zeros(Float128,(3,1)) #angular momentum
    m0 = m[1]*v[1,:]+m[2]*v[2,:]+m[3]*v[3,:]#initial momentum
    com = (m[1]*r[1,:]+m[2]*r[2,:]+m[3]*r[3,:])/(m[1]+m[2]+m[3])
    for x in 1:3#normalize to center of mass
        r[x,:] -= com
    end
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
    angular_m = zeros(Float128,(3,1))
    inertial_v = zeros(Float128,(3,3)) #velocity in inertial frame
    inertial_r = zeros(Float128,(3,3)) #positions in inertial frame
    perror = zeros(Float128, (1,6)) #periodicity error
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
    return hcat(hcat(reshape(inertial_r,(1,9)),reshape(inertial_v,(1,9))),[1e18*energy/e0-1e18 1e18*sqrt(((angular_m-a0)'*(angular_m-a0))[1]) maximum(perror)])
end




function Relative(r, v, m, dt, t_end)
    e0,m0,a0 = initialize(r,v,m)

    results=hcat(hcat([0],hcat(reshape(r,(1,9))),hcat(reshape(v,(1,9))),zeros((1,3)))) #initialize results array
    
    for i in 1:3,j in 1:3 #convert positions and velocities into relative perspective of body 3
        r[i,j]-=r[3,j]
        v[i,j]-=v[3,j]
    end 
    r = r[1:2,:] #discard data of body 3 (should be zero anyway)
    v = v[1:2,:]
    

    local a = zeros(Float128,(2,3))
    local jk = zeros(Float128,(2,3))
    local s = zeros(Float128,(2,3))
    local c = zeros(Float128,(2,3))

    for i in 1:2 #loop through bodies 1, 2
        #calculate in relation to body 3
        r2 = r[i,:]'*r[i,:]
        r3 = r2*sqrt(r2)
        a3i = m[i] * r[i,:] / r3 #acceleration of 3 to i
        a[i,:] -= a3i*m[3]/m[i] #body i to 3
        a[i,:] -= a3i #-body 3 to i
        a[3-i,:] -= a3i #-body 3 to 3-i
        alpha = (r[i,:]'*v[i,:])/r2 #see paper for coefficients alpha, beta, and gamma
        jk3i = m[i] * v[i,:] / r3 - 3*alpha*a3i #jerk 3 to i
        jk[i,:] -= jk3i*m[3]/m[i]
        jk[i,:]  -= jk3i
        jk[3-i,:] -= jk3i
    end

    #the calculations for the pair 12 happens outside the loop because it doesn't fit in as well
    r_12 = r[2,:]-r[1,:] #relative positions 1 to 2
    v_12 = v[2,:]-v[1,:]
    r2_12 = r_12'*r_12
    r3_12 = r2_12*sqrt(r2_12)
    a_12 = m[2] * r_12 / r3_12
    a[1,:] += a_12
    a[2,:] -= a_12*m[1]/m[2]
    alpha_12 = (r_12'*v_12)/r2_12
    jk_12 = m[2] * v_12 / r3_12 - 3*alpha_12*a_12
    jk[1,:] += jk_12
    jk[2,:] -= jk_12*m[1]/m[2]
    
    for i in 1:2 #loop through bodies 1, 2
        #calculate in relation to body 3
        r2 = r[i,:]'*r[i,:]
        r3 = r2*sqrt(r2)
        a3i = m[i] * r[i,:] / r3 #acceleration of 3 to i
        alpha = (r[i,:]'*v[i,:])/r2 #see paper for coefficients alpha, beta, and gamma
        jk3i = m[i] * v[i,:] / r3 - 3*alpha*a3i #jerk 3 to i
        beta = (v[i,:]'*v[i,:] + r[i,:]'*a[i,:])/r2 + alpha^2
        s3i = m[i] * a[i,:] / r3 - 6*alpha*jk3i - 3*beta*a3i #snap 3 to i
        s[i,:] -= s3i*m[3]/m[i] #body i to 3
        s[i,:] -= s3i #-body 3 to i
        s[3-i,:] -= s3i #-body 3 to 3-i
        gamma = (3*v[i,:]'*a[i,:] + r[i,:]'*jk[i,:])/r2 + alpha*(3*beta-4*alpha^2)
        c3i = m[i] * jk[i,:] / r3 - 9*alpha*s3i - 9*beta*jk3i - 3*gamma*a3i
        c[i,:] -= c3i*m[3]/m[i] #body i to 3
        c[i,:] -= c3i #-body 3 to i
        c[3-i,:] -= c3i #-body 3 to 3-i
    end
    #calculate pair 12s
    ta_12 = a[2,:]-a[1,:]
    tjk_12 = jk[2,:]-jk[1,:]
    beta_12 = (v_12'*v_12 + r_12'*ta_12)/r2_12 + alpha_12^2
    s_12 = m[2] * ta_12 / r3_12 - 6*alpha_12*jk_12 - 3*beta_12*a_12 #snap i to j
    s[1,:] += s_12 #calculate snape of body i
    s[2,:] -= s_12*m[1]/m[2] #body j
    gamma_12 = (3*v_12'*ta_12 + r_12'*tjk_12)/r2_12 + alpha_12*(3*beta_12-4*alpha_12^2)
    c_12 = m[2] * tjk_12 / r3_12 - 9*alpha_12*s_12 - 9*beta_12*jk_12 - 3*gamma_12*a_12
    c[1,:] += c_12 #crackle of body i
    c[2,:] -= c_12*m[1]/m[2] #body j
    
    

    #main loop
    step = 0
    for t in 0:dt:t_end
        #save old values
        old_r = copy(r)
        old_v = copy(v)
        old_a = copy(a)
        old_jk = copy(jk)
        old_s = copy(s)
        old_c = copy(c)

        #predictor, taylor series
        r += v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120 
        v += a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24
        
        #calculate acceleration, jerk, snap, and crackle
        #initialize values
        a = zeros(Float128,(2,3))
        jk = zeros(Float128,(2,3))
        s = zeros(Float128,(2,3))
        c = zeros(Float128,(2,3))
        
        for i in 1:2 #loop through bodies 1, 2
            #calculate in relation to body 3
            r2 = r[i,:]'*r[i,:]
            
            r3 = r2*sqrt(r2)
            a3i = m[i] * r[i,:] / r3 #acceleration of 3 to i
            
            a[i,:] -= a3i*m[3]/m[i] #body i to 3
            a[i,:] -= a3i #-body 3 to i
            a[3-i,:] -= a3i #-body 3 to 3-i
            alpha = (r[i,:]'*v[i,:])/r2 #see paper for coefficients alpha, beta, and gamma
            jk3i = m[i] * v[i,:] / r3 - 3*alpha*a3i #jerk 3 to i
            jk[i,:] -= jk3i*m[3]/m[i]
            jk[i,:]  -= jk3i
            jk[3-i,:] -= jk3i
        end
        
        #the calculations for the pair 12 happens outside the loop because it doesn't fit in as well
        r_12 = r[2,:]-r[1,:] #relative positions 1 to 2
        v_12 = v[2,:]-v[1,:]
        r2_12 = r_12'*r_12
        r3_12 = r2_12*sqrt(r2_12)
        a_12 = m[2] * r_12 / r3_12
        a[1,:] += a_12
        a[2,:] -= a_12*m[1]/m[2]
        alpha_12 = (r_12'*v_12)/r2_12
        jk_12 = m[2] * v_12 / r3_12 - 3*alpha_12*a_12
        jk[1,:] += jk_12
        jk[2,:] -= jk_12*m[1]/m[2]

        
        for i in 1:2 #loop through bodies 1, 2
            #calculate in relation to body 3
            r2 = r[i,:]'*r[i,:]
            r3 = r2*sqrt(r2)
            a3i = m[i] * r[i,:] / r3 #acceleration of 3 to i
            alpha = (r[i,:]'*v[i,:])/r2 #see paper for coefficients alpha, beta, and gamma
            jk3i = m[i] * v[i,:] / r3 - 3*alpha*a3i #jerk 3 to i
            beta = (v[i,:]'*v[i,:] + r[i,:]'*a[i,:])/r2 + alpha^2
            s3i = m[i] * a[i,:] / r3 - 6*alpha*jk3i - 3*beta*a3i #snap 3 to i
            s[i,:] -= s3i*m[3]/m[i] #body i to 3
            s[i,:] -= s3i #-body 3 to i
            s[3-i,:] -= s3i #-body 3 to 3-i
            gamma = (3*v[i,:]'*a[i,:] + r[i,:]'*jk[i,:])/r2 + alpha*(3*beta-4*alpha^2)
            c3i = m[i] * jk[i,:] / r3 - 9*alpha*s3i - 9*beta*jk3i - 3*gamma*a3i
            c[i,:] -= c3i*m[3]/m[i] #body i to 3
            c[i,:] -= c3i #-body 3 to i
            c[3-i,:] -= c3i #-body 3 to 3-i
        end
        #calculate pair 12s
        ta_12 = a[2,:]-a[1,:]
        tjk_12 = jk[2,:]-jk[1,:]
        beta_12 = (v_12'*v_12 + r_12'*ta_12)/r2_12 + alpha_12^2
        s_12 = m[2] * ta_12 / r3_12 - 6*alpha_12*jk_12 - 3*beta_12*a_12 #snap i to j
        s[1,:] += s_12 #calculate snape of body i
        s[2,:] -= s_12*m[1]/m[2] #body j
        gamma_12 = (3*v_12'*ta_12 + r_12'*tjk_12)/r2_12 + alpha_12*(3*beta_12-4*alpha_12^2)
        c_12 = m[2] * tjk_12 / r3_12 - 9*alpha_12*s_12 - 9*beta_12*jk_12 - 3*gamma_12*a_12
        c[1,:] += c_12 #crackle of body i
        c[2,:] -= c_12*m[1]/m[2] #body j
        
        
        #corrector
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        
        if step % 1 == 0
            
            #conversion to inertial frame
            new = hcat(t,error(r,v,m,m0,sum_mass,e0,a0))
            results = vcat(results,new)
            println("t=",t)
        end
        step +=1

    end
    return results
end
using Plots

arr = zeros(Float128,(1002,8)) #1000 steps, 8 sizes of timesteps
for i in 1:8 #dt = 1e-1, 1e-2, ..... 1e-8
    dt = 1/(10^i)
    t_end = dt*1000
    results = Relative(r,v,m,dt,t_end) #integrate 1000 steps
    arr[:,i] = results[:,20] #save the energy error of each step of the sim
end

using CSV
using DataFrames
df = convert(DataFrame,arr)
CSV.write("RelativeConvergence.csv",df)
