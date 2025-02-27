#Kevin Shao Feb 2, 2021
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations
using Distributed
using DistributedArrays
@everywhere using Quadmath




@everywhere function cross(x,y) #cross product given 2d vectors (only considering 2d space), used to calculate angular momentum
    return x[1]*y[2]-x[2]*y[1]
end

@everywhere function initialize(r,v,m) #calculate initial energy and momentum
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




@everywhere function RelativeConversion(r,v,m,m0,sum_mass)
    inertial_v = zeros(Float128,(3,2)) #velocity in inertial frame
    inertial_r = zeros(Float128,(3,2)) #positions in inertial frame
    #conversion to inertial frame
    inertial_v[3,:] = (m0 - m[2]*v[2,:] - m[1]*v[1,:])/sum_mass #derived from conservation of momentum
    inertial_v[2,:] = v[2,:] + inertial_v[3,:]
    inertial_v[1,:] = v[1,:] + inertial_v[3,:]
    inertial_r[3,:] = -(m[1]*r[1,:]+m[2]*r[2,:])/sum_mass #find centre of mass
    inertial_r[2,:] = r[2,:] + inertial_r[3,:]
    inertial_r[1,:] = r[1,:] + inertial_r[3,:]
    return hcat(reshape(inertial_r,(1,6)),reshape(inertial_v,(1,6)))
end




@everywhere function Relative(r, v, m, dt, t_end)
    e0,m0,a0 = initialize(r,v,m)
    results=hcat(hcat([0],hcat(reshape(r,(1,6))),hcat(reshape(v,(1,6))),zeros((1,3)))) #initialize results array
    resolution = convert(Int64, round((t_end/dt)/1000, digits=0))#1000 datapoints per sim
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
        pa = a + jk*dt + s*(dt^2)/2 + c*(dt^3)/6
        pjk = jk + s*dt + c*(dt^2)/2

        #calculate acceleration, jerk, snap, and crackle
        #initialize values
        a = zeros(Float128,(2,2))
        jk = zeros(Float128,(2,2))
        s = zeros(Float128,(2,2))
        c = zeros(Float128,(2,2))
        #Remember, body 3 is the  stationary one
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
            beta = (v[i,:]'*v[i,:] + r[i,:]'*pa[i,:])/r2 + alpha^2
            s3i = m[i] * pa[i,:] / r3 - 6*alpha*jk3i - 3*beta*a3i #snap 3 to i
            s[i,:] -= s3i*m[3]/m[i] #body i to 3
            s[i,:] -= s3i #-body 3 to i
            s[3-i,:] -= s3i #-body 3 to 3-i
            gamma = (3*v[i,:]'*pa[i,:] + r[i,:]'*pjk[i,:])/r2 + alpha*(3*beta-4*alpha^2)
            c3i = m[i] * pjk[i,:] / r3 - 9*alpha*s3i - 9*beta*jk3i - 3*gamma*a3i
            c[i,:] -= c3i*m[3]/m[i] #body i to 3
            c[i,:] -= c3i #-body 3 to i
            c[3-i,:] -= c3i #-body 3 to 3-i
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
        #calculate pair 12s
        ta_12 = pa[2,:]-pa[1,:]
        tjk_12 = pjk[2,:]-pjk[1,:]
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
        
        
        if step % resolution == 0
            #conversion to inertial frame
            new = hcat(t,RelativeConversion(r,v,m,m0,sum_mass))
            results = new
            println("t=",t)
        end
        step +=1
        
    end
    return results
end




@everywhere function Inertial(r, v, m, dt, t_end)
    intr = copy(r)
    intv = copy(v)
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
        
        
        if step % resolution == 0 #record results 1000 times per sim
            new = hcat(hcat(reshape(r,(1,6)),reshape(v,(1,6))),[t])
            results = new
            println("t=",t)
        end
        step += 1
    end
    return results
end

#starting conditions
@everywhere m = [1. 1. 1.] #masses

@everywhere sum_mass = m[1]+m[2]+m[3]
@everywhere dt = 1e-5 #timestep
@everywhere t_end = 1 #time end
@everywhere r = zeros(Float128,(3,2)) #initialize positions and vectors as Float128
@everywhere v = zeros(Float128,(3,2))
@everywhere intr = [0.970040	-0.24309; #data
-0.97004	0.24309;
0 0 ]
@everywhere intv = [0.46620	0.43237;
0.46620	0.43237;
-0.93241 -0.86473]
for i in 1:3,j in 1:2 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end



@everywhere function runR(r, v, m, dt, t_end)
    return @elapsed(Relative(r, v, m, dt, t_end))
end
@everywhere function runI(r, v, m, dt, t_end)
    return @elapsed(Inertial(r, v, m, dt, t_end))
end

results = zeros((40,8))
using CSV
using DataFrames
using SharedArrays

Results = SharedArray{Float64}(results)
addprocs(4)

for  i in 3:6 
    t_end=dt*10^i #1e3, 1e4 , 1e5, 1e6 steps
    for  j in 1:13 #run 3*13+1 = 40 tests
        #run speed tests in parallel on cores 2-4, reserve core 1 as master command
        b = remotecall(runR,2, r, v, m, dt, t_end)
        c = remotecall(runR,3, r, v, m, dt, t_end)
        d = remotecall(runR,4, r, v, m, dt, t_end)
        
       
        Results[j,(i-2)*2-1] = fetch(b)
        Results[j+13,(i-2)*2-1] = fetch(c)
        Results[j+26,(i-2)*2-1] = fetch(d)
    end
    b = remotecall(runR,2, r, v, m, dt, t_end)
    Results[40,(i-2)*2-1] = fetch(b)
    println("1e",i,"step Relative done")
    for  j in 1:13
        
        b = remotecall(runI,2, r, v, m, dt, t_end)
        c = remotecall(runI,3, r, v, m, dt, t_end)
        d = remotecall(runI,4, r, v, m, dt, t_end)
        
        Results[j,(i-2)*2] = fetch(b)
        Results[j+13,(i-2)*2] = fetch(c)
        Results[j+26,(i-2)*2] = fetch(d)
    end
    b = remotecall(runI,2, r, v, m, dt, t_end)
    Results[40,(i-2)*2] = fetch(b)
    println("1e",i,"step Inertial done")
    dataframe = convert(DataFrame,Results)
    CSV.write("Balanced,Inertial.csv",dataframe)
end


