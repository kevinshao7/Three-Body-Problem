using Distributed
using DistributedArrays
using SharedArrays
@everywhere using Quadmath

starting_r = [1.08105966433283395241374390321269010e+00 -1.61103999936333666101824156054682023e-06 0.;
-5.40556847423408105134957741609652478e-01 3.45281693188283016303154284469911822e-01 0.;
-5.40508088505425823287375981275225727e-01 -3.45274810552283676957903446556133749e-01 0.]
starting_v = [2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.; 
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 0.;
-1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 0.]
m = [1. 1. 1.]


#algorithms
@everywhere function periodicity(r,v,intr, intv)
    local initial_r = zeros(Float128, (3,3))
    local initial_v = zeros(Float128, (3,3))
    initial_r = copy(intr)
    initial_v = copy(intv)
    for i in 1:3,j in 1:3 #convert initial positions and velocities into relative perspective of body 3
        initial_r[i,j] -= initial_r[3,j]
        initial_v[i,j] -= initial_v[3,j]
    end 
    perror = zeros(Float128, (1,4)) #periodicity error
    for i in 1:2
        perror[i] = sqrt((initial_r[i,:]-r[i,:])'*(initial_r[i,:]-r[i,:])) #calculate distance from original state
        perror[i+2] = sqrt((initial_v[i,:]-v[i,:])'*(initial_v[i,:]-v[i,:]))
    end
    return maximum(perror)
end

@everywhere function run(start_r, start_v, m, dt, t_end, resolution, intr, intv)

    local r = zeros(Float128,(3,3))
    local v = zeros(Float128,(3,3))
    r = copy(start_r)
    v = copy(start_v)

    periodicity_error = [0] #initialize results array (periodicity error)


    m0 = m[1]*v[1,:]+m[2]*v[2,:]+m[3]*v[3,:] #system momentum


    println(start_r)

    for i in 1:3,j in 1:3 #convert positions and velocities into relative perspective of body 3
        r[i,j] = r[i,j]-r[3,j]
        v[i,j] = v[i,j]-v[3,j]
    end 
    println(start_r)

    local r = r[1:2,:] #discard data of body 3 (should be zero anyway)
    local v = v[1:2,:]
    
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
        
        
        if step % resolution == 0
            
            periodicity_error = vcat(periodicity_error, periodicity(r,v,intr, intv))
            if step % (resolution*100) == 0
                println("t=",t)
            end
        end
        step +=1
        
    end
    inertial_v = zeros(Float128,(3,3)) #velocity in inertial frame
    inertial_r = zeros(Float128,(3,3)) #positions in inertial frame
    sum_mass = m[1]+m[2]+m[3]
    #convert to inertial
    inertial_v[3,:] = (m0 - m[2]*v[2,:] - m[1]*v[1,:])/sum_mass #derived from conservation of momentum
    inertial_v[2,:] = v[2,:] + inertial_v[3,:]
    inertial_v[1,:] = v[1,:] + inertial_v[3,:]
    inertial_r[3,:] = -(m[1]*r[1,:]+m[2]*r[2,:])/sum_mass #find centre of mass
    inertial_r[2,:] = r[2,:] + inertial_r[3,:]
    inertial_r[1,:] = r[1,:] + inertial_r[3,:]

    return minimum(periodicity_error[2:end]), inertial_r, inertial_v #don't return early phases when close to start
end

run(starting_r, starting_v, m, 1e-3, 8, 10, r, v)
