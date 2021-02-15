using Distributed
using DistributedArrays
using SharedArrays

#best estimate
r = []
v = []
m = [1 1 1]
#period ~ 6.32591

#algorithms
function periodicity(r,v,intr, intv)
    perror = zeros(Float128, (1,6)) #periodicity error
    for i in 1:3
        perror[i] = sqrt((intr[i,:]-r[i,:])'*(intr[i,:]-r[i,:])) #calculate distance from original state
        perror[i+3] = sqrt((intv[i,:]-v[i,:])'*(intv[i,:]-v[i,:]))
    end
    return maximum(perror)
end

function run(r, v, dt, t_end)
    periodicity=[0] #initialize results array
    for i in 1:3,j in 1:3 #convert positions and velocities into relative perspective of body 3
        r[i,j]-=r[3,j]
        v[i,j]-=v[3,j]
    end 
    r = r[1:2,:] #discard data of body 3 (should be zero anyway)
    v = v[1:2,:]
    intr = r
    intv = v

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
        
        step +=1
        if step % 100 == 1
            #conversion to inertial frame
            results = vcat(results, periodicity(r,v,intr, intv))
            println("t=",t)
        end

        
    end
    return minimum(results), r, v
end


addprocs(4)

#creat searchtable i 1:11, j 1:11, k 1:11 
function search_table() 
    searchtable = [0 0 0]
    for i in -5:5
        for j in -5:5
            for k in -5:5
                searchtable = vcat(searchtable,[i j k])
            end
        end
    end
    return searchtable[2:end,:]
end

searchtable = search_table() #1331 cases

#iterate positions
for body in 1:3
    for depth in 1:4 #search depth
        results = zeros(Float128, (11, 11, 11)) #initialize results
        #search iteration
        for i in 1:332
            
            core1_r = r #initialize core positions 
            core2_r = r
            core3_r = r
            core4_r = r
            
            core1_r[body,:] += 10^-(depth+1)*searchtable[i,:] #grid search parameters
            core2_r[body,:] += 10^-(depth+1)*searchtable[i+332,:]
            core3_r[body,:] += 10^-(depth+1)*searchtable[i+664,:]
            core4_r[body,:] += 10^-(depth+1)*searchtable[i+996,:]

            #period ~ 6.32591
            core1 = remotecall(run,1, core1_r, v, m, 1e-3, 6.325) #coarse simulation
            core2 = remotecall(run,2, core2_r, v, m, 1e-3, 6.325)
            core3 = remotecall(run,3, core3_r, v, m, 1e-3, 6.325)
            core4 = remotecall(run,4, core4_r, v, m, 1e-3, 6.325)

            core1_p, core1_r, core1_v = fetch(core1) #fetch coarse
            core2_p, core2_r, core2_v = fetch(core2)
            core3_p, core3_r, core3_v = fetch(core3)
            core4_p, core4_r, core4_v = fetch(core4)

            core1 = remotecall(run,1, core1_r, v, m, 1e-5, 0.001) #fine simulation
            core2 = remotecall(run,2, core2_r, v, m, 1e-5, 0.001)
            core3 = remotecall(run,3, core3_r, v, m, 1e-5, 0.001)
            core4 = remotecall(run,4, core4_r, v, m, 1e-5, 0.001)

            core1_p, core1_r, core1_v = fetch(core1) #fetch fine
            core2_p, core2_r, core2_v = fetch(core2)
            core3_p, core3_r, core3_v = fetch(core3)
            core4_p, core4_r, core4_v = fetch(core4)

            results[searchtable[i,:]+[6 6 6]] = core1_p #save periodicity error into results
            results[searchtable[i+332,:]+[6 6 6]] = core2_p
            results[searchtable[i+664,:]+[6 6 6]] = core3_p
            results[searchtable[i+996,:]+[6 6 6]] = core4_p
            
        end
        #cases 1329:1331
        core1_r = r #initialize core positions 
        core2_r = r
        core3_r = r
        
        core1_r[body,:] += 10^-(depth+1)*searchtable[1329,:] #grid search parameters
        core2_r[body,:] += 10^-(depth+1)*searchtable[1330+332,:]
        core3_r[body,:] += 10^-(depth+1)*searchtable[1331+664,:]

        #period ~ 6.32591
        core1 = remotecall(run,1, core1_r, v, m, 1e-3, 6.325) #coarse simulation
        core2 = remotecall(run,2, core2_r, v, m, 1e-3, 6.325)
        core3 = remotecall(run,3, core3_r, v, m, 1e-3, 6.325)

        core1_p, core1_r, core1_v = fetch(core1) #fetch coarse
        core2_p, core2_r, core2_v = fetch(core2)
        core3_p, core3_r, core3_v = fetch(core3)

        core1 = remotecall(run,1, core1_r, core1_v, m, 1e-5, 0.001) #fine simulation
        core2 = remotecall(run,2, core2_r, core2_v, m, 1e-5, 0.001)
        core3 = remotecall(run,3, core3_r, core3_v, m, 1e-5, 0.001)

        core1_p, core1_r, core1_v = fetch(core1) #fetch fine
        core2_p, core2_r, core2_v = fetch(core2)
        core3_p, core3_r, core3_v = fetch(core3)

        results[searchtable[i,:]+[6 6 6]] = core1_p #save periodicity error into results
        results[searchtable[i+332,:]+[6 6 6]] = core2_p
        results[searchtable[i+664,:]+[6 6 6]] = core3_p


        argmin(results)
    end
end