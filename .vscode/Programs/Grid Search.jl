using Distributed
using DistributedArrays
using SharedArrays
@everywhere using Quadmath
#specify cores using command -p 4

#best estimate
intr = [1.08105966433283395241374390321269010e+00 -1.61103999936333666101824156054682023e-06 0.;
-5.40556847423408105134957741609652478e-01 3.45281693188283016303154284469911822e-01 0.;
-5.40508088505425823287375981275225727e-01 -3.45274810552283676957903446556133749e-01 0.]
intv = [2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.; 
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 0.1;
-1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 -0.1]
m = [1 1 1]
#period ~ 6.325913985
r = zeros(Float128,(3,3)) #initialize positions and vectors as Float128
v = zeros(Float128,(3,3))
for i in 1:3,j in 1:3 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end




#algorithms
@everywhere function periodicity(r,v,intr, intv)
    for i in 1:3,j in 1:3 #convert initial positions and velocities into relative perspective of body 3
        intr[i,j] -= intr[3,j]
        intv[i,j] -= intv[3,j]
    end 
    perror = zeros(Float128, (1,4)) #periodicity error
    for i in 1:2
        perror[i] = sqrt((intr[i,:]-r[i,:])'*(intr[i,:]-r[i,:])) #calculate distance from original state
        perror[i+2] = sqrt((intv[i,:]-v[i,:])'*(intv[i,:]-v[i,:]))
    end
    return maximum(perror)
end

@everywhere function run(r, v, m, dt, t_end, resolution, intr, intv)
    results=[0] #initialize results array (periodicity)
    m0 = m[1]*v[1,:]+m[2]*v[2,:]+m[3]*v[3,:] #system momentum

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
            
            results = vcat(results, periodicity(r,v,intr, intv))
            println("t=",t)
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

    return minimum(results[2:end]), inertial_r, inertial_v
end






procs(4)

#creat searchtable i 1:11, j 1:11, k 1:11 
@everywhere function search_table() 
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



#step 1: refine angular momentum (bugged, scrapped)
#step 2: refine positions
#step 3: refine velocities

using CSV
using DataFrames

#refine angular momentum
#best guess between 0.1 and 0.3
function phase1_am(r,v,m)
    
    @everywhere results = zeros(Float128, (2000,2)) #initialize results array
    am_results = SharedArray{Float64}(results)
    for i in 1:500
        angular_momentum = 4*(i-1)*1e-4 #ranges from 0 to 1.996
        core1_v = v #initialize core positions 
        core1_v[2,3] += angular_momentum 
        core1_v[3,3] -= angular_momentum
        core1_intv = core1_v

        core2_v = v 
        core2_v[2,3] += (angular_momentum + 1e-4)
        core2_v[3,3] -= (angular_momentum + 1e-4)
        core2_intv = core2_v

        core3_v = v 
        core3_v[2,3] += (angular_momentum + 2e-4)
        core3_v[3,3] -= (angular_momentum + 2e-4)
        core3_intv = core3_v

        core4_v = v 
        core4_v[2,3] += (angular_momentum + 3e-4)
        core4_v[3,3] -= (angular_momentum + 3e-4)
        core4_intv = core4_v
        

        #period ~ 6.32591
        coarse1 = remotecall(run,1, r, core1_v, m, 1e-3, 6.325, 1000, r, core1_intv) #coarse simulation
        coarse2 = remotecall(run,2, r, core2_v, m, 1e-3, 6.325, 1000, r, core2_intv)
        coarse3 = remotecall(run,3, r, core3_v, m, 1e-3, 6.325, 1000, r, core3_intv)
        coarse4 = remotecall(run,4, r, core4_v, m, 1e-3, 6.325, 1000, r, core4_intv)

        coarse1_p, coarse1_r, coarse1_v = fetch(coarse1) #fetch coarse
        coarse2_p, coarse2_r, coarse2_v = fetch(coarse2)
        coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
        coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)

        fine1 = @spawnat 1 run(coarse1_r, coarse1_v, m, 1e-5, 0.01, 1, r, core1_intv)#fine simulation
        fine2 = @spawnat 2 run(coarse2_r, coarse2_v, m, 1e-5, 0.01, 1, r, core2_intv)
        fine3 = @spawnat 3 run(coarse3_r, coarse3_v, m, 1e-5, 0.01, 1, r, core3_intv)
        fine4 = @spawnat 4 run(coarse4_r, coarse4_v, m, 1e-5, 0.01, 1, r, core4_intv)

        fine1_p, fine1_r, fine1_v = fetch(fine1) #fetch fine
        fine2_p, fine2_r, fine2_v = fetch(fine2)
        fine3_p, fine3_r, fine3_v = fetch(fine3)
        fine4_p, fine4_r, fine4_v = fetch(fine4)

        am_results[i*4-3,:] = [angular_momentum fine1_p] #save periodicity error into results
        am_results[i*4-2,:] = [(angular_momentum + 1e-4) fine2_p]
        am_results[i*4-1,:] = [(angular_momentum + 2e-4) fine3_p]
        am_results[i*4,:] = [(angular_momentum + 3e-4) fine4_p]
        println("Progress =",i,"/500")
    end
    println("DONE")
    println("argmin =",argmin(am_results))
    df = convert(DataFrame,am_results)
    CSV.write("Phase1_AM.csv",df)
end


function phase2_p(r,v,m)#refine positions
    for body in 1:3
        for depth in 1:4 #search depth
            r_results = zeros(Float128, (11, 11, 11)) #initialize results
            searchtable = search_table() #1331 cases
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
                core1 = remotecall(run,1, core1_r, v, m, 1e-3, 6.325, 1000) #coarse simulation
                core2 = remotecall(run,2, core2_r, v, m, 1e-3, 6.325, 1000)
                core3 = remotecall(run,3, core3_r, v, m, 1e-3, 6.325, 1000)
                core4 = remotecall(run,4, core4_r, v, m, 1e-3, 6.325, 1000)

                core1_p, core1_r, core1_v = fetch(core1) #fetch coarse
                core2_p, core2_r, core2_v = fetch(core2)
                core3_p, core3_r, core3_v = fetch(core3)
                core4_p, core4_r, core4_v = fetch(core4)

                core1 = remotecall(run,1, core1_r, core1_v, m, 1e-5, 0.001, 1) #fine simulation
                core2 = remotecall(run,2, core2_r, core2_v, m, 1e-5, 0.001, 1)
                core3 = remotecall(run,3, core3_r, core3_v, m, 1e-5, 0.001, 1)
                core4 = remotecall(run,4, core4_r, core4_v, m, 1e-5, 0.001, 1)

                core1_p, core1_r, core1_v = fetch(core1) #fetch fine
                core2_p, core2_r, core2_v = fetch(core2)
                core3_p, core3_r, core3_v = fetch(core3)
                core4_p, core4_r, core4_v = fetch(core4)

                r_results[searchtable[i,:]+[6 6 6]] = core1_p #save periodicity error into results
                r_results[searchtable[i+332,:]+[6 6 6]] = core2_p
                r_results[searchtable[i+664,:]+[6 6 6]] = core3_p
                r_results[searchtable[i+996,:]+[6 6 6]] = core4_p
                
            end
            #cases 1329:1331
            core1_r = r #initialize core positions 
            core2_r = r
            core3_r = r
            
            core1_r[body,:] += 10^-(depth+1)*searchtable[1329,:] #grid search parameters
            core2_r[body,:] += 10^-(depth+1)*searchtable[1330,:]
            core3_r[body,:] += 10^-(depth+1)*searchtable[1331,:]

            #period ~ 6.32591
            core1 = remotecall(run,1, core1_r, v, m, 1e-3, 6.325, 1000) #coarse simulation
            core2 = remotecall(run,2, core2_r, v, m, 1e-3, 6.325, 1000)
            core3 = remotecall(run,3, core3_r, v, m, 1e-3, 6.325, 1000)

            core1_p, core1_r, core1_v = fetch(core1) #fetch coarse
            core2_p, core2_r, core2_v = fetch(core2)
            core3_p, core3_r, core3_v = fetch(core3)

            core1 = remotecall(run,1, core1_r, core1_v, m, 1e-5, 0.001) #fine simulation
            core2 = remotecall(run,2, core2_r, core2_v, m, 1e-5, 0.001)
            core3 = remotecall(run,3, core3_r, core3_v, m, 1e-5, 0.001)

            core1_p, core1_r, core1_v = fetch(core1) #fetch fine
            core2_p, core2_r, core2_v = fetch(core2)
            core3_p, core3_r, core3_v = fetch(core3)

            r_results[searchtable[1329,:]+[6 6 6]] = core1_p #save periodicity error into results
            r_results[searchtable[1330,:]+[6 6 6]] = core2_p
            r_results[searchtable[1331,:]+[6 6 6]] = core3_p

            println("DONE Body =",body," Depth =",depth)
            println("argmin =",argmin(r_results))
            println("minimum error =",minimum(r_results))
            df = convert(DataFrame,r_results)
            CSV.write(("Phase2R,B",Body,"D",depth,".csv"),df)
            r[body,:] += 10^-(depth+1)*(argmin(r_results)-[6 6 6]) #refine position by converging on periodic solution
        end
    end
    println("DONE")
    println("Phase 2 Positions:",r)
end

function phase3_v(r,v,m)#refine velocities
    for body in 1:3
        for depth in 1:4 #search depth
            @everywhere v_results = zeros(Float128, (1331, 4)) #initialize results
            @everywhere searchtable = search_table() #1331 cases
            v_results[:,1:3] = searchtable
            #search iteration
            for i in 1:443
                

                core2_intv = v #initialize core velocities
                core3_intv = v
                core4_intv = v
                
                
                core2_intv[body,:] += searchtable[i,:]/10^(depth+1)#grid search parameters
                core3_intv[body,:] += searchtable[i+443,:]/10^(depth+1)
                core4_intv[body,:] += searchtable[i+886,:]/10^(depth+1)
                
                #period ~ 92.8
                coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3, 92.7, 10000, r, core2_intv)#coarse simulation
                coarse3 = remotecall(run,3, r, core3_intv, m, 1e-3, 92.7, 10000, r, core3_intv)
                coarse4 = remotecall(run,4, r, core4_intv, m, 1e-3, 92.7, 10000, r, core4_intv)
                
                coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
                coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
                coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)
                
                fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4, 0.2, 1, r, core2_intv) #fine simulation
                fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-4, 0.2, 1, r, core3_intv)
                fine4 = remotecall(run,4, coarse4_r, coarse4_v, m, 1e-4, 0.2, 1, r, core4_intv)
                
                fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch fine
                fine3_p, fine3_r, fine3_v = fetch(fine3)
                fine4_p, fine4_r, fine4_v = fetch(fine4)
                
                v_results[i, 4] = fine2_p #save periodicity error into results
                v_results[i+443, 4] = fine3_p
                v_results[i+886, 4] = fine4_p
                println("progress = ",i,"/443")
            end
            #cases 1330:1331
            core2_intv = v #initialize core velocities
            core3_intv = v

            core2_v[body,:] += searchtable[1330,:]/10^(depth+1) #grid search parameters
            core3_v[body,:] += searchtable[1331,:]/10^(depth+1)

            #period ~ 92.8
            coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3, 92.7, 10000, r, core2_intv) #coarse simulation
            coarse3 = remotecall(run,3, r, core3_intv, m, 1e-3, 92.7, 10000, r, core3_intv)

            coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
            coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)

            fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4, 0.2, 1, r, core2_intv) #fine simulation
            fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-4, 0.2, 1, r, core3_intv)

            fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch fine
            fine3_p, fine3_r, fine3_v = fetch(fine3)

            v_results[1330, 4] = fine2_p #save periodicity error into results
            v_results[1331, 4] = fine3_p

            println("DONE Body =",body," Depth =",depth)
            println("argmin =",argmin(v_results[:,4]))
            println("minimum error =",minimum(v_results[:,4]))
            df = convert(DataFrame,v_results)
            name = string("Phase3V,B",body,"D",depth,".csv")
            rename!(df,[:"x cord",:"y cord",:"z cord",:"periodicity error"])
            CSV.write(name,df)
            r = argmin(v_results[:,4])
            v[body,:] += searchtable[r,:]/10^(depth+1) #refine position by converging on periodic solution

        end
    end
    println("DONE")
    println("Phase 3 Velocities:",v)
end

phase3_v(r,v,m)