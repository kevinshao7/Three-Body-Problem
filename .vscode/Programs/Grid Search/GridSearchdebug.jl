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

@everywhere function Floatify(intr,intv)
    @everywhere r = zeros(Float128,(3,3)) #initialize positions and vectors as Float128
    @everywhere v = zeros(Float128,(3,3))
    for i in 1:3,  j in 1:3 #read data into Float128 arrays (Julia is finnicky in this way)
        r[i,j]=intr[i,j]
        v[i,j]=intv[i,j]
    end
    return r, v
end
@everywhere r, v = Floatify(intr,intv)




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


    for i in 1:3,j in 1:3 #convert positions and velocities into relative perspective of body 3
        r[i,j] = r[i,j]-r[3,j]
        v[i,j] = v[i,j]-v[3,j]
    end 

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


function phase2_r(r,v,m)#refine velocities
    for body in 1:3
        for depth in 1:4 #search depth
            r_results = zeros(Float128, (1333, 4)) #initialize results
            searchtable = search_table() #1331 cases


            r_results[1,1:3] = r[body,:]
            coarse_p, coarse_r, coarse_v = run(r,v,m,1e-3,92.7,1000,r,v)

            fine_p, fine_r, fine_v = run(r,v,m,1e-4,0.3,1,r,v)

            r_results[1,4] = fine_p
            for i in 2:1332
                r_results[i,1:3] = copy(r[body,:])
            end
            r_results[2:1332,1:3] += searchtable/10^(depth+1)
            #search iteration
            for i in 1:443
            
                core2_intr = copy(r) #initialize core velocities
                core3_intr = copy(r)
                core4_intr = copy(r)
                
                
                core2_intr[body,:] += searchtable[i,:]/10^(depth+1)#grid search parameters
                core3_intr[body,:] += searchtable[i+443,:]/10^(depth+1)
                core4_intr[body,:] += searchtable[i+886,:]/10^(depth+1)
                
                #period ~ 92.8
                coarse2 = remotecall(run,2, core2_intr, v, m, 1e-3,92.7,1000, core2_intr, v)#coarse simulation
                coarse3 = remotecall(run,3, core3_intr, v, m,  1e-3,92.7,1000, core3_intr, v)
                coarse4 = remotecall(run,4, core4_intr, v, m,  1e-3,92.7,1000, core4_intr, v)
                
                coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
                coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
                coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)
                
                fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4,0.3,1, core2_intr, v)#coarse simulation
                fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-4,0.3,1, core3_intr, v)
                fine4 = remotecall(run,4, coarse4_r, coarse4_v, m,  1e-4,0.3,1, core4_intr, v)
                
                fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch coarse
                fine3_p, fine3_r, fine3_v = fetch(fine3)
                fine4_p, fine4_r, fine4_v = fetch(fine4)

                r_results[i+1, 4] = fine2_p #save periodicity error into results
                r_results[i+444, 4] = fine3_p
                r_results[i+887, 4] = fine4_p
                println("progress = ",i,"/443")
            end
            #cases 1330:1331

            core2_intr = copy(r) #initialize core velocities
            core3_intr = copy(r)

            core2_intr[body,:] += searchtable[1330,:]/10^(depth+1) #grid search parameters
            core3_intr[body,:] += searchtable[1331,:]/10^(depth+1)

            #period ~ 92.8
            coarse2 = remotecall(run,2, core2_intr, v, m,  1e-3,92.7,1000, core2_intr, v) #coarse simulation
            coarse3 = remotecall(run,3, core3_intr, v, m,  1e-3,92.7,1000, core3_intr, v)

            coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
            coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)

            fine2 = remotecall(run,2, coarse2_r, coarse2_v, m,  1e-4,0.3,1, core2_intr, v) #coarse simulation
            fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-4,0.3,1, core2_intr, v)

            fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch coarse
            fine3_p, fine3_r, fine3_v = fetch(fine3)

            r_results[1331, 4] = fine2_p #save periodicity error into results
            r_results[1332, 4] = fine3_p



            sleep(2)
            row = argmin(r_results[2:1332,4])
            println(searchtable[row,1:3])
            r[body,:] += searchtable[row,1:3]/10^(depth+1) #refine position by converging on periodic solution using optimal node
            r_results[1333,1:3] = r[body,:]
            r_results[1333,4] = minimum(r_results[2:1332,4])
            
            println("DONE Body =",body," Depth =",depth)
            println("argmin =",row)
            println("minimum error =",minimum(r_results[2:1332,4]))
            df = convert(DataFrame,r_results)
            name = string("Phase2Rtest,B",body,"D",depth,".csv")
            rename!(df,[:"x cord",:"y cord",:"z cord",:"periodicity error"])
            CSV.write(name,df)
        end
    end
    println("DONE")
    println("Phase 2 Positions:",r)
end


function phase3_v(r,v,m)#refine velocities
    for body in 1:3
        for depth in 1:4 #search depth
            v_results = zeros(Float128, (1333, 4)) #initialize results
            searchtable = search_table() #1331 cases


            v_results[1,1:3] = v[body,:]
            
            coarse_p, coarse_r, coarse_v = run(r,v,m,1e-3,6.2,1000,r,v)
            

            fine_p, fine_r, fine_v = run(r,v,m,1e-4,0.3,1,r,v)
       
            v_results[1,4] = fine_p
            for i in 2:1332
                v_results[i,1:3] = copy(v[body,:])
            end
            v_results[2:1332,1:3] += searchtable/10^(depth+1)
            #search iteration
            for i in 1:443
            
                core2_intv = copy(v)
                core3_intv = copy(v)
                core4_intv = copy(v)

                core2_intv[body,:] = copy(v_results[i+1,1:3]) #initialize core velocities
                core3_intv[body,:] = copy(v_results[i+444,1:3])
                core4_intv[body,:] = copy(v_results[i+887,1:3])
                
            
                
                #period ~ 92.8
                coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3,6.2,1000, r, core2_intv)#coarse simulation
                coarse3 = remotecall(run,3, r, core3_intv, m,  1e-3,6.2,1000, r, core3_intv)
                coarse4 = remotecall(run,4, r, core4_intv, m,  1e-3,6.2,1000, r, core4_intv)
                
                coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
                coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
                coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)
                
                fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4,0.3,1, r, core2_intv)#coarse simulation
                fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-4,0.3,1, r, core3_intv)
                fine4 = remotecall(run,4, coarse4_r, coarse4_v, m,  1e-4,0.3,1, r, core4_intv)
                
                fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch coarse
                fine3_p, fine3_r, fine3_v = fetch(fine3)
                fine4_p, fine4_r, fine4_v = fetch(fine4)

                v_results[i+1, 4] = fine2_p #save periodicity error into results
                v_results[i+444, 4] = fine3_p
                v_results[i+887, 4] = fine4_p
                println("progress = ",i,"/443")
            end
            #cases 1330:1331

            core2_intv = copy(v)
            core3_intv = copy(v)


            core2_intv[body,:] = copy(v_results[1331,1:3]) #initialize core velocities
            core3_intv[body,:] = copy(v_results[1332,1:3])


            #period ~ 92.8
            coarse2 = remotecall(run,2, r, core2_intv, m,  1e-3,6.2,1000, r, core2_intv) #coarse simulation
            coarse3 = remotecall(run,3, r, core3_intv, m,  1e-3,6.2,1000, r, core3_intv)

            coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
            coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)

            fine2 = remotecall(run,2, coarse2_r, coarse2_v, m,  1e-4,0.3,1, r, core2_intv) #coarse simulation
            fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-4,0.3,1, r, core3_intv)

            fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch coarse
            fine3_p, fine3_r, fine3_v = fetch(fine3)

            v_results[1331, 4] = fine2_p #save periodicity error into results
            v_results[1332, 4] = fine3_p



            sleep(2)
            row = argmin(v_results[2:1332,4])
            println(searchtable[row,1:3])
            v[body,:] = searchtable[row,1:3]/10^(depth+1) #refine position by converging on periodic solution using optimal node
            v_results[1333,1:3] = v[body,:]
            v_results[1333,4] = minimum(v_results[2:1332,4])
            
            println("DONE Body =",body," Depth =",depth)
            println("argmin =",row)
            println("minimum error =",minimum(v_results[2:1332,4]))
            df = convert(DataFrame,v_results)
            name = string("Phase3Vtest2,B",body,"D",depth,".csv")
            rename!(df,[:"x cord",:"y cord",:"z cord",:"periodicity error"])
            CSV.write(name,df)
        end
    end
    println("DONE")
    println("Phase 3 Velocities:",v)
end

coarse_p, coarse_r, coarse_v = run(r,v,m,1e-3,92.7,1000,r,v)
fine_p, fine_r, fine_v = run(coarse_r, coarse_v,m,1e-4,0.3,1,r,v)