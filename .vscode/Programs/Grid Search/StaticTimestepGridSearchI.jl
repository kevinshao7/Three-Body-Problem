using Distributed
using DistributedArrays
using SharedArrays
@everywhere using Quadmath
#specify cores using command -p 4




#best estimate
@everywhere intr =[1.0969596643328337 -1.6110399993633367e-6 0.0; -0.5405568474234081 0.348481693188283 0.0; -0.5405080885054259 -0.348481693188283 0.0]


@everywhere intv =[2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.;
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 9.85900000000000109601216990995453671e-02 ;
 -1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 -9.85900000000000109601216990995453671e-02]
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
    perror = zeros(Float128, (1,6)) #periodicity error
    for i in 1:3
        perror[i] = sqrt((intr[i,:]-r[i,:])'*(intr[i,:]-r[i,:])) #calculate distance from original state
        perror[i+3] = sqrt((intv[i,:]-v[i,:])'*(intv[i,:]-v[i,:]))
    end
    return maximum(perror)
end


@everywhere function run(start_r, start_v, m, dt, t_end, resolution, intr, intv)
    r = copy(start_r) #save initial positions and velocities
    v = copy(start_v)
    periodicity_error = [0]
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
        
        
        
        
        if step % resolution == 0
            
            periodicity_error = vcat(periodicity_error, periodicity(r,v,intr, intv))
            if step % (resolution*1000) == 0
                println("t=",t)
            end
        end
        step +=1
        
    end

    return resolution*dt*argmin(periodicity_error[2:end]), minimum(periodicity_error[2:end]), r, v #don't return early phases when close to start
end





procs(4)

#creat searchtable i 1:11, j 1:11, k 1:11 




#step 1: refine angular momentum (bugged, scrapped)
#step 2: refine positions
#step 3: refine velocities

using CSV
using DataFrames



function phase1_r(r,v,m)#refine velocities
    for body in 1:3
        for depth in 1:4 #search depth
            r_results = zeros(Float128, (1333, 4)) #initialize results
            searchtable = search_table() #1331 cases


            r_results[1,1:3] = r[body,:]
            coarse_p, coarse_r, coarse_v = run(r,v,m,1e-3,92.7,1000,r,v)

            fine_p, fine_r, fine_v = run(coarse_r,coarse_v,m,1e-4,0.3,1,r,v)

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
            name = string("Phase2R/3/22,B",body,"D",depth,".csv")
            rename!(df,[:"x cord",:"y cord",:"z cord",:"periodicity error"])
            CSV.write(name,df)
        end
    end
    println("DONE")
    println("Phase 1 Positions:",r)
end

#phase1_r(r,v,m)

function phase2_v(r,v,m)#refine velocities
    for body in 1:3
        for depth in 1:4 #search depth
            v_results = zeros(Float128, (1333, 4)) #initialize results
            searchtable = search_table() #1331 cases


            v_results[1,1:3] = v[body,:]

            coarse_p, coarse_r, coarse_v = run(r,v,m,1e-3,92.7,1000,r,v)


            fine_p, fine_r, fine_v = run(coarse_r,coarse_v,m,1e-4,0.3,1,r,v)


            v_results[1,4] = fine_p
            for i in 2:1332
                v_results[i,1:3] = copy(v[body,:])
            end
            v_results[2:1332,1:3] += searchtable/10^(depth+1)
            #search iteration
            for i in 1:443
            
                core2_intv = copy(v) #initialize core velocities
                core3_intv = copy(v)
                core4_intv = copy(v)
                
                
                core2_intv[body,:] += searchtable[i,:]/10^(depth+1)#grid search parameters
                core3_intv[body,:] += searchtable[i+443,:]/10^(depth+1)
                core4_intv[body,:] += searchtable[i+886,:]/10^(depth+1)
                
                #period ~ 92.8
                coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3,92.7,1000, r, core2_intv)#coarse simulation
                coarse3 = remotecall(run,3, r, core3_intv, m,  1e-3,92.7,1000, r, core3_intv)
                coarse4 = remotecall(run,4, r, core4_intv, m,  1e-3,92.7,1000, r, core4_intv)
                
                coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
                coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
                coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)
                
                fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4,0.3,1, r, core2_intv)#fine simulation
                fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-4,0.3,1, r, core3_intv)
                fine4 = remotecall(run,4, coarse4_r, coarse4_v, m,  1e-4,0.3,1, r, core4_intv)
                
                fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch fine
                fine3_p, fine3_r, fine3_v = fetch(fine3)
                fine4_p, fine4_r, fine4_v = fetch(fine4)

                v_results[i+1, 4] = fine2_p #save periodicity error into results
                v_results[i+444, 4] = fine3_p
                v_results[i+887, 4] = fine4_p
                println("progress = ",i,"/443"," body ",body," depth ",depth)
            end
            #cases 1330:1331

            core2_intv = copy(v) #initialize core velocities
            core3_intv = copy(v)

            core2_intv[body,:] = core2_intv[body,:] .+ searchtable[1330,:]/10^(depth+1) #grid search parameters
            core3_intv[body,:] = core3_intv[body,:] .+ searchtable[1331,:]/10^(depth+1)

            #period ~ 92.8
            coarse2 = remotecall(run,2, r, core2_intv, m,  1e-3,92.7,1000, r, core2_intv) #coarse simulation
            coarse3 = remotecall(run,3, r, core3_intv, m,  1e-3,92.7,1000, r, core3_intv)

            coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
            coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)

            fine2 = remotecall(run,2, coarse2_r, coarse2_v, m,  1e-4,0.3,1, r, core2_intv) #fine simulation
            fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-4,0.3,1, r, core3_intv)

            fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch fine
            fine3_p, fine3_r, fine3_v = fetch(fine3) #fetch fine

            v_results[1331, 4] = fine2_p #save periodicity error into results
            v_results[1332, 4] = fine3_p



            sleep(2)
            row = argmin(v_results[2:1332,4])
            println(searchtable[row,1:3])
            v[body,:] += searchtable[row,1:3]/10^(depth+1) #refine position by converging on periodic solution using optimal node
            v_results[1333,1:3] = v[body,:]
            v_results[1333,4] = minimum(v_results[2:1332,4])
            
            println("DONE Body =",body," Depth =",depth)
            println("argmin =",row)
            println("minimum error =",minimum(v_results[2:1332,4]))
            df = convert(DataFrame,v_results)
            name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\Phase3V_3_22,B",body,"D",depth,".csv")
            rename!(df,[:"x cord",:"y cord",:"z cord",:"periodicity error"])
            CSV.write(name,df)
        end
    end
    println("DONE")
    println("Phase 2 Velocities:",v)
end

#phase2_v(r,v,m)

#search 2
function phase0_am(r,intv,m)#refine angular velocities
    v = copy(intv)
    for order in 3:6
        am_results = zeros(Float128, (21, 3)) #initialize results
        zrange = LinRange(-10, 10, 21)
        zrange = zrange/(10^order)
        for i in 1:21
            am_results[i,1] = copy(zrange[i])
            am_results[i,1] += v[2,3]
        end
        zarray = am_results[:,1]
        #search iteration
        for i in 1:7
        
            core2_intv = copy(v) #initialize core velocities
            core3_intv = copy(v)
            core4_intv = copy(v)
            
            core2_intv[2,3] = zarray[i]#grid search parameters
            core2_intv[3,3] = -zarray[i]
            core3_intv[2,3] = zarray[i+7]
            core3_intv[3,3] = -zarray[i+7]
            core4_intv[2,3] = zarray[i+14]
            core4_intv[3,3] = -zarray[i+14]
            
            #period ~ 92.8
            coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3,30,1000, r, core2_intv)#coarse simulation
            coarse3 = remotecall(run,3, r, core3_intv, m,  1e-3,30,1000, r, core3_intv)
            coarse4 = remotecall(run,4, r, core4_intv, m,  1e-3,30,1000, r, core4_intv)
            
            coarse2_p, coarse2_e, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
            coarse3_p, coarse3_e, coarse3_r, coarse3_v = fetch(coarse3)
            coarse4_p, coarse4_e, coarse4_r, coarse4_v = fetch(coarse4)
            
            fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-3,80,1, r, core2_intv)#fine simulation
            fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-3,80,1, r, core3_intv)
            fine4 = remotecall(run,4, coarse4_r, coarse4_v, m,  1e-3,80,1, r, core4_intv)
            
            fine2_p, fine2_e, fine2_r, fine2_v = fetch(fine2) #fetch fine
            fine3_p, fine3_e, fine3_r, fine3_v = fetch(fine3)
            fine4_p, fine4_e, fine4_r, fine4_v = fetch(fine4)

            am_results[i, 2] = fine2_e #save periodicity error into results
            am_results[i+7, 2] = fine3_e
            am_results[i+14, 2] = fine4_e
            am_results[i, 3] = fine2_p #save periodicity error into results
            am_results[i+7, 3] = fine3_p
            am_results[i+14, 3] = fine4_p
            println("progress = ",i,"/7","order = -1e",order)
            println("Core2intv = ", core2_intv)
            println(fine2_e)
            println(fine2_p)
        end
        sleep(2)
        row = argmin(am_results[:,2])
        println(am_results[row,1])
        newv = copy(v)
        newv[2,3] = am_results[row,1]
        newv[3,3] = -am_results[row,1]
        v = copy(newv)
        println("argmin =",row)
        println("z =",am_results[row,1])
        println("minimum error =",minimum(am_results[:,2]))
        df = convert(DataFrame,am_results)
        name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\Phase0AM_3_26_1e-",order,".csv")
        rename!(df,[:"Vz",:"periodicity error",:"period"])
        CSV.write(name,df)

    end

    println("DONE")
    println("Phase 0 Angular Velocities:",v)
end

#phase0_am(intr,intv,m)

@everywhere function positions_search_table() 
    searchtable = [0 0]
    for i in -10:10
        for j in -10:10
            searchtable = vcat(searchtable,[i j])
        end
    end
    return searchtable[2:end,:]
end



function phase4_r(r,v,m,order)#refine positions velocities

    am_results = zeros(Float128, (441, 4)) #initialize results
    searchtable = positions_search_table()
    for i in 1:441
        am_results[i,1:2] = order*copy(searchtable[i,1:2])
        am_results[i,1] += r[1,1]
        am_results[i,2] += r[2,2]
    end

    #search iteration
    for i in 1:147
    
        core2_intr = copy(r) #initialize core velocities
        core3_intr = copy(r)
        core4_intr = copy(r)
        
        core2_intr[1,1] = am_results[i,1]#grid search parameters
        core2_intr[2,2] = am_results[i,2]
        core2_intr[3,2] = -am_results[i,2]
        core3_intr[1,1] = am_results[i+147,1]#grid search parameters
        core3_intr[2,2] = am_results[i+147,2]
        core3_intr[3,2] = -am_results[i+147,2]
        core4_intr[1,1] = am_results[i+294,1]#grid search parameters
        core4_intr[2,2] = am_results[i+294,2]
        core4_intr[3,2] = -am_results[i+294,2]
        
        #period ~ 92.8
        coarse2 = remotecall(run,2, core2_intr, v, m, 1e-3,30,1000, core2_intr, v)#coarse simulation
        coarse3 = remotecall(run,3, core3_intr, v, m, 1e-3,30,1000, core3_intr, v)
        coarse4 = remotecall(run,4, core4_intr, v, m, 1e-3,30,1000, core4_intr, v)
        
        coarse2_p, coarse2_e, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
        coarse3_p, coarse3_e, coarse3_r, coarse3_v = fetch(coarse3)
        coarse4_p, coarse4_e, coarse4_r, coarse4_v = fetch(coarse4)
        
        fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-3,80,1, core2_intr, v)#fine simulation
        fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-3,80,1, core3_intr, v)
        fine4 = remotecall(run,4, coarse4_r, coarse4_v, m, 1e-3,80,1, core4_intr, v)
        
        fine2_p, fine2_e, fine2_r, fine2_v = fetch(fine2) #fetch fine
        fine3_p, fine3_e, fine3_r, fine3_v = fetch(fine3)
        fine4_p, fine4_e, fine4_r, fine4_v = fetch(fine4)

        am_results[i, 3] = fine2_e #save periodicity error into results
        am_results[i+147, 3] = fine3_e
        am_results[i+294, 3] = fine4_e
        am_results[i, 4] = fine2_p #save periodicity error into results
        am_results[i+147, 4] = fine3_p
        am_results[i+294, 4] = fine4_p
        println("progress = ",i,"/147")
    end

    sleep(2)
    row = argmin(am_results[:,3])
    println(am_results[row,1:2])
    newr = copy(r)
    newr[1,1] = am_results[row,1]#grid search parameters
    newr[2,2] = am_results[row,2]
    newr[3,2] = -am_results[row,2]
    println("argmin =",row)
    println("results =",am_results[row,1:2])
    println("newr =",newr)
    println("minimum error =",minimum(am_results[:,3]))
    println("period =",am_results[row,4])
    df = convert(DataFrame,am_results)
    name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\Phase4R_3_26_1e-4.csv")
    rename!(df,[:"23Ry",:"1Rx",:"periodicity error",:"period"])
    CSV.write(name,df)

    println("DONE")
end
#phase4_r(intr,intv,m,1e-4)

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


function phase5_together(r,v,m,order)#refine positions velocities

    results = zeros(Float128, (1331, 5)) #initialize results
    searchtable = search_table()
    results[:,1:3] = order*copy(searchtable)
    

    #search iteration
    for i in 1:443
    
        core2_intr = copy(r) #initialize core velocities
        core3_intr = copy(r)
        core4_intr = copy(r)
        
        core2_intr[1,1] += results[i,1]#grid search parameters
        core2_intr[2,2] += results[i,2]
        core2_intr[3,2] -= results[i,2]
        core2_intv[2,3] += results[i,3]
        core2_intv[3,3] -= results[i,3]

        core3_intr[1,1] += results[i+443,1]#grid search parameters
        core3_intr[2,2] += results[i+443,2]
        core3_intr[3,2] -= results[i+443,2]
        core3_intv[2,3] += results[i+443,3]
        core3_intv[3,3] -= results[i+443,3]

        core4_intr[1,1] += results[i+886,1]#grid search parameters
        core4_intr[2,2] += results[i+886,2]
        core4_intr[3,2] -= results[i+886,2]
        core4_intv[2,3] += results[i+886,3]
        core4_intv[3,3] -= results[i+886,3]
        
        #period ~ 92.8
        coarse2 = remotecall(run,2, core2_intr, core2_intv, m, 1e-3,30,1000, core2_intr, core2_intv)#coarse simulation
        coarse3 = remotecall(run,3, core3_intr, core3_intv, m, 1e-3,30,1000, core3_intr, core3_intv)
        coarse4 = remotecall(run,4, core4_intr, core4_intv, m, 1e-3,30,1000, core4_intr, core4_intv)
        
        coarse2_p, coarse2_e, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
        coarse3_p, coarse3_e, coarse3_r, coarse3_v = fetch(coarse3)
        coarse4_p, coarse4_e, coarse4_r, coarse4_v = fetch(coarse4)
        
        fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-3,80,1, core2_intr, core2_intv)#fine simulation
        fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-3,80,1, core3_intr, core3_intv)
        fine4 = remotecall(run,4, coarse4_r, coarse4_v, m, 1e-3,80,1, core4_intr, core4_intv)
        
        fine2_p, fine2_e, fine2_r, fine2_v = fetch(fine2) #fetch fine
        fine3_p, fine3_e, fine3_r, fine3_v = fetch(fine3)
        fine4_p, fine4_e, fine4_r, fine4_v = fetch(fine4)

        am_results[i, 4] = fine2_e #save periodicity error into results
        am_results[i+443, 4] = fine3_e
        am_results[i+886, 4] = fine4_e
        am_results[i, 5] = fine2_p #save periodicity error into results
        am_results[i+443, 5] = fine3_p
        am_results[i+886, 5] = fine4_p
        println("progress = ",i,"/443")
    end
    core2_intr = copy(r) #initialize core velocities
    core3_intr = copy(r)
    
    core2_intr[1,1] += results[1330,1]#grid search parameters
    core2_intr[2,2] += results[1330,2]
    core2_intr[3,2] -= results[1330,2]
    core2_intv[2,3] += results[1330,3]
    core2_intv[3,3] -= results[1330,3]

    core3_intr[1,1] += results[1331,1]#grid search parameters
    core3_intr[2,2] += results[1331,2]
    core3_intr[3,2] -= results[1331,2]
    core3_intv[2,3] += results[1331,3]
    core3_intv[3,3] -= results[1331,3]

    
    #period ~ 92.8
    coarse2 = remotecall(run,2, core2_intr, core2_intv, m, 1e-3,30,1000, core2_intr, core2_intv)#coarse simulation
    coarse3 = remotecall(run,3, core3_intr, core3_intv, m, 1e-3,30,1000, core3_intr, core3_intv)
    
    coarse2_p, coarse2_e, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
    coarse3_p, coarse3_e, coarse3_r, coarse3_v = fetch(coarse3)
    
    fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-3,80,1, core2_intr, core2_intv)#fine simulation
    fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-3,80,1, core3_intr, core3_intv)
    
    fine2_p, fine2_e, fine2_r, fine2_v = fetch(fine2) #fetch fine
    fine3_p, fine3_e, fine3_r, fine3_v = fetch(fine3)

    am_results[1330, 4] = fine2_e #save periodicity error into results
    am_results[1331, 4] = fine3_e
    am_results[1330, 5] = fine2_p #save periodicity error into results
    am_results[1331, 5] = fine3_p



    sleep(2)
    row = argmin(results[:,4])
    println(results[row,1:3])
    newr = copy(r)
    newr[1,1] += results[row,1]#grid search parameters
    newr[2,2] += results[row,2]
    newr[3,2] -= results[row,2]
    newv = copy(v)
    newv[2,3] += results[row,3]
    newv[3,3] -= results[row,3]
    println("argmin =",row)
    println("results =",results[row,1:3])
    println("newr =",newr)
    println("newv =",newv)
    println("minimum error =",minimum(results[:,4]))
    println("period =",results[row,5])
    df = convert(DataFrame,results)
    name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\Phase5Together_3_27_1e-3.csv")
    rename!(df,[:"1RX",:"23Ry",:"23Vz" ,:"periodicity error",:"period"])
    CSV.write(name,df)

    println("DONE")
end
#phase5_r(intr,intv,m,1e-3)