
using Distributed
using DistributedArrays
using SharedArrays
@everywhere using Quadmath
#specify cores using command -p 4




#best estimate
#Setup (Rotating Figure-Eight)
ra = 0.970040
rb = 0.24309
intr = [ra	-rb 0.;
-ra	rb 0.;
0.00000	0.00000 0.]
a = 0.46620
b = 0.43237
intv = [a b 0.;
a	b 0.;
-2a	-2b 0.]

 m = [1 1 1]
dt = 1e-4
t_end = 100
sum_mass = 3
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
phase5_together(intr,intv,m,1e-3)