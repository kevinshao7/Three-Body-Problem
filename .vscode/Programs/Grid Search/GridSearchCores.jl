#added variable ncores
using Distributed
using DistributedArrays
using SharedArrays
@everywhere using Quadmath
using CSV
using DataFrames
#specify ncores using command -p 4




#best estimate
#Setup (Rotating Figure-Eight)
@everywhere ra =1.08105966433283395241374390321269010e+00/2
@everywhere rb = 3.45281693188283016303154284469911822e-01
@everywhere intr = [ra*2 0. 0.;
-ra rb 0.;
-ra -rb 0.]

@everywhere va = 1.09709414564358525218941225169958387e+00
@everywhere vb = 2.33529804567645806032430881887516834e-01
@everywhere vz = 9.85900000000000109601216990995453671e-02
@everywhere intv =[0. vb*2 0.;
va -vb vz ;
 -va -vb -vz]
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
r, v = Floatify(intr,intv)

@everywhere r,v


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
            if step % (resolution*10000) == 0
                println("t=",t)
            end
        end
        step +=1
        
    end

    return resolution*dt*argmin(periodicity_error[2:end]), minimum(periodicity_error[2:end]), r, v #don't return early phases when close to start
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


function phase5_together(r,v,m,order,ncores)#refine positions velocities
    sims = cld(1331,ncores)
    results = zeros(Float128, (sims*ncores, 5)) #initialize results
    searchtable = search_table()
    results[1:1331,1:3] = order*copy(searchtable)
    for i in 1:1331
        results[i,1] += r[1,1]
        results[i,2] += r[2,2]
        results[i,3] += v[2,3]
    end
    #search iteration
    for i in 1:sims
        coreintr = zeros(3,3,ncores)
        coreintv = zeros(3,3,ncores)

        for j in 1:ncores
            coreintr[:,:,j] = copy(r) #initialize core velocities
            coreintv[:,:,j] = copy(v) #initialize core velocities
            coreintr[1,1,j] = results[i+(j-1)*sims,1]#grid search parameter
            coreintr[2,1,j] = -results[i+(j-1)*sims,1]/2
            coreintr[3,1,j] = -results[i+(j-1)*sims,1]/2
            coreintr[2,2,j] = results[i+(j-1)*sims,2]#grid search parameter
            coreintr[3,2,j] = -results[i+(j-1)*sims,2]#grid search parameter
            coreintv[2,3,j] = results[i+(j-1)*sims,3]#grid search parameter
            coreintv[3,3,j] = -results[i+(j-1)*sims,3]#grid search parameter
            # println("coreintr=",coreintr[:,:,j])
            # println("coreintv=",coreintv[:,:,j])
        end

        coarse_p = zeros(1,ncores)
        coarse_e = zeros(1,ncores)
        coarse_r = zeros(3,3,ncores)
        coarse_v = zeros(3,3,ncores)
        
        #period ~ 92.8
        coarseprocess = Array{Future,1}(undef, ncores)

        for j in 1:ncores
            coarseprocess[j] = remotecall(run,j, coreintr[:,:,j], coreintv[:,:,j], m, 1e-3,92.5,1000, coreintr[:,:,j], coreintv[:,:,j])
        end
        for j in 1:ncores
            coarse_p[j],coarse_e[j],coarse_r[:,:,j],coarse_v[:,:,j] = fetch(coarseprocess[j])
            # println("coreintr=",coarse_r[:,:,j])
            # println("coreintv=",coarse_v[:,:,j])
        end
       
        fine_p = zeros(1,ncores)
        fine_e = zeros(1,ncores)
        fine_r = zeros(3,3,ncores)
        fine_v = zeros(3,3,ncores)
        fineprocess = Array{Future,1}(undef, ncores)
        for j in 1:ncores
            fineprocess[j] = remotecall(run,j, coarse_r[:,:,j], coarse_v[:,:,j], m, 1e-4,0.5,1, coreintr[:,:,j], coreintv[:,:,j])
        end
        for j in 1:ncores
            fine_p[j],fine_e[j],fine_r[:,:,j],fine_v[:,:,j] = fetch(fineprocess[j])
            # println("fineintr=",fine_r[:,:,j])
            # println("fineintv=",fine_v[:,:,j])
            # println("fineintr=",fine_p[j])
            # println("fineintv=",fine_e[j])
        end
        for j in 1:ncores
            results[i+(j-1)*sims,4] = fine_e[j]
            results[i+(j-1)*sims,5] = fine_p[j]
            
        end
        println("progress = ",i,"/",sims)
    end
    

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
    name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\GridSearchncores_3_27_1e-3.csv")
    rename!(df,[:"1RX",:"23Ry",:"23Vz" ,:"periodicity error",:"period"])
    CSV.write(name,df)

    println("DONE")
end
phase5_together(intr,intv,m,1e-3,1)

# coarse_p,coarse_e,coarse_r,coarse_v = run(r,v, m, 1e-3, 30, 1000, r, v)
# fine_p,fine_e,fine_r,fine_v = run(coarse_r,coarse_v, m, 1e-3, 80, 1, r, v)
# println("finep=",fine_p)
# println("finee=",fine_e)