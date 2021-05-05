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
    searchtable = [0 0 0 0 0]
    for i in -1:1
        for j in -1:1
            for k in -1:1
                for l in -1:1
                    for m in -1:1
                        searchtable = vcat(searchtable,[i j k l m])
                    end
                end
            end
        end
    end
    return searchtable[2:end,:]
end
searchtable=search_table()
print(size(searchtable))

function phase5_together(rx,ry,vx,vy,vz,m,rx_step,ry_step,vx_step,vy_step,vz_step,ncores)#refine positions velocities
    sims = cld(243,ncores)
    results = zeros(Float128, (sims*ncores, 7)) #initialize results
    searchtable = search_table()

    for iteration in 1:5
        for i in 1:243
            results[i,1] = rx + rx*(0.5^rx_step)*searchtable[i,1]
            results[i,2] = ry + ry*(0.5^ry_step)*searchtable[i,2]
            results[i,3] = vx + vx*(0.5^vx_step)*searchtable[i,3]
            results[i,4] = vy + vy*(0.5^vy_step)*searchtable[i,4]
            results[i,5] = vz + vz*(0.5^vz_step)*searchtable[i,5]
        end
        #search iteration
        for i in 1:sims
            coreintr = zeros(3,3,ncores)
            coreintv = zeros(3,3,ncores)

            for j in 1:ncores


                coreintr[1,1,j] = results[i+(j-1)*sims,1]*2#grid search parameter
                coreintr[2,1,j] = -results[i+(j-1)*sims,1]
                coreintr[3,1,j] = -results[i+(j-1)*sims,1]

                coreintr[1,2,j] = 0.
                coreintr[2,2,j] = results[i+(j-1)*sims,2]
                coreintr[3,2,j] = -results[i+(j-1)*sims,2]

                coreintr[1,3,j] = 0.
                coreintr[2,3,j] = 0.
                coreintr[3,3,j] = 0.

                coreintv[1,1,j] = 0.
                coreintv[2,1,j] = results[i+(j-1)*sims,3]
                coreintv[3,1,j] = -results[i+(j-1)*sims,3]

                coreintv[1,2,j] = results[i+(j-1)*sims,4]*2
                coreintv[2,2,j] = -results[i+(j-1)*sims,4]
                coreintv[3,2,j] = -results[i+(j-1)*sims,4]

                coreintv[1,3,j] = 0.
                coreintv[2,3,j] = results[i+(j-1)*sims,5]
                coreintv[3,3,j] = -results[i+(j-1)*sims,5]
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
                coarseprocess[j] = remotecall(run,j, coreintr[:,:,j], coreintv[:,:,j], m, 1e-3,90,1000, coreintr[:,:,j], coreintv[:,:,j])
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
                fineprocess[j] = remotecall(run,j, coarse_r[:,:,j], coarse_v[:,:,j], m, 1e-3,5,1, coreintr[:,:,j], coreintv[:,:,j])
            end
            for j in 1:ncores
                fine_p[j],fine_e[j],fine_r[:,:,j],fine_v[:,:,j] = fetch(fineprocess[j])
                # println("fineintr=",fine_r[:,:,j])
                # println("fineintv=",fine_v[:,:,j])
                # println("fineintr=",fine_p[j])
                # println("fineintv=",fine_e[j])
            end
            for j in 1:ncores
                results[i+(j-1)*sims,6] = fine_e[j]
                results[i+(j-1)*sims,7] = fine_p[j]
                
            end
            println("progress = ",i,"/",sims)
        end

        row = argmin(results[:,6])
        println("argmin =",row)
        println("results =",results[row,1:3])
        println("minimum error =",minimum(results[:,6]))
        println("period =",results[row,7])

        if results[row,1] == 0
            rx_step += 1
        else
            rx = copy(results[row,1])
        end
        
        if results[row,2] == 0
            ry_step += 1
        else
            ry = copy(results[row,2])
        end

        if results[row,3] == 0
            vx_step += 1
        else
            vx = copy(results[row,3])
        end

        if results[row,4] == 0
            vy_step += 1
        else
            vy = copy(results[row,4])
        end

        if results[row,5] == 0
            vz_step += 1
        else
            vz = copy(results[row,5])
        end

        sleep(2)
        
        df = convert(DataFrame,results)
        name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\GridSearch5d_B_",iteration,"_.csv")
        rename!(df,[:"Rx",:"Ry",:"Vx",:"Vy",:"Vz" ,:"periodicity error",:"period"])
        CSV.write(name,df)
    end

    println("DONE")
end


@everywhere ra = 0.540329
@everywhere rb = 0.344882
@everywhere intr = [ra*2 0. 0.;
-ra rb 0.;
-ra -rb 0.]

@everywhere va = 1.097194
@everywhere vb = 0.23333
@everywhere vz = 0.0989
@everywhere intv =[0. vb*2 0.;
va -vb vz ;
 -va -vb -vz]

rx_step = 1
ry_step = 1
vx_step = 1
vy_step = 1
vz_step = 1


phase5_together(ra,rb,va,vb,vz,m,rx_step,ry_step,vx_step,vy_step,vz_step,1)

# coarse_p,coarse_e,coarse_r,coarse_v = run(r,v, m, 1e-3, 30, 1000, r, v)
# fine_p,fine_e,fine_r,fine_v = run(coarse_r,coarse_v, m, 1e-3, 80, 1, r, v)
# println("finep=",fine_p)
# println("finee=",fine_e)

