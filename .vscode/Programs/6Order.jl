#Kevin Shao Jan 9, 2020
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations

using Quadmath

#starting conditions
p1 = -0.93240737
p2 = -0.86473146
global m = [1. 1. 1.]
dt = 1e-6
t_end = 1
period = 6.32591398
#quantity[body dimension]

function energy(r,v,m)
    ke=0
    pe=0
    momentum = 0
    perror = [0 0 0]
    for x in 1:3
        ke += 0.5*m[x]*v[x,:]'*v[x,:]
        momentum += m[x]*sqrt(v[x,:]'*v[x,:])
        perror[x] = intr[x,:]'*r[x,:]
        for y in x+1:3
            xy = r[x,:]-r[y,:]
            pe -= m[x]*m[y]/sqrt(xy'*xy)
        end
    end
    return [ke+pe momentum max(perror)]
end


function eval(r, v, dt, t_end, results)
    e0 = energy(r,v,m)[1]
    local a = zeros(Float128,(3,2))
    local jk = zeros(Float128,(3,2))
    local s = zeros(Float128,(3,2))
    local c = zeros(Float128,(3,2))
    for i in 1:3
        for j in i+1:3 
            rij = r[j,:]-r[i,:] 
            vij = v[j,:]-v[i,:]
            r2 = rij'*rij
            r3 = r2*sqrt(r2)
            a[i,:] += m[j] * rij / r3
            a[j,:] -= m[i] * rij / r3
            alpha = (rij'*vij)/r2
            aij = m[j] * rij / r3
            jk[i,:] += m[j] * vij / r3 - 3*alpha*aij
            jk[j,:] -= m[i] * vij / r3 - 3*alpha*aij
        end
    end
    for i in 1:3
        for j in i+1:3 
            rij = r[j,:]-r[i,:] 
            vij = v[j,:]-v[i,:]
            r2 = rij'*rij
            r3 = r2*sqrt(r2)
            taij = a[j,:]-a[i,:]
            tjkij = jk[j,:]-jk[i,:]
            aij = m[j] * rij / r3
            alpha = (rij'*vij)/r2
            jkij= m[j] * vij / r3 - 3*alpha*aij
            beta = (vij'*vij + rij'*taij)/r2 + alpha^2
            sij = m[j] * taij / r3 - 6*alpha*jkij - 3*beta*aij
            s[i,:] += sij
            s[j,:] -= sij
            gamma = (3*vij'*taij + rij'*tjkij)/r2 + alpha*(3*beta-4*alpha^2)
            c[i,:] += m[j] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij
            c[j,:] -= m[i] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij
        end
    end
    
    #main loop
    step = 0
    for t in 0:dt:t_end
        old_r = r
        old_v = v
        old_a = a
        old_jk = jk
        old_s = s
        old_c = c
        #predictor
        pr = r + v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120
        pv = v + a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24
        #calculate acceleration, jerk, snap, and crackle
        #calculate in order
        #try totalling acceleration
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
            end
        end
        for i in 1:3
            for j in i+1:3 
                rij = pr[j,:]-pr[i,:] 
                vij = pv[j,:]-pv[i,:]
                r2 = rij'*rij
                r3 = r2*sqrt(r2)
                taij = a[j,:]-a[i,:]
                tjkij = jk[j,:]-jk[i,:]
                aij = m[j] * rij / r3
                alpha = (rij'*vij)/r2
                jkij= m[j] * vij / r3 - 3*alpha*aij
                beta = (vij'*vij + rij'*taij)/r2 + alpha^2
                sij = m[j] * taij / r3 - 6*alpha*jkij - 3*beta*aij
                s[i,:] += sij
                s[j,:] -= sij
                gamma = (3*vij'*taij + rij'*tjkij)/r2 + alpha*(3*beta-4*alpha^2)
                c[i,:] += m[j] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij
                c[j,:] -= m[i] * tjkij / r3 - 9*alpha*sij - 9*beta*jkij - 3*gamma*aij
            end
        end



        #corrector
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        step +=1
        if step % 100 == 1
            new = hcat(reshape(r,(1,6)),hcat(t,(1e18*energy(r,v,m)/e0)-1e18))
            results = vcat(results,new)
            println("t=",t)
        end
        if t==t_end
            println(v)
            println(r)
            endv = v
            endr = r
        end
        
    end
    return results
end
using Plots

r = zeros(Float128,(3,2))
v = zeros(Float128,(3,2))
intr = [0.970040	-0.24309;
-0.97004	0.24309;
0.00000	0.00000]
intv = [0.46620	0.43237;
0.46620	0.43237;
-0.93241	-0.86473]
for i in 1:3,j in 1:2
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end
results=hcat(reshape(r,(1,6)),zeros(Float128,(1,2)))

results = eval(r,v,dt,t_end,results)
p = plot(results[:,1:3],results[:,4:6])
q = plot(results[:,7],results[:,8],title=string("Energy (1e12) dt=",dt))
plot(p,q,layout=(2,1))
savefig("6Order.png")