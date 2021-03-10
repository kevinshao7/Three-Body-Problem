using Quadmath

#starting conditions
p1 = -0.93240737
p2 = -0.86473146
global m = [1. 1. 1.]
dt = 1e-7
t_end = 1e-2
#quantity[body dimension]

function energy(r,v,m)
    ke=0
    pe=0
    for x in 1:3
        ke += 0.5*m[x]*v[x,:]'*v[x,:]
        for y in x+1:3
            xy = r[x,:]-r[y,:]
            pe -= m[x]m[y]/sqrt(xy'*xy)
        end
    end
    return ke+pe
end


function eval(r, v, dt, t_end, results)
    local a = zeros(Float128,(3,2))
    local jk = zeros(Float128,(3,2))
    for i in 1:3
        for j in i+1:3 
            rji = r[j,:]-r[i,:] 
            vji = v[j,:]-v[i,:] 
            r2 = rji'*rji
            r3 = r2*sqrt(r2)
            rv = rji'*vji
            rv/= r2
            a[i,:] += m[j] * rji / r3
            a[j,:] -= m[i] * rji / r3
            jk[i,:] += m[j] * (vji-3 * rv * rji) / r3
            jk[j,:] -= m[i] * (vji-3 * rv * rji) / r3
            
        end
    end
    
    #main loop
    
    for t in 0:dt:t_end
        old_r = r
        old_v = v
        old_a = a
        old_jk = jk
        #predictor
        r += v*dt + a*dt*dt/2 + jk*dt*dt*dt/6
        v += a*dt + jk*dt*dt/2
        #calculate acceleration and jerk
        a = zeros(3,2)
        jk = zeros(3,2)
        for i in 1:3
            for j in i+1:3 
                rji = r[j,:]-r[i,:] 
                vji = v[j,:]-v[i,:] 
                r2 = rji'*rji
                r3 = r2*sqrt(r2)
                rv = rji'*vji
                rv/= r2
                a[i,:] += m[j] * rji / r3
                a[j,:] -= m[i] * rji / r3
                jk[i,:] += m[j] * (vji-3 * rv * rji) / r3
                jk[j,:] -= m[i] * (vji-3 * rv * rji) / r3
                
            end
        end
        
        
        #corrector
        v = old_v + (old_a + a)*dt/2 + (old_jk - jk)*dt*dt/12
        r = old_r + (old_v + v)*dt/2 + (old_a - a)*dt*dt/12
        
    
        
        
        if t*100000 % 1 == 0
            new = hcat(reshape(r,(1,6)),hcat(t,1e6*energy(r,v,m)/e0-1e6))
            results = vcat(results,new)
    
        end
        if t* 100 % 1 == 0
            println("t=",t)
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
e0 = energy(r,v,m)
results = eval(r,v,dt,t_end,results)
p = plot(results[:,1:3],results[:,4:6])
q = plot(results[:,7],results[:,8],title=string("Energy difference (*1e6) dt=",dt))
plot(p,q,layout=(2,1))
savefig("4Order2.png")

