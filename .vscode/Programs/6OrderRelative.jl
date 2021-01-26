#Kevin Shao Jan 9, 2020
#Credit 6th and 8th Order Hermite Integrator for N-body Simulations

using Quadmath

#starting conditions
p1 = -0.93240737
p2 = -0.86473146
global m = [1. 1. 1.]
dt = 1e-4
t_end = 20
#quantity[body dimension]




function energy(r,v,m,int_momentum)
    ke=0
    pe=0
    sm = m[1]+m[2]+m[3]
    iv = zeros(Float128,(3,2))
    #conversion to inertial
    iv[3,:] = (int_momentum - m[2]*v[2,:] - m[1]*v[1,:])/sm
    iv[2,:] = v[2,:] + iv[3,:]
    iv[1,:] = v[1,:] + iv[3,:]
    for i in 1:3
        ke += 0.5*m[i]*iv[i,:]'*iv[i,:]
    end
    rij = r[1,:]-r[2,:]
    pe -= m[1]*m[3]/sqrt(r[1,:]'*r[1,:])
    pe -= m[2]*m[3]/sqrt(r[2,:]'*r[2,:])
    pe -= m[1]*m[2]/sqrt(rij'*rij)
    return ke+pe
end




function eval(r, v, dt, t_end, results, int_momentum)
    e0 = energy(r,v,m, int_momentum)
    local a = zeros(Float128,(2,2))
    local jk = zeros(Float128,(2,2))
    local s = zeros(Float128,(2,2))
    local c = zeros(Float128,(2,2))
    rij = r[2,:]-r[1,:] 
    vij = v[2,:]-v[1,:]
    #organize into arrays 31, 32, 12
    # to convert from aij to aji, aji = -aij*mi/mj
    r2 = [r[1,:]'*r[1,:], r[2,:]'*r[2,:], rij'*rij]
    r3 = r2.*[sqrt(r2[1]), sqrt(r2[2]), sqrt(r2[3])]
    aij = [m[1]*r[1,:]/r3[1], m[2]*r[2,:]/r3[2], m[2]*rij/r3[3]] #31, 32, 12
    a[1,:] = aij[3] - (m[1]+m[3])*r[1,:]/r3[1] - aij[2]
    a[2,:] = -m[1]*rij/r3[3] - (m[2]+m[3])*r[2,:]/r3[2] - aij[1]
    alpha = [(r[1,:]'*v[1,:])/r2[1], (r[2,:]'*v[2,:])/r2[2], (rij'*vij)/r2[3]]
    jkij = [m[1]*v[1,:]/r3[1]-3*alpha[1]*aij[1], m[2]*v[2,:]/r3[2]-3*alpha[2]*aij[2], m[2]*vij/r3[3]-3*alpha[3]*aij[3]]
    jk[1,:] = jkij[3] #jk12 + jk13 - jk31 - jk32
    jk[1,:] += -m[3]*v[1,:]/r3[1] + 3*alpha[1]*aij[1]*m[3]/m[1] #alpha13 = alpha31
    jk[1,:] -= jkij[1]
    jk[1,:] -= jkij[2]
    jk[2,:] = -jkij[3]*m[1]/m[2] #jk21 + jk23 - jk31 - jk32
    jk[2,:] += -m[3]*v[2,:]/r3[2] + 3*alpha[2]*aij[2]*m[3]/m[2] #alpha23 = alpha32
    jk[2,:] -= jkij[1]
    jk[2,:] -= jkij[2]
    
    taij = a[2,:] - a[1,:]
    tjkij = jk[2,:] - jk[1,:]
    beta = [(v[1]'*v[1]+r[1]'*a[1])/r2[1]+alpha[1]^2,
    (v[2]'*v[2]+r[2]'*a[2])/r2[2]+alpha[2]^2,
    (vij'*vij+rij'*taij)/r2[3]+alpha[3]^2
    ]
    sij = [m[1]*a[1,:]/r3[1] - 6*alpha[1]*jkij[1] - 3*beta[1]*aij[1],
    m[2]*a[2,:]/r3[2] - 6*alpha[2]*jkij[2] - 3*beta[2]*aij[2],
    m[2]*taij/r3[3] - 6*alpha[3]*jkij[3] - 3*beta[3]*aij[3]
    ]
    s[1,:] = sij[3]
    s[1,:] += sij[1]*m[3]/m[1]
    s[1,:] -= sij[1]
    s[1,:] -= sij[2]
    s[2,:] = -sij[3]*m[1]/m[2]
    s[2,:] += -sij[2]*m[3]/m[2]
    s[2,:] -= sij[1]
    s[2,:] -= sij[2]
    gamma = [(3*v[1,:]'*a[1,:] + r[1,:]'*jk[1,:])/r2[1] + alpha[1]*(3*beta[1]-4*alpha[1]^2),
    (3*v[2,:]'*a[2,:] + r[2,:]'*jk[2,:])/r2[2] + alpha[2]*(3*beta[2]-4*alpha[2]^2),
    (3*vij'*taij + rij'*tjkij)/r2[3] + alpha[3]*(3*beta[3]-4*alpha[3]^2)
    ]
    cij = [m[1]*jk[1,:]/r3[1] - 9*alpha[1]*sij[1] - 9*beta[1]*jkij[1] - 3*gamma[1]*aij[1],
    m[2]*jk[2,:]/r3[2] - 9*alpha[2]*sij[2] - 9*beta[2]*jkij[2] - 3*gamma[2]*aij[2],
    m[2]*tjkij/r3[3] - 9*alpha[3]*sij[3] - 9*beta[3]*jkij[3] - 3*gamma[3]*aij[3]
    ]
    c[1,:] = cij[3]
    c[1,:] += cij[1]*m[3]/m[1]
    c[1,:] -= cij[1]
    c[1,:] -= cij[2]
    c[2,:] = -cij[3]*m[1]/m[2]
    c[2,:] += -cij[2]*m[3]/m[2]
    c[2,:] -= cij[1]
    c[2,:] -= cij[2]
    
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
        r += v*dt + a*(dt^2)/2 + jk*(dt^3)/6 + s*(dt^4)/24 + c*(dt^5)/120
        v += + a*dt + jk*(dt^2)/2 + s*(dt^3)/6 + c*(dt^4)/24
        #calculate acceleration, jerk, snap, and crackle
        #calculate in order
        #try totalling acceleration
        a = zeros(Float128,(2,2))
        jk = zeros(Float128,(2,2))
        s = zeros(Float128,(2,2))
        c = zeros(Float128,(2,2))
        rij = r[2,:]-r[1,:] 
        vij = v[2,:]-v[1,:]
        #organize into arrays 31, 32, 12
        # to convert from aij to aji, aji = -aij*mi/mj
        r2 = [r[1,:]'*r[1,:], r[2,:]'*r[2,:], rij'*rij]
        r3 = r2.*[sqrt(r2[1]), sqrt(r2[2]), sqrt(r2[3])]
        aij = [m[1]*r[1,:]/r3[1], m[2]*r[2,:]/r3[2], m[2]*rij/r3[3]] #31, 32, 12
        a[1,:] = aij[3] - (m[1]+m[3])*r[1,:]/r3[1] - aij[2]
        a[2,:] = -m[1]*rij/r3[3] - (m[2]+m[3])*r[2,:]/r3[2] - aij[1]
        alpha = [(r[1,:]'*v[1,:])/r2[1], (r[2,:]'*v[2,:])/r2[2], (rij'*vij)/r2[3]]
        jkij = [m[1]*v[1,:]/r3[1]-3*alpha[1]*aij[1], m[2]*v[2,:]/r3[2]-3*alpha[2]*aij[2], m[2]*vij/r3[3]-3*alpha[3]*aij[3]]
        jk[1,:] = jkij[3] #jk12 + jk13 - jk31 - jk32
        jk[1,:] += -m[3]*v[1,:]/r3[1] + 3*alpha[1]*aij[1]*m[3]/m[1] #alpha13 = alpha31
        jk[1,:] -= jkij[1]
        jk[1,:] -= jkij[2]
        jk[2,:] = -jkij[3]*m[1]/m[2] #jk21 + jk23 - jk31 - jk32
        jk[2,:] += -m[3]*v[2,:]/r3[2] + 3*alpha[2]*aij[2]*m[3]/m[2] #alpha23 = alpha32
        jk[2,:] -= jkij[1]
        jk[2,:] -= jkij[2]
        
        taij = a[2,:] - a[1,:]
        tjkij = jk[2,:] - jk[1,:]
        beta = [(v[1]'*v[1]+r[1]'*a[1])/r2[1]+alpha[1]^2,
        (v[2]'*v[2]+r[2]'*a[2])/r2[2]+alpha[2]^2,
        (vij'*vij+rij'*taij)/r2[3]+alpha[3]^2
        ]
        sij = [m[1]*a[1,:]/r3[1] - 6*alpha[1]*jkij[1] - 3*beta[1]*aij[1],
        m[2]*a[2,:]/r3[2] - 6*alpha[2]*jkij[2] - 3*beta[2]*aij[2],
        m[2]*taij/r3[3] - 6*alpha[3]*jkij[3] - 3*beta[3]*aij[3]
        ]
        s[1,:] = sij[3]
        s[1,:] += sij[1]*m[3]/m[1]
        s[1,:] -= sij[1]
        s[1,:] -= sij[2]
        s[2,:] = -sij[3]*m[1]/m[2]
        s[2,:] += -sij[2]*m[3]/m[2]
        s[2,:] -= sij[1]
        s[2,:] -= sij[2]
        gamma = [(3*v[1,:]'*a[1,:] + r[1,:]'*jk[1,:])/r2[1] + alpha[1]*(3*beta[1]-4*alpha[1]^2),
        (3*v[2,:]'*a[2,:] + r[2,:]'*jk[2,:])/r2[2] + alpha[2]*(3*beta[2]-4*alpha[2]^2),
        (3*vij'*taij + rij'*tjkij)/r2[3] + alpha[3]*(3*beta[3]-4*alpha[3]^2)
        ]
        cij = [m[1]*jk[1,:]/r3[1] - 9*alpha[1]*sij[1] - 9*beta[1]*jkij[1] - 3*gamma[1]*aij[1],
        m[2]*jk[2,:]/r3[2] - 9*alpha[2]*sij[2] - 9*beta[2]*jkij[2] - 3*gamma[2]*aij[2],
        m[2]*tjkij/r3[3] - 9*alpha[3]*sij[3] - 9*beta[3]*jkij[3] - 3*gamma[3]*aij[3]
        ]
        c[1,:] = cij[3]
        c[1,:] += cij[1]*m[3]/m[1]
        c[1,:] -= cij[1]
        c[1,:] -= cij[2]
        c[2,:] = -cij[3]*m[1]/m[2]
        c[2,:] += -cij[2]*m[3]/m[2]
        c[2,:] -= cij[1]
        c[2,:] -= cij[2]


        #corrector
        v = old_v + (old_a + a)*dt/2 + ((old_jk - jk)*dt^2)/10 + ((old_s + s)*dt^3)/120
        r = old_r + (old_v + v)*dt/2 + ((old_a - a)*dt^2)/10 + ((old_jk + jk)*dt^3)/120
        
        step +=1
        if step % 100 == 1
            new = hcat(reshape(r,(1,4)),reshape(v,(1,4)))
            results = vcat(results,new)
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
0 0 ]
intv = [0.46620	0.43237;
0.46620	0.43237;
-0.93241 -0.86473]
for i in 1:3,j in 1:2
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end
int_momentum = m[1]*v[1,:]+m[2]*v[2,:]+m[3]*v[3,:]
for i in 1:2,j in 1:2
    r[i,j]-=r[3,j]
    v[i,j]-=v[3,j]
end
r = r[1:2,:]
v = v[1:2,:]

results=hcat(reshape(r,(1,4)),reshape(v,(1,4)))

results = eval(r,v,dt,t_end,results,int_momentum)
p = plot(results[:,1:2],results[:,3:4])
q = plot(results[:,5],results[:,6],title=string("Energy (1e12) dt=",dt))
plot(p,q,layout=(2,1))
savefig("6OrderRelative.png")