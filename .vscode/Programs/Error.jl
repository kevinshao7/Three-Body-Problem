function initialize(r,v,m) #calculate initial energy and momentum
    m0 = [0,0]  #initialize linear momentum
    e0 = 0  #energy
    a0 = [0,0] #angular momentum
    for x in 1:3
        e0 += 0.5*m[x]*v[x,:]'*v[x,:] #calculate kinetic energy
        m0 += m[x]*v[x,:] #linear momentum
        a0 += cross(r[x,:],(m[x]*v[x,:])) #angular momentum
        for y in x+1:3 #we obtain all xy pairs (1,2), (1,3), and (2,3)
            xy = r[x,:]-r[y,:] #calculate relative distance
            e0 -= m[x]*m[y]/sqrt(xy'*xy) #potential energy
        end
    end
    return e0, m0, a0
end

e0, m0, a0 = initialize(r,v,m)

function error(intr,intv,r,v,m,e0,m0,a0)
    ke=0
    pe=0
    momentum = 0
    perror = [0. 0. 0.]
    for x in 1:3
        ke += 0.5*m[x]*v[x,:]'*v[x,:]
        momentum += m[x]*sqrt(v[x,:]'*v[x,:])
        perror[x] = sqrt((intr[x,:]-r[x,:])'*(intr[x,:]-r[x,:]))
        for y in x+1:3
            xy = r[x,:]-r[y,:]
            pe -= m[x]*m[y]/sqrt(xy'*xy)
        end
    end
    return [(ke+pe)/e0 sqrt((momentum-m0)'*(momentum-m0)) maximum(perror)]
end
