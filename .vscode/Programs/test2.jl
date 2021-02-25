intr = [1.08105966433283395241374390321269010e+00 -1.61103999936333666101824156054682023e-06 0.;
-5.40556847423408105134957741609652478e-01 3.45281693188283016303154284469911822e-01 0.;
-5.40508088505425823287375981275225727e-01 -3.45274810552283676957903446556133749e-01 0.]
intv = [2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.; 
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 0.1;
-1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 -0.1]
r = zeros(Float128,(3,3)) #initialize positions and vectors as Float128
v = zeros(Float128,(3,3))
for i in 1:3,j in 1:3 #read data into Float128 arrays (Julia is finnicky in this way)
    r[i,j]=intr[i,j]
    v[i,j]=intv[i,j]
end


function initialize(r,v,m) #calculate initial energy and momentum
    e0 = 0  #initialize energy
    a0 = zeros(Float128,(3,1)) #angular momentum
    m0 = m[1]*v[1,:]+m[2]*v[2,:]+m[3]*v[3,:]#initial momentum
    for x in 1:3
        e0 += 0.5*m[x]*v[x,:]'*v[x,:] #calculate kinetic energy
        a0 += cross(r[x,:],(m[x]*v[x,:])) #angular momentum
        for y in x+1:3 #we obtain all xy pairs (1,2), (1,3), and (2,3)
            xy = r[x,:]-r[y,:] #calculate relative distance
            e0 -= m[x]*m[y]/sqrt(xy'*xy) #potential energy
        end
    end
    return e0,m0,a0
end

initialize(r,v,m)


lambda = (r[i,:]'*a0) / (a0'*a0)
c = lambda*a0 #point of perpendicularity to angular momentum axis
perp_position[i,:] = p-c
r2 = (perp_position[i,:])'*(perp_position[i,:])
moment_of_inertia += m[i]*r2

angular_velocity = (a0/moment_of_inertia)
rotation_rate = sqrt(angular_velocity'*angular_velocity)
tangential_velocity = cross(angular_velocity,perp_position)
