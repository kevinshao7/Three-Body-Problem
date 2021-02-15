using Quadmath
using LinearAlgebra

r = [0.970040	-0.24309 0.; #data
-0.97004	0.24309 0.;
0. 0. 0.]
v = [0.46620	0.43237 0.;
0.46620	0.43237 0.;
-0.93241 -0.86473 0.]
m = [1. 1. 1.]
sum_mass = m[1]+m[2]+m[3]

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
    return e0, m0, a0
end



function error(r,v,m,m0,sum_mass,e0,a0)
    energy = 0
    angular_m = zeros(Float128,(3,1))
    inertial_v = zeros(Float128,(3,3)) #velocity in inertial frame
    inertial_r = zeros(Float128,(3,3)) #positions in inertial frame
    perror = zeros(Float128, (1,6)) #periodicity error
    #conversion to inertial frame
    inertial_v[3,:] = (m0 - m[2]*v[2,:] - m[1]*v[1,:])/sum_mass #derived from conservation of momentum
    inertial_v[2,:] = v[2,:] + inertial_v[3,:]
    inertial_v[1,:] = v[1,:] + inertial_v[3,:]
    inertial_r[3,:] = -(m[1]*r[1,:]+m[2]*r[2,:])/sum_mass #find centre of mass
    inertial_r[2,:] = r[2,:] + inertial_r[3,:]
    inertial_r[1,:] = r[1,:] + inertial_r[3,:]
    for i in 1:3
        energy += 0.5*m[i]*inertial_v[i,:]'*inertial_v[i,:] 
        angular_m += cross(inertial_r[i,:],(m[i]*inertial_v[i,:]))
    end
    rij = r[1,:]-r[2,:] #distance 1 to 2
    energy -= m[1]*m[3]/sqrt(r[1,:]'*r[1,:]) #potential energy
    energy -= m[2]*m[3]/sqrt(r[2,:]'*r[2,:])
    energy -= m[1]*m[2]/sqrt(rij'*rij)
    return (angular_m-a0)'*(angular_m-a0)
end

e0, m0, a0 = initialize(r,v,m)
println(error(r,v,m,m0,sum_mass,e0,a0))