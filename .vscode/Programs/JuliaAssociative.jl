intv = [2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.; 
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 0.;
-1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 0.]
v == intv
x == intv
println(intv)
for i in 1:3,j in 1:3 #convert positions and velocities into relative perspective of body 3
    v[i,j] = v[i,j]-v[3,j]
end 
println(intv)