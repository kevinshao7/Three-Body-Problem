@everywhere r = [1.08105966433283395241374390321269010e+00 -1.61103999936333666101824156054682023e-06 0.;
-5.40556847423408105134957741609652478e-01 3.45281693188283016303154284469911822e-01 0.;
-5.40508088505425823287375981275225727e-01 -3.45274810552283676957903446556133749e-01 0.]
@everywhere v =[2.75243295633073549888088404898033989e-05 4.67209878061247366553801605406549997e-01 0.;
1.09709414564358525218941225169958387e+00 -2.33529804567645806032430881887516834e-01 0.;
 -1.09713166997314851403413883510571396e+00 -2.33670073493601606031632948953538829e-01 0.]
m = [1 1 1]


am_results = zeros(Float128, (2001, 3)) #initialize results
zrange = LinRange(-0.00001, 0.00001, 2001)
for i in 1:2001
    am_results[i,1] = copy(zrange[i])
    am_results[i,1] += 9.85900000000000109601216990995453671e-02
end
zarray = am_results[:,1]
#search iteration
i=1

core2_intv = copy(v) #initialize core velocities
core3_intv = copy(v)
core4_intv = copy(v)

println(zarray[i])
core2_intv[2,3] += zarray[i]#grid search parameters
core2_intv[3,3] -= zarray[i]
core3_intv[2,3] += zarray[i+667]
core3_intv[3,3] -= zarray[i+667]
core4_intv[2,3] += zarray[i+1334]
core4_intv[3,3] -= zarray[i+1334]

#period ~ 92.8
coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3,30,1000, r, core2_intv)#coarse simulation
coarse3 = remotecall(run,3, r, core3_intv, m,  1e-3,30,1000, r, core3_intv)
coarse4 = remotecall(run,4, r, core4_intv, m,  1e-3,30,1000, r, core4_intv)

coarse2_p, coarse2_e, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
coarse3_p, coarse3_e, coarse3_r, coarse3_v = fetch(coarse3)
coarse4_p, coarse4_e, coarse4_r, coarse4_v = fetch(coarse4)
println("Coarse done, starting Fine")

fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-3,80,1, r, core2_intv)#fine simulation
fine3 = remotecall(run,3, coarse3_r, coarse3_v, m,  1e-3,80,1, r, core3_intv)
fine4 = remotecall(run,4, coarse4_r, coarse4_v, m,  1e-3,80,1, r, core4_intv)

fine2_p, fine2_e, fine2_r, fine2_v = fetch(fine2) #fetch fine
fine3_p, fine3_e, fine3_r, fine3_v = fetch(fine3)
fine4_p, fine4_e, fine4_r, fine4_v = fetch(fine4)

am_results[i, 2] = fine2_e #save periodicity error into results
am_results[i+667, 2] = fine3_e
am_results[i+1334, 2] = fine4_e
am_results[i, 3] = fine2_p #save periodicity error into results
am_results[i+667, 3] = fine3_p
am_results[i+1334, 3] = fine4_p
println("progress = ",i,"/667")


sleep(2)
row = argmin(am_results[:,2])
println(am_results[row,1])

println("argmin =",row)
println("z =",am_results[row,1])
println("minimum error =",minimum(am_results[:,2]))
df = convert(DataFrame,am_results)
name = string("C:\\Users\\shaoq\\Documents\\GitHub\\rebound\\.vscode\\Programs\\Grid Search\\Grid Search Data\\Grid Search 4.0\\Phase0AM_3_23(2).csv")
rename!(df,[:"Vz",:"periodicity error",:"period"])
CSV.write(name,df)

println("DONE")
println("Phase 0 Angular Velocities:",v)