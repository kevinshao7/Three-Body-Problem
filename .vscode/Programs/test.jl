i = 1
body = 1
depth = 1

@everywhere v_results = zeros(Float128, (1331, 4)) #initialize results
@everywhere searchtable = search_table() #1331 cases
v_results[:,1:3] = searchtable

core2_intv = v #initialize core velocities
core3_intv = v
core4_intv = v


core2_intv[body,:] += searchtable[i,:]/10^(depth+1)#grid search parameters
core3_intv[body,:] += searchtable[i+443,:]/10^(depth+1)
core4_intv[body,:] += searchtable[i+886,:]/10^(depth+1)

#period ~ 92.8
coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3, 92.7, 10000, r, core2_intv)#coarse simulation
coarse3 = remotecall(run,3, r, core3_intv, m, 1e-3, 92.7, 10000, r, core3_intv)
coarse4 = remotecall(run,4, r, core4_intv, m, 1e-3, 92.7, 10000, r, core4_intv)

coarse2_p, coarse2_r, coarse2_v = fetch(coarse2) #fetch coarse
coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)

fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4, 0.2, 1, r, core2_intv) #fine simulation
fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-4, 0.2, 1, r, core3_intv)
fine4 = remotecall(run,4, coarse4_r, coarse4_v, m, 1e-4, 0.2, 1, r, core4_intv)

fine2_p, fine2_r, fine2_v = fetch(fine2) #fetch fine
fine3_p, fine3_r, fine3_v = fetch(fine3)
fine4_p, fine4_r, fine4_v = fetch(fine4)

v_results[i, 4] = fine2_p #save periodicity error into results
v_results[i+443, 4] = fine3_p
v_results[i+886, 4] = fine4_p
println("progress = ",i,"/443")