i = 1

core1_intv = v #initialize core velocities
core2_intv = v
core3_intv = v
core4_intv = v

core1_intv[body,:] += searchtable[i,:]/10^(depth+1) #grid search parameters
core2_intv[body,:] += searchtable[i+332,:]/10^(depth+1)
core3_intv[body,:] += searchtable[i+664,:]/10^(depth+1)
core4_intv[body,:] += searchtable[i+996,:]/10^(depth+1)

#period ~ 92.8
coarse1 = remotecall(run,1, r, core1_intv, m, 1e-3, 92.7, 10000, r, core1_intv) #coarse simulation
coarse2 = remotecall(run,2, r, core2_intv, m, 1e-3, 92.7, 10000, r, core2_intv)
coarse3 = remotecall(run,3, r, core3_intv, m, 1e-3, 92.7, 10000, r, core3_intv)
coarse4 = remotecall(run,4, r, core4_intv, m, 1e-3, 92.7, 10000, r, core4_intv)

coarse1_p, coarse1_r, coarse1_v = fetch(coarse1) #fetch coarse
coarse2_p, coarse2_r, coarse2_v = fetch(coarse2)
coarse3_p, coarse3_r, coarse3_v = fetch(coarse3)
coarse4_p, coarse4_r, coarse4_v = fetch(coarse4)

fine1 = remotecall(run,1, coarse1_r, coarse1_v, m, 1e-4, 0.2, 10, r, core1_intv) #fine simulation
fine2 = remotecall(run,2, coarse2_r, coarse2_v, m, 1e-4, 0.2, 10, r, core2_intv)
fine3 = remotecall(run,3, coarse3_r, coarse3_v, m, 1e-4, 0.2, 10, r, core3_intv)
fine4 = remotecall(run,4, coarse4_r, coarse4_v, m, 1e-4, 0.2, 10, r, core4_intv)

fine1_p, fine1_r, fine1_v = fetch(fine1) #fetch fine
fine2_p, fine2_r, fine2_v = fetch(fine2)
fine3_p, fine3_r, fine3_v = fetch(fine3)
fine4_p, fine4_r, fine4_v = fetch(fine4)

# v_results[searchtable[i,:]+[6 6 6]] .= fine1_p #save periodicity error into results
# v_results[searchtable[i+332,:]+[6 6 6]] .= fine2_p
# v_results[searchtable[i+664,:]+[6 6 6]] .= fine3_p
v_results[6 .+ searchtable[i+996,:]] .= fine4_p