using Plots
using CSV
using DataFrames

df = CSV.read("3DPeriodicCandidate.csv",DataFrame)
results = convert(Matrix,df)

s = 928
e = 932
title = plot(title=string("6 Order Hermite, dt =",dt),ticks=false, labels=false,grid = false, showaxis = false, bottom_margin = -100Plots.px)
system = plot(results[s:e,2:4],results[s:e,5:7],results[s:e,8:10],title="System",linewidth = 3)
velocities = plot(results[s:e,11:13],results[s:e,14:16],results[s:e,17:19],title="Velocities",linewidth = 3)
energy = plot(results[:,1],results[:,20],title="Energy Error (1e18)",linewidth = 3)
angular_m = plot(results[:,1],results[:,21],title="Angular Momentum Error (1e18)",linewidth = 3)
period = plot(results[s:e,1],results[s:e,22],title="Max Periodicity Error",linewidth = 3)
plot(title,system,velocities,energy,angular_m,period,layout=(6,1),size=(500,1000))
savefig("6Order3D.png")