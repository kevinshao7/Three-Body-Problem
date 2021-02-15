using Plots
using CSV
using DataFrames

df = CSV.read("Race.csv",DataFrame)
arr = convert(Matrix,df)





#cleaning data
global new_arr = []
for i in 1:40
    if arr[i,8] < 1000
        global new_arr = vcat(new_arr,arr[i,8])
    end
end

x = vcat(vcat(3*ones((40,1)),4*ones((40,1))),vcat(5*ones((40,1)),6*ones((40,1))))
relative = vcat(vcat(arr[:,1],arr[:,3]),vcat(arr[:,5],arr[:,7]))
DataR = hcat(x,relative)
inertial = vcat(vcat(arr[:,2],arr[:,4]),vcat(arr[:,6],new_arr))
y = vcat(vcat(3*ones((40,1)),4*ones((40,1))),vcat(5*ones((40,1)),6*ones((36,1))))
DataI = hcat(y,inertial)



scatter(DataR[:,1],DataR[:,2],label="Relative",yaxis= :log10)
scatter!(DataI[:,1],DataI[:,2],label="Inertial",yaxis=log10,legend =:bottomright,ylabel="Integration Time (s) (log_10)",xlabel="number of timesteps (log_10)",title="Relative vs Inertial Integration Speed Comparison")

using StatsPlots


savefig("Speed_Comparison.png")

