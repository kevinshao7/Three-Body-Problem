i = 1
body = 1
depth = 1
using Quadmath
using Distributed
using DistributedArrays
using SharedArrays
using CSV
using DataFrames

function search_table() 
    searchtable = [0 0 0]
    for i in -5:5
        for j in -5:5
            for k in -5:5
                searchtable = vcat(searchtable,[i j k])
            end
        end
    end
    return searchtable[2:end,:]
end

v_results = zeros(Float128, (1331, 4)) #initialize results
searchtable = search_table() #1331 cases
v_results[:,1:3] = searchtable


println("DONE Body =",body," Depth =",depth)
println("argmin =",argmin(v_results[:,4]))
println("minimum error =",minimum(v_results[:,4]))
df = convert(DataFrame,v_results)
name = string("Phase3V,B",body,"D",depth,".csv")
rename!(df,[:"x cord",:"y cord",:"z cord",:"periodicity error"])
CSV.write(name,df)
r = argmin(v_results[:,4])
v[body,:] += searchtable[r,:]/10^(depth+1) #refine position by converging on periodic solution