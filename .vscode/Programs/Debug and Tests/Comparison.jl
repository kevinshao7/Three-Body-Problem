using Quadmath
using Plots
using DataFrames
using CSV

inertial = DataFrame(CSV.File("6OrderInertial.csv"))
relative = DataFrame(CSV.File("6OrderRelative.csv"))

I = convert(Matrix{Float128},inertial)
R = convert(Matrix{Float128},relative)


function compare(I,R)
    rows,columns = size(I)
    results = zeros(Float128,(1,13))
    for i in 1:1:rows
        results = vcat(results,[i sqrt(I[i,1]*R[i,1]) sqrt(I[i,2]*R[i,2]) sqrt(I[i,3]*R[i,3]) sqrt(I[i,4]*R[i,4]) sqrt(I[i,5]*R[i,5]) sqrt(I[i,6]*R[i,6]) sqrt(I[i,7]*R[i,7]) sqrt(I[i,8]*R[i,8]) sqrt(I[i,9]*R[i,9]) sqrt(I[i,10]*R[i,10]) sqrt(I[i,11]*R[i,11]) sqrt(I[i,12]*R[i,12])])
    end
    return results
end

results = compare(I,R)
l = ["r1x" "r2x" "r3x" "r1y" "r2y" "r3y" "v1x" "v2x" "v3x" "v1y" "v2y" "v3y"]
plot(results[10:1000,1],results[10:1000,2:13],labels=l,size=(700,400),yaxis=log)
savefig("Comparison.png")