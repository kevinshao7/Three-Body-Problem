using Plots
using CSV
using DataFrames

df = CSV.read("Phase1_AM.csv",DataFrame)
arr = convert(Matrix,df)
plot(arr[:,1],ylabel="Error",title="Phase 1 Angular Momentum Search")

savefig("Phase_1_Angular_Momentum_Search.png")