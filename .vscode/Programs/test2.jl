using Plots
using CSV
using DataFrames

df = CSV.read("Phase1_AM.csv",DataFrame)
arr = convert(Matrix,df)