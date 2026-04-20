using Revise
using LinearAlgebra
using ProfileView
using Plots
using CSV
using BenchmarkTools

includet("src/jot/stage1.jl")
Revise.track(Stage1, "src/jot/utils.jl")
using .Stage1

f = CSV.File(open("data/example.csv"), header=false).Column1
params = Dict("γ1" => 2.5e-2, "γ2" => 1e3, "γ3" => 1e-5, "β" => 2.7, "a" => 22.2, "κ" => 1e-7)

@btime begin
  dh = DataHolder(f, params);
  sl = ADMMSolver(length(f), dh, 10)
  solve_stage1!(sl)
  nothing
end
