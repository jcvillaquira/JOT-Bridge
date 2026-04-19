using Revise
using LinearAlgebra
using ProfileView
using Plots
using CSV

includet("src/jot/stage1.jl")
Revise.track(Stage1, "src/jot/utils.jl")
using .Stage1

f = CSV.File(open("data/example.csv"), header=false).Column1
params = Dict("γ1" => 0.1, "γ2" => 1.0, "γ3" => 1.0, "β" => 1.0, "a" => 0.5, "κ" => 1e-7)
dh = DataHolder(f, params)

@time begin
  dh = DataHolder(f, params)
  sl = ADMMSolver(length(f), dh, 20)
  solve_stage1!(sl)
end
visualize(sl)
