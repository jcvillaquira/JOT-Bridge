using Revise
using LinearAlgebra

includet("src/jot/stage1.jl")
Revise.track(Stage1, "src/jot/utils.jl")
using .Stage1

f = rand(5)
params = Dict("γ1" => 1.0, "γ2" => 1.0, "γ3" => 1.0, "β" => 0.1, "a" => 1.0)
dh = DataHolder(f, params)
sl = ADMMSolver(length(f), 10)

solve_stage1!(sl, dh)
