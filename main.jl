using Revise
using LinearAlgebra
using ProfileView
using Plots
using CSV
using BenchmarkTools
using Debugger
using Tables
using SparseArrays
using IterativeSolvers

includet("src/jot/stage1.jl")
Revise.track(Stage1, "src/jot/utils.jl")
using .Stage1

f = CSV.File(open("data/example.csv"), header=false).Column1
params = Dict("γ1" => 0.05, "γ2" => 1000.0, "γ3" => 0.05, "β" => 12.5, "a" => 50.0, "κ" => 1e-7)

dh = DataHolder(f, params);
sl = ADMMSolver(length(f), dh, 5_000);
@profview solve_stage1!(sl);

visualize(sl)

