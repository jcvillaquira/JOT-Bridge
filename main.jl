using Revise
using LinearAlgebra
using ProfileView
using Plots

includet("src/jot/stage1.jl")
Revise.track(Stage1, "src/jot/utils.jl")
using .Stage1

@time begin
  N = 1000
  f = rand(N) + 5 .* (1:N .> N / 2)
  params = Dict("γ1" => 0.1, "γ2" => 1.0, "γ3" => 1.0, "β" => 1.0, "a" => 0.5, "κ" => 0.0)
  dh = DataHolder(f, params)
  sl = ADMMSolver(length(f), dh, 20)
  solve_stage1!(sl)
end

visualize(sl)

"v = jumps"
"w = trend"
"n = noise"

function find_zero_crossings(sl; tol = 1e-4)
  z = transpose(sl.data_holder.D) * sl.data_holder.D * sl.v
  z .*= (abs.(z) .> tol)
end

pl = plot(sl.v)
z = transpose(dh.D) * dh.D * sl.v
z .*= (abs.(z) .> 10e-1)
plot!(pl, z)

