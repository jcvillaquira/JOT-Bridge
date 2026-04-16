
module Stage1

using LinearAlgebra
include("utils.jl")

export DataHolder, ADMMSolver, solve_stage1!

function φ(t, a)
  a_bar = sqrt(2 / a)
  if t < a_bar
    return -(a / 2) * t^2 + sqrt(2 * a) * t
  end
  return 1
end

mutable struct DataHolder
  f::Vector{Float64}
  N::Int
  params::Dict{String,Float64}
  D::Matrix{Float64}
  H::Matrix{Float64}
  L::Matrix{Float64}
  y::Vector{Float64}
  extra::Dict{String,Matrix{Float64}}
end


function DataHolder(f::Vector{Float64}, params::Dict{String,Float64})
  N = length(f)
  D = zeros(N - 1, N)
  D[:, 1:end-1] = -I(N - 1)
  D[:, 2:end] += I(N - 1)
  H = collect(Tridiagonal(fill(1.0, N - 1), [-1.0, fill(-2.0, N - 2)..., -1.0], fill(1.0, N - 1)))
  L = Array{Float64,2}(undef, N + N + (N - 1), N + N + (N - 1))
  y = Vector{Float64}(undef, N + N + (N - 1))
  extra = Dict("ddt" => D * transpose(D), "dt" => transpose(D))
  params["λ"] = params["β"] / params["γ1"]
  params["ν"] = params["λ"] / (params["λ"] - params["a"])
  params["ζ"] = sqrt(2 * params["a"]) / (params["λ"] - params["a"])
  initialize_linear_system!(f, L, y; D=D, H=H, extra=extra, params=params)
  return DataHolder(f, N, params, D, H, L, y, extra)
end


mutable struct ADMMSolver
  N::Int
  max_iterations::Int
  iterations::Int
  v::Vector{Float64}
  w::Vector{Float64}
  g::Vector{Float64}
  t::Vector{Float64}
  ρ::Vector{Float64}
end

function ADMMSolver(N, max_iterations)
  v = zeros(Float64, N)
  w = zeros(Float64, N)
  g = zeros(Float64, N)
  t = zeros(Float64, N - 1)
  ρ = zeros(Float64, N - 1)
  ADMMSolver(N, max_iterations, 0, v, w, g, t, ρ)
end

function perform_iteration!(solver, data_holder)
  N = solver.N
  # x subproblem
  update_linear_system!(solver, data_holder)
  x = data_holder.L \ data_holder.y
  solver.v = x[1:N]
  solver.w = x[N+1:2N]
  solver.g = x[2N+1:end]
  # t subproblem
  q = data_holder.D * solver.v + (solver.ρ / data_holder.params["β"])
  for j in 1:length(solver.t)
    solver.t[j] = min(1, max(data_holder.params["ν"] - data_holder.params["ζ"] / abs(q[j]))) * q[j]
  end
  # ρ update
  solver.ρ .-= data_holder.params["β"] * (solver.t - data_holder.D * solver.v)
  solver.iterations += 1
end


function solve_stage1!(solver, data_holder)
  for _ in (solver.iterations+1):(solver.max_iterations)
    perform_iteration!(solver, data_holder)
  end
end

end
