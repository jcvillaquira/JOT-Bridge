
module Stage1

using LinearAlgebra
using ProgressBars
using Plots
using SparseArrays
using Statistics

include("utils.jl")

export DataHolder, ADMMSolver, solve_stage1!, visualize

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
  D::SparseMatrixCSC
  H::SparseMatrixCSC
  L::Dict{String, SparseMatrixCSC}
  CA1B::Matrix{Float64}
  F::Factorization
  y::Dict{String, Vector{Float64}}
  extra::Dict{String,SparseMatrixCSC}
end


function DataHolder(f::Vector{Float64}, params::Dict{String,Float64})
  update_params!(params)
  N = length(f)
  D, H = create_dh_input(N)
  extra = Dict("ddt" => D * transpose(D), "dt" => transpose(D))
  L, F, CA1B, y = initialize_linear_system!(f; D=D, H=H, extra=extra, params=params)
  return DataHolder(f, N, params, D, H, L, CA1B, F, y, extra)
end


mutable struct ADMMSolver
  N::Int
  max_iterations::Int
  iterations::Int
  data_holder::DataHolder
  v::Vector{Float64}
  w::Vector{Float64}
  g::Vector{Float64}
  t::Vector{Float64}
  ρ::Vector{Float64}
  n::Vector{Float64}
end

function ADMMSolver(N, data_holder, max_iterations)
  v = zeros(Float64, N)
  w = zeros(Float64, N)
  n = zeros(Float64, N)
  g = zeros(Float64, N - 1)
  t = zeros(Float64, N - 1)
  ρ = zeros(Float64, N - 1)
  ADMMSolver(N, max_iterations, 0, data_holder, v, w, g, t, ρ, n)
end

function perform_iteration!(solver)
  N = solver.N
  # x subproblem
  x = solve_linear_system(solver)
  solver.v = x[1:N]
  solver.w = x[N+1:2N]
  solver.g = x[2N+1:end]
  # t subproblem
  q = solver.data_holder.D * solver.v + (solver.ρ / solver.data_holder.params["β"])
  for j in 1:length(solver.t)
    t_max = max(solver.data_holder.params["ν"] - solver.data_holder.params["ζ"] / abs(q[j]), 0.0)
    solver.t[j] = min(1, t_max) * q[j]
  end
  # ρ update
  solver.ρ .-= solver.data_holder.params["β"] * (solver.t - solver.data_holder.D * solver.v)
  update_linear_system!(solver)
  solver.iterations += 1
end


function solve_stage1!(solver)
  for _ in ProgressBar((solver.iterations+1):(solver.max_iterations))
    perform_iteration!(solver)
  end
  solver.n = solver.data_holder.extra["dt"] * solver.g
end

function visualize(sl; shift = false)
  val_shift = 0.0
  if shift
    val_shift = mean(sl.v) - mean(sl.data_holder.f)
  end
  pl = plot(sl.data_holder.f, label="Signal")
  plot!(pl, sl.v .- val_shift, label="Jumps (v)")
  plot!(pl, sl.w .+ val_shift, label="Trend (w)")
  plot!(pl, sl.n, label="Noise (n)")
  return pl
end

end
