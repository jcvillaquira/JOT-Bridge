module Stage1

using LinearAlgebra
using ProgressBars
using Plots
using SparseArrays
using Statistics
using LinearMaps

include("utils.jl")

export DataHolder, ADMMSolver, solve_stage1!, visualize, DmCA1B

function φ(t::T, a) where T
  a_bar = sqrt(2 / a)
  if t < a_bar
    return -(a / 2) * t^2 + sqrt(2 * a) * t
  end
  return one(T)
end

mutable struct DataHolder{T}
  f::Vector{T}
  N::Int
  params::Dict{String,Float64}
  D::SparseMatrixCSC{T}
  H::SparseMatrixCSC{T}
  L::Dict{String, SparseMatrixCSC{T}}
  F::Factorization
  y1::Vector{T}
  y2::Vector{T}
  pert::T
  extra::Dict{String,SparseMatrixCSC{T}}
end


function DataHolder(f::Vector{T}, params::Dict{String,Float64}) where T
  update_params!(params)
  N = length(f)
  D, H = create_dh_input(N)
  extra = Dict("ddt" => D * transpose(D), "dt" => transpose(D))
  L, F, y1, y2, pert = initialize_linear_system!(f; D=D, H=H, extra=extra, params=params)
  return DataHolder{T}(f, N, params, D, H, L, F, y1, y2, pert, extra)
end


mutable struct SchurComplement{T}
  data_holder::DataHolder{T}
  v0::Vector{T}
  v1_1::Vector{T}
  v1_2::Vector{T}
  v2_1::Vector{T}
  v2_2::Vector{T}
end


function SchurComplement(data_holder::DataHolder{T}) where T
  v0 = similar(data_holder.f, data_holder.N)
  v1_1 = similar(v0, 2 * data_holder.N)
  v1_2 = similar(v1_1)
  v2_1 = similar(v0, data_holder.N - 1)
  v2_2 = similar(v2_1)
  return SchurComplement{T}(data_holder, v0, v1_1, v1_2, v2_1, v2_2)
end


function (op::SchurComplement{T})(y::Vector{T}, x::Vector{T}) where T
  # D * x
  mul!(op.v2_1, op.data_holder.L["D"], x)
  # C * inv(A) * B * x
  mul!(op.v1_1, op.data_holder.L["B"], x)
  ldiv!(op.v1_2, op.data_holder.F, op.v1_1)
  mul!(y, op.data_holder.L["C"], op.v1_2)
  y .*= -one(T)
  axpy!(op.data_holder.pert, x, y)
  # Substraction
  y .+= op.v2_1
  return y
end


mutable struct ADMMSolver{T}
  N::Int
  max_iterations::Int
  iterations::Int
  data_holder::DataHolder{T}
  schur_op::SchurComplement{T}
  v::Vector{T}
  w::Vector{T}
  g::Vector{T}
  t::Vector{T}
  ρ::Vector{T}
  n::Vector{T}
end

function ADMMSolver(N::Int, data_holder::DataHolder{T}, max_iterations::Int) where T
  v = similar(data_holder.f)
  w = similar(v)
  n = similar(v)
  g = similar(data_holder.f, N - 1)
  t = similar(g)
  ρ = zeros(T, N - 1)
  return ADMMSolver{T}(N, max_iterations, 0, data_holder, SchurComplement(data_holder), v, w, g, t, ρ, n)
end

function perform_iteration!(solver::ADMMSolver)
  N = solver.N
  # x subproblem
  schur_solve_linear_system!(solver)
  # t subproblem
  q = solver.data_holder.D * solver.v + (solver.ρ / solver.data_holder.params["β"])
  solver.t .= solver.data_holder.params["ν"] .- solver.data_holder.params["ζ"] ./ abs.(q)
  solver.t .= min.(max.(solver.t, 0.0), 1.0)
  solver.t .*= q
  # ρ update
  solver.ρ .-= solver.data_holder.params["β"] * (solver.t - solver.data_holder.D * solver.v)
  update_linear_system!(solver)
  solver.iterations += 1
end


function solve_stage1!(solver::ADMMSolver)
  for _ in (solver.iterations+1):(solver.max_iterations)
    perform_iteration!(solver)
  end
  solver.n = solver.data_holder.extra["dt"] * solver.g
  nothing
end

function visualize(sl::ADMMSolver; shift = false)
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
