using LinearAlgebra
using SparseArrays
using IterativeSolvers
using LinearMaps

function initialize_linear_system!(f::Vector; D, H, extra, params)
  N = length(f)
  # First row
  I_N = spdiagm(N, N, 0 => fill(one(eltype(f)), N))
  L11 = I_N + params["β"] * transpose(D) * D
  L12 = I_N
  L13 = extra["dt"]
  # Second row
  L21 = I_N
  L22 = I_N + params["γ2"] * transpose(H) * H
  L23 = transpose(D)
  # Third row
  L31 = D
  L32 = D
  L33 = extra["ddt"]
  pert = 2.0 * params["γ3"]
  # Combine
  if params["κ"] > 0
    L11 .+= params["κ"] .* I_N
    L22 .+= params["κ"] .* I_N
    pert += params["κ"]
  end
  L = Dict("A" => [L11 L12; L21 L22],
           "B" => [L13; L23],
           "C" => [L31 L32],
           "D" => L33)
  y1 = [f; f]
  y2 = D * f
  F = factorize(Symmetric(L["A"]))
  return L, F, y1, y2, pert
end


function update_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  dh.pert = 2 * dh.params["γ3"] * norm(solver.g)^2 + dh.params["κ"]
  solver.data_holder.y1[1:N] = dh.f + dh.extra["dt"] * (dh.params["β"] * solver.t - solver.ρ )
end


function direct_solve_linear_system!(solver)
  """
  Solve Lx=y by forming L and doing L\\y (deprecated)
  """
  dh = solver.data_holder
  N = dh.N
  L = [dh.L["A"] dh.L["B"]; dh.L["C"] dh.L["D"]]
  y = [dh.y1; dh.y2]
  x = L \ y
  solver.v = x[1:N]
  solver.w = x[N+1:2N]
  solver.g = x[2N+1:end]
end


function schur_solve_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  y_temp = dh.y2 - dh.L["C"] * ( dh.F \ dh.y1 )
  # CA1B = x -> dh.L["C"] * ( dh.F \ ( dh.L["B"] * x ) )
  # S = LinearMap(x -> ( dh.L["D"] * x ) .+ ( dh.pert .* x .- CA1B(x) ), N - 1)
  S = LinearMap(solver.dmca1b, N - 1; ismutating = true)
  solver.g = minres(S, y_temp; log = false, reltol = 1e-3)
  x1 = dh.F \ ( dh.y1 - dh.L["B"] * solver.g )
  solver.v = x1[1:N]
  solver.w = x1[N+1:2N]
end


function update_params!(params)
  λ = params["β"] / params["γ1"]
  @assert params["a"] < λ "a < β / γ₁ is required"
  params["λ"] = 2λ
  params["ν"] = params["λ"] / (params["λ"] - params["a"])
  params["ζ"] = sqrt(2 * params["a"]) / (params["λ"] - params["a"])
end

function create_dh_input(N)
  D = spdiagm(N - 1, N, 0 => fill(-1.0, N - 1), 1 => fill(1.0, N - 1))
  H = spdiagm(0 => fill(-2.0, N), -1 => fill(1.0, N-1), 1 => fill(1.0, N-1))
  H[1, 1] = -1.0
  H[end, end] = -1.0
  return D, H
end
