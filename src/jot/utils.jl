using LinearAlgebra
using SparseArrays

function initialize_linear_system!(f; D, H, extra, params)
  N = length(f)
  # First row
  I_N = spdiagm(N, N, 0 => fill(1.0, N))
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
  # Combine
  if params["κ"] > 0
    L11 .+= params["κ"] .* I_N
    L22 .+= params["κ"] .* I_N
  end
  L = Dict("A" => [L11 L12; L21 L22],
           "B" => [L13; L23],
           "C" => [L31 L32],
           "D" => L33)
  y1 = [f; f]
  y2 = D * f
  y = Dict("y1" => y1, "y2" => y2)
  F = factorize(Symmetric(L["A"]))
  return L, F, L["C"] * ( F \ L["B"] ), y
end


function update_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  pert = 2 * dh.params["γ3"] * norm(solver.g)^2 + dh.params["κ"]
  solver.data_holder.L["D"] = dh.extra["ddt"] + pert * I(N - 1)
  solver.data_holder.y["y1"][1:N] = dh.f + dh.extra["dt"] * (dh.params["β"] * solver.t - solver.ρ )
end


function solve_linear_system(solver)
  dh = solver.data_holder
  y_temp = dh.y["y2"] - dh.L["C"] * ( dh.F \ dh.y["y1"] )
  x2 = ( dh.L["D"] - dh.CA1B ) \ y_temp
  x1 = dh.F \ ( dh.y["y1"] - dh.L["B"] * x2 )
  return [x1; x2]
end


function update_params!(params)
  λ = params["β"] / params["γ1"]
  @assert params["a"] < λ "a < β / γ₁ is required"
  params["λ"] = λ
  params["ν"] = params["λ"] / (params["λ"] - params["a"])
  params["ζ"] = sqrt(2 * params["a"]) / (params["λ"] - params["a"])
end

function create_dh_input(N)
  D = spdiagm(N - 1, N, 0 => fill(-1.0, N - 1), 1 => fill(1.0, N - 1))
  H = spdiagm(0 => fill(-2.0, N), -1 => fill(1.0, N-1), 1 => fill(1.0, N-1))
  return D, H
end
