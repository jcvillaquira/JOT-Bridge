using LinearAlgebra

function initialize_linear_system!(f; D, H, extra, params)
  N = length(f)
  # First row
  L11 = I(N) + params["β"] * transpose(D) * D
  L12 = I(N)
  L13 = extra["dt"]
  # Second row
  L21 = I(N)
  L22 = I(N) + params["γ2"] * transpose(H) * H
  L23 = transpose(D)
  # Third row
  L31 = D
  L32 = D
  L33 = extra["ddt"]
  # Combine
  L = Dict("A" => [L11 L12; L21 L22],
           "B" => [L13; L23],
           "C" => [L31 L32],
           "D" => L33)
  y1 = [f; f]
  y2 = D * f
  y = Dict("y1" => y1, "y2" => y2)
  # if params["κ"] != 0
  #   L .+= I(size(L, 1))
  # end
  F = factorize(L["A"])
  L["CA1B"] = L["C"] * ( F \ L["B"] )
  return L, F, y
end


function update_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  solver.data_holder.L["D"] = dh.extra["ddt"] + 2 * dh.params["γ3"] * norm(solver.g)^2 * I(N - 1)
  solver.data_holder.y["1"] = dh.f + dh.params["β"] * dh.extra["dt"] * (solver.t - solver.ρ / dh.params["β"])
end


function solve_linear_system(solver)
  dh = solver.data_holder
  y_temp = dh.y["y2"] - dh.L["C"] * ( dh.F \ dh.y["y1"] )
  x2 = ( dh.L["D"] - dh.L["CA1B"] ) \ y_temp
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
  D = zeros(N - 1, N)
  D[:, 1:end-1] = -I(N - 1)
  D[:, 2:end] += I(N - 1)
  H = collect(Tridiagonal(fill(1.0, N - 1), [-1.0, fill(-2.0, N - 2)..., -1.0], fill(1.0, N - 1)))
  return D, H
end
