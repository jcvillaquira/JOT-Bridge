using LinearAlgebra

function initialize_linear_system!(f, L, y; D, H, extra, params)
  N = length(f)
  # First row
  L[1:N, 1:N] = I(N) + params["β"] * transpose(D) * D
  L[1:N, N+1:2N] = I(N)
  L[1:N, 2N+1:end] = extra["dt"]
  # Second row
  L[N+1:2N, 1:N] = I(N)
  L[N+1:2N, N+1:2N] = I(N) + params["γ2"] * transpose(H) * H
  L[N+1:2N, 2N+1:end] = transpose(D)
  # Third row
  L[2N+1:end, 1:N] = D
  L[2N+1:end, N+1:2N] = D
  L[2N+1:end, 2N+1:end] = extra["ddt"]
  ## Right hand side
  y[1:N] = f
  y[N+1:2N] = f
  y[2N+1:end] = D * f
  if params["κ"] != 0
    L .+= I(size(L, 1))
  end
end


function update_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  solver.data_holder.L[2N+1:end, 2N+1:end] = dh.extra["ddt"] + 2 * dh.params["γ3"] * norm(solver.g)^2 * I(N - 1)
  if dh.params["κ"] != 0
    solver.data_holder.L[2N+1:end, 2N+1:end] .+= dh.params["κ"] * I(N-1)
  end
  solver.data_holder.y[1:N] = dh.f + dh.params["β"] * dh.extra["dt"] * (solver.t - solver.ρ / dh.params["β"])
end
