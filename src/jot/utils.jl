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
end


function update_linear_system!(solver, data_holder)
  N = data_holder.N
  data_holder.L[2N+1:end, 2N+1:end] = data_holder.extra["ddt"] + 2 * data_holder.params["γ3"] * norm(solver.g)^2 * I(N - 1)
  data_holder.y[1:N] = data_holder.f + data_holder.params["β"] * data_holder.extra["dt"] * (solver.t - solver.ρ / data_holder.params["β"])
end
