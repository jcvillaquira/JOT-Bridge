using LinearAlgebra

f = rand(5)
γ₁ = 1.0
γ₂ = 1.0
γ₃ = 1.0
γ = [γ₁, γ₂, γ₃]
N = length(f)
max_iter = 100
β = 0.1
κ = 1.0
a = 1

## x-subproblem
D = zeros(N - 1, N)
D[:, 1:end-1] = -I(N - 1)
D[:, 2:end] += I(N - 1)
H = collect(Tridiagonal(fill(1.0, N - 1), [-1.0, fill(-2.0, N - 2)..., -1.0], fill(1.0, N - 1)))

### Coeff Matrix
L = Array{Float64,2}(undef, N + N + (N - 1), N + N + (N - 1))
y = Vector{Float64}(undef, N + N + (N - 1))
function initialize_linear_system!(L, y; N, β, γ₂, f)
  # First row
  L[1:N, 1:N] = I(N) + β * transpose(D) * D
  L[1:N, N+1:2N] = I(N)
  L[1:N, 2N+1:end] = transpose(D)
  # Second row
  L[N+1:2N, 1:N] = I(N)
  L[N+1:2N, N+1:2N] = I(N) + γ₂ * transpose(H) * H
  L[N+1:2N, 2N+1:end] = transpose(D)
  # Third row
  L[2N+1:end, 1:N] = D
  L[2N+1:end, N+1:2N] = D
  ## Right hand side
  y[N+1:2N] = f
  y[2N+1:end] = D * f
end
initialize_linear_system!(L, y; N=N, β=β, γ₂=γ[2], f=f)

function update_linear_system!(L, y; N, β, γ₃, t, ρ, f)
  L[2N+1:end, 2N+1:end] = D * transpose(D) + 2 * γ₃ * norm(g)^2 * I(N - 1)
  y[1:N] = f + β * transpose(D) * (t - ρ / β)
end


v = zeros(Float64, N)
w = zeros(Float64, N)
g = zeros(Float64, N)
t = D * v
ρ = t # TODO: This is according to code (size N-1). According to paper it is N.
# ρ = zeros(Float64, N) 


function update_linear_system!(L, y; N, β, γ₃, t, ρ, f)
  L[2N+1:end, 2N+1:end] = D * transpose(D) + 2 * γ₃ * norm(g)^2 * I(N - 1)
  y[1:N] = f + β * transpose(D) * (t - ρ / β)
end
for n in 1:max_iter
  # x-subproblem
  update_linear_system!(L, y; N=N, β=β, γ₃=γ[3], t=t, ρ=ρ, f=f)
  x = L \ y
  v = x[1:N]
  w = x[N+1:2N]
  g = x[2N+1:end]
  # t-subproblem
  q = D * v + (ρ / β)
  λ = β / γ[1]
  ν = λ / (λ - a)
  ζ = sqrt(2a) / (λ - a)
  for j in 1:length(t)
    t[j] = min(1, max(ν - ζ / abs(q[j]))) * q[j]
  end
  # ρ-subproblem
  ρ .-= β .* (t - D * v)
end


