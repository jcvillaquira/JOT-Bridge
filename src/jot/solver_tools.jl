using SparseArrays
using LinearAlgebra

mutable struct SchurOperator
  A::Factorization
  B::SparseMatrixCSC
  C::SparseMatrixCSC
  D::SparseMatrixCSC
  y1::Vector{Float64}
  y2::Vector{Float64}
  y1t::Vector{Float64}
  y2t::Vector{Float64}
  auxN1::Vector{Float64}
  auxN1_::Vector{Float64}
  aux2N1::Vector{Float64}
  aux2N2::Vector{Float64}
end

function SchurOperator(dh)
  N = dh.N
  y1 = [dh.f; dh.f]
  y2 = dh.D * dh.f
  y1t = similar(y1)
  y2t = similar(y2)
  auxN1 = Vector{Float64}(undef, N - 1)
  auxN1_ = Vector{Float64}(undef, N - 1)
  aux2N1 = Vector{Float64}(undef, 2N)
  aux2N2 = Vector{Float64}(undef, 2N)
  SchurOperator(dh.F, dh.L["B"], dh.L["C"], dh.L["D"], y1, y2, y1t, y2t, auxN1, auxN1_, aux2N1, aux2N2)
end


function update_schur_operator!(schur_op::SchurOperator, D::SparseMatrixCSC)
  schur_op.D = D
  # ldiv!(schur_op.y1t, schur_op.A, schur_op.y1)
  # mul!(schur_op.y2t, schur_op.C, schur_op.y1t)
  # schur_op.y2t .= schur_op.y2 .- schur_op.y2t
end

function get_schur_map(schur_op)
  function S(x) # x is of size N-1
    mul!(schur_op.aux2N1, schur_op.B, x)
    ldiv!(schur_op.aux2N2, schur_op.A, schur_op.aux2N1)
    mul!(schur_op.auxN1, schur_op.C, schur_op.aux2N2)
    mul!(schur_op.auxN1_, schur_op.D, x)
    schur_op.auxN1 .= schur_op.auxN1_ .- schur_op.auxN1
    return copy(schur_op.auxN1)
  end
  return S
end


function schur_solve_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  update_schur_operator!(solver.schur_op, dh.L["D"])
  y_temp = dh.y["y2"] - dh.L["C"] * ( dh.F \ dh.y["y1"] )
  CA1B = x -> dh.L["C"] * ( dh.F \ ( dh.L["B"] * x ) )
  DD = Symmetric(dh.L["D"])
  S = LinearMap(x -> DD * x - CA1B(x), N - 1)
  # S = LinearMap(get_schur_map(solver.schur_op), N - 1)
  solver.g = minres(S, y_temp; log = false, reltol = 1e-3)
  x1 = dh.F \ ( dh.y["y1"] - dh.L["B"] * solver.g )
  solver.v = x1[1:N]
  solver.w = x1[N+1:2N]
end


function direct_solve_linear_system!(solver)
  dh = solver.data_holder
  N = dh.N
  L = [dh.L["A"] dh.L["B"]; dh.L["C"] dh.L["D"]]
  y = [dh.y["y1"]; dh.y["y2"]]
  x = L \ y
  solver.v = x[1:N]
  solver.w = x[N+1:2N]
  solver.g = x[2N+1:end]
end

