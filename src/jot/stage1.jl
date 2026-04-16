module Stage1

function φ(t, a)
  a_bar = sqrt(2 / a)
  if t < a_bar
    return -(a / 2) * t^2 + sqrt(2 * a) * t
  end
  return 1
end

mutable struct ADMM_Solver
  data::Vector{Float64}
end

end
