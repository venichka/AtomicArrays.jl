module meanfield_module

export ProductState, densityoperator

import ..integrate_base

import OrdinaryDiffEq, DiffEqCallbacks, SteadyStateDiffEq
import Sundials

using QuantumOpticsBase, LinearAlgebra
using ..interaction_module, ..AtomicArrays

# Define Spin 1/2 operators
const spinbasis = SpinBasis(1//2)
const id = dense(identityoperator(spinbasis))
const sigmax_ = dense(sigmax(spinbasis))
const sigmay_ = dense(sigmay(spinbasis))
const sigmaz_ = dense(sigmaz(spinbasis))
const sigmap_ = dense(sigmap(spinbasis))
const sigmam_ = dense(sigmam(spinbasis))

"""
Class describing a Meanfield state (Product state).
The data layout is [sx1 sx2 ... sy1 sy2 ... sz1 sz2 ...]
# Arguments
* `N`: Number of spins.
* `data`: Vector of length 3*N.
"""
mutable struct ProductState{T1<:Int,T2<:Real}
    N::T1
    data::Vector{T2}
end

"""
    meanfield.ProductState(N)
Meanfield state with all Pauli expectation values equal to zero.
"""
ProductState(N::T) where T<:Int = ProductState(N, zeros(float(T), 3*N))

"""
    meanfield.ProductState(data)
Meanfield state created from real valued vector of length 3*spinnumber.
"""
ProductState(data::Vector{<:Real}) = ProductState(dim(data), data)

"""
    meafield.ProductState(rho)
Meanfield state from density operator.
"""
function ProductState(rho::DenseOpType)
    N = quantum.dim(rho)
    basis = quantum.basis(N)
    state = ProductState(N)
    sx, sy, sz = splitstate(s)
    f(ind, op) = real(expect(embed(basis, ind, op), rho))
    for k=1:N
        sx[k] = f(k, sigmax_)
        sy[k] = f(k, sigmay_)
        sz[k] = f(k, sigmaz_)
    end
    return state
end

"""
    meanfield.blochstate(phi, theta[, N=1])
Product state of `N` single spin Bloch states.
All spins have the same azimuthal angle `phi` and polar angle `theta`.
"""
function blochstate(phi::Vector{T1}, theta::Vector{T2}) where {T1<:Real, T2<:Real}
    N = length(phi)
    @assert length(theta)==N
    state = ProductState(N)
    sx, sy, sz = splitstate(state)
    for k=1:N
        sx[k] = cos(phi[k])*sin(theta[k])
        sy[k] = sin(phi[k])*sin(theta[k])
        sz[k] = cos(theta[k])
    end
    return state
end

function blochstate(phi::Real, theta::Real, N::Int=1)
    state = ProductState(N)
    sx, sy, sz = splitstate(state)
    for k=1:N
        sx[k] = cos(phi)*sin(theta)
        sy[k] = sin(phi)*sin(theta)
        sz[k] = cos(theta)
    end
    return state
end

"""
    meanfield.dim(state)
Number of spins described by this state.
"""
function dim(state::Vector{T}) where T<:Real
    N, rem = divrem(length(state), 3)
    @assert rem==0
    return N
end

"""
    meanfield.splitstate(N, data)
    meanfield.splitstate(state)
Split state into sx, sy and sz parts.
"""
splitstate(N::Int, data::Vector{<:Real}) = view(data, 1:1*N), view(data, 1*N+1:2*N), view(data, 2*N+1:3*N)
splitstate(state::ProductState) = splitstate(state.N, state.data)


"""
    meanfield.densityoperator(sx, sy, sz)
    meanfield.densityoperator(state)
Create density operator from independent sigma expectation values.
"""
function densityoperator(sx::Real, sy::Real, sz::Real)
    return 0.5*(id + sx*sigmax_ + sy*sigmay_ + sz*sigmaz_)
end
function densityoperator(state::ProductState)
    sx, sy, sz = splitstate(state)
    rho = densityoperator(sx[1], sy[1], sz[1])
    for i=2:state.N
        rho = tensor(rho, densityoperator(sx[i], sy[i], sz[i]))
    end
    return rho
end

"""
    meanfield.sx(state)
Sigma x expectation values of state.
"""
sx(x::ProductState) = view(x.data, 1:x.N)

"""
    meanfield.sy(state)
Sigma y expectation values of state.
"""
sy(x::ProductState) = view(x.data, x.N+1:2*x.N)

"""
    meanfield.sz(state)
Sigma z expectation values of state.
"""
sz(x::ProductState) = view(x.data, 2*x.N+1:3*x.N)


"""
    meanfield.timeevolution(T, S::SpinCollection, state0[; fout])
Meanfield time evolution.
# Arguments
* `T`: Points of time for which output will be generated.
* `S`: [`SpinCollection`](@ref) describing the system.
* `state0`: Initial ProductState.
* `fout` (optional): Function with signature `fout(t, state)` that is called whenever output
    should be generated.
"""
function timeevolution(T, S::SpinCollection, state0::ProductState; fout=nothing, kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Ω = interaction_module.OmegaMatrix(S)
    Γ = interaction_module.GammaMatrix(S)

    function f(dy, y, p, t)
        sx, sy, sz = splitstate(N, y)
        dsx, dsy, dsz = splitstate(N, dy)
        @inbounds for k=1:N
            dsx[k] = -S.spins[k].delta*sy[k] - 0.5*S.gammas[k]*sx[k]
            dsy[k] = S.spins[k].delta*sx[k] - 0.5*S.gammas[k]*sy[k]
            dsz[k] = -S.gammas[k]*(1+sz[k])
            for j=1:N
                if j==k
                    continue
                end
                dsx[k] += Ω[k,j]*sy[j]*sz[k] + 0.5*Γ[k,j]*sx[j]*sz[k]
                dsy[k] += -Ω[k,j]*sx[j]*sz[k] + 0.5*Γ[k,j]*sy[j]*sz[k]
                dsz[k] += Ω[k,j]*(sx[j]*sy[k] - sy[j]*sx[k]) - 0.5*Γ[k,j]*(sx[j]*sx[k] + sy[j]*sy[k])
            end
        end
    end

    if isa(fout, Nothing)
        fout_(t, state::ProductState) = deepcopy(state)
    else
        fout_ = fout
    end

    return integrate_base(T, f, state0, fout_; kwargs...)
end

"""
    meanfield.timeevolution_symmetric(T, state0, Ωeff, Γeff[; γ, δ0, fout])
Symmetric meanfield time evolution.
# Arguments
* `T`: Points of time for which output will be generated.
* `state0`: Initial ProductState.
* `Ωeff`: Effective dipole-dipole interaction.
* `Γeff`: Effective collective decay rate.
* `γ=1`: Single spin decay rate.
* `δ0=0`: Phase shift for rotated symmetric meanfield time evolution.
* `fout` (optional): Function with signature `fout(t, state)` that is called whenever output
    should be generated.
"""
function timeevolution_symmetric(T, state0::ProductState, Ωeff::Real, Γeff::Real; γ::Real=1.0, δ0::Real=0., fout=nothing, kwargs...)
    N = 1
    @assert state0.N==N
    function f(dy, y, p, t)
        sx, sy, sz = splitstate(N, y)
        dsx, dsy, dsz = splitstate(N, dy)
        dsx[1] = -δ0*sy[1] + Ωeff*sy[1]*sz[1] - 0.5*γ*sx[1] + 0.5*Γeff*sx[1]*sz[1]
        dsy[1] = δ0*sx[1] - Ωeff*sx[1]*sz[1] - 0.5*γ*sy[1] + 0.5*Γeff*sy[1]*sz[1]
        dsz[1] = -γ*(1+sz[1]) - 0.5*Γeff*(sx[1]^2+sy[1]^2)
    end

    if isa(fout, Nothing)
        fout_ = (t, state) -> deepcopy(state)
    else
        fout_ = fout
    end

    return integrate_base(T, f, state0, fout_; kwargs...)

end


"""
    meanfield.rotate(axis, angles, state)
Rotations on the Bloch sphere for the given [`ProductState`](@ref).
# Arguments
* `axis`: Rotation axis.
* `angles`: Rotation angle(s).
* `state`: [`ProductState`](@ref) that should be rotated.
"""
function rotate(axis::Vector{T1}, angles::Vector{T2}, state::ProductState) where {T1<:Real, T2<:Real}
    @assert length(axis)==3
    @assert length(angles)==state.N
    w = axis/norm(axis)
    sx, sy, sz = splitstate(state)
    state_rot = ProductState(state.N)
    sx_rot, sy_rot, sz_rot = splitstate(state_rot)
    v = zeros(T1, 3)
    for i=1:state.N
        v[1], v[2], v[3] = sx[i], sy[i], sz[i]
        θ = angles[i]
        sx_rot[i], sy_rot[i], sz_rot[i] = cos(θ)*v + sin(θ)*(w × v) + (1-cos(θ))*(w ⋅ v)*w
    end
    return state_rot
end

rotate(axis::Vector{T}, angle::Real, state::ProductState) where {T<:Real} = rotate(axis, ones(T, state.N)*angle, state)


"""
    meanfield_module.f(dy, y, p, t)

Meanfield equations
"""
function f(dy, y, p, t)
    N, gammas, Delta, Ω, Γ, Om_R = p
    sx, sy, sz = splitstate(N, y)
    dsx, dsy, dsz = splitstate(N, dy)
    @inbounds for k=1:N
        dsx[k] = (-Delta[k]*sy[k] - 0.5*gammas[k]*sx[k] -
            2*sz[k]*imag(Om_R[k]))
        dsy[k] = (Delta[k]*sx[k] - 0.5*gammas[k]*sy[k] +
            2*sz[k]*real(Om_R[k]))
        dsz[k] = (-gammas[k]*(1+sz[k]) +
            2*sx[k]*imag(Om_R[k]) -
            2*sy[k]*real(Om_R[k]))
        for j=1:N
            if j==k
                continue
            end
            dsx[k] += Ω[k,j]*sy[j]*sz[k] + 0.5*Γ[k,j]*sx[j]*sz[k]
            dsy[k] += -Ω[k,j]*sx[j]*sz[k] + 0.5*Γ[k,j]*sy[j]*sz[k]
            dsz[k] += (Ω[k,j]*(sx[j]*sy[k] - sy[j]*sx[k]) -
                0.5*Γ[k,j]*(sx[j]*sx[k] + sy[j]*sy[k]))
        end
    end
end


"""
    meanfield_module.f(dy, y, p, t)

Meanfield equations for symbolic computations
"""
function f_sym(dy, y, p, t)
    N, gammas, Delta, Ω, Γ, Om_R = p
    sx = view(y, 1:1*N); sy = view(y, 1*N+1:2*N); sz = view(y, 2*N+1:3*N)
    dsx = view(dy, 1:1*N); dsy = view(dy, 1*N+1:2*N); dsz = view(dy, 2*N+1:3*N)
    @inbounds for k=1:N
        dsx[k] = (-Delta[k]*sy[k] - 0.5*gammas[k]*sx[k] -
            2*sz[k]*imag(Om_R[k]))
        dsy[k] = (Delta[k]*sx[k] - 0.5*gammas[k]*sy[k] +
            2*sz[k]*real(Om_R[k]))
        dsz[k] = (-gammas[k]*(1+sz[k]) +
            2*sx[k]*imag(Om_R[k]) -
            2*sy[k]*real(Om_R[k]))
        for j=1:N
            if j==k
                continue
            end
            dsx[k] += Ω[k,j]*sy[j]*sz[k] + 0.5*Γ[k,j]*sx[j]*sz[k]
            dsy[k] += -Ω[k,j]*sx[j]*sz[k] + 0.5*Γ[k,j]*sy[j]*sz[k]
            dsz[k] += (Ω[k,j]*(sx[j]*sy[k] - sy[j]*sx[k]) -
                0.5*Γ[k,j]*(sx[j]*sx[k] + sy[j]*sy[k]))
        end
    end
end


"""
    meanfield_module.timeevolution_field_meanfield(T, S::SpinCollection, Om_R::Vector, state0[; fout])

Meanfield time evolution.
# Arguments
* `T`: Points of time for which output will be generated.
* `S`: [`SpinCollection`](@ref) describing the system.
* `Om_R`: Interaction constant of atoms with incident field
* `state0`: Initial ProductState.
* `fout` (optional): Function with signature `fout(t, state)` that is called whenever output
    should be generated.
"""
function timeevolution_field(T, S::SpinCollection, Om_R::Vector{ComplexF64},
                                       state0::ProductState;
                                       fout=nothing, kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Delta = [spin.delta for spin in S.spins] 
    gammas = [gamma for gamma in S.gammas] 
    Ω = real(interaction_module.OmegaMatrix(S))
    Γ = real(interaction_module.GammaMatrix(S))
    p = (N, gammas, Delta, Ω, Γ, Om_R)

    if isa(fout, Nothing)
        fout_(t, state::ProductState) = deepcopy(state)
    else
        fout_ = fout
    end

    return integrate(T, f, state0, p, fout_; kwargs...)
end


"""
    meanfield_module.steady_state_field(T, S::SpinCollection, Om_R::Vector ,state0[; fout])
MPC steady state.
# Arguments
* `T`: Points of time for which output will be generated.
* `S`: SpinCollection describing the system.
* `E`: Field inpinging the system
* `state0`: Initial MPCState.
* `fout` (optional): Function with signature fout(t, state) that is called
    whenever output should be generated.
"""
function steady_state_field(T, S::SpinCollection, Om_R::Vector{ComplexF64}, state0::ProductState; fout=nothing, alg::SteadyStateDiffEq.SteadyStateDiffEqAlgorithm=SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.VCABM()), kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Delta = [spin.delta for spin in S.spins] 
    gammas = [gamma for gamma in S.gammas] 
    Ω = real(interaction_module.OmegaMatrix(S))
    Γ = real(interaction_module.GammaMatrix(S))
    p = (N, gammas, Delta, Ω, Γ, Om_R)

    if isa(fout, Nothing)
        fout_ = (t, state) -> deepcopy(state)
    else
        fout_ = fout
    end

    return steady_state(f, state0, p, fout_; alg=alg, kwargs...)
end


"""
    meanfield_module.mapexpect(op, states, num)

Expectation values for a operator of a spin collection.
# Arguments
* `op`: operator
* `states`: solution of the meanfield equation
* `num`: number of atom in the collection
"""
mapexpect(op, states::Array{ProductState{Int64, Float64}, 1}, num::Int) = map(s->(op(s)[num]), states)


"""
    meanfield_module.sigma_matrices(states, t_ind)

Expectation values for sx, sy, sz, sm, sp of a spin collection.
# Arguments
* `states`: solution of the meanfield equation
* `t_ind`: index in the time array
"""
function sigma_matrices(states::Array{ProductState{Int64, Float64}, 1}, t_ind::Int)
    n = length(states[1].data)÷3
    σx = [mapexpect(sx, states, i)[t_ind] for i=1:n]
    σy = [mapexpect(sy, states, i)[t_ind] for i=1:n]
    σz = [mapexpect(sz, states, i)[t_ind] for i=1:n]
    σm = 0.5.*(σx .- 1im.*σy)
    σp = 0.5.*(σx .+ 1im.*σy)
    return [σx, σy, σz, σm, σp]
end


"""
    integrate()
"""
function integrate(T::Vector, f::Function, state0::S, p::Tuple, fout::Function;
                    alg = OrdinaryDiffEq.AutoVern7(OrdinaryDiffEq.RadauIIA5()),
                    callback = nothing, kwargs...) where S

    if isa(state0, Vector{<:Real})
        x0 = state0
        N = length(state0)
        fout_diff = (u, t, integrator) -> fout(t, deepcopy(u))
    else
        x0 = state0.data
        N = state0.N
        fout_diff = (u, t, integrator) -> fout(t, S(N, deepcopy(u)))
    end

    out_type = pure_inference(fout, Tuple{eltype(T),typeof(state0)})
    out = DiffEqCallbacks.SavedValues(eltype(T),out_type)
    scb = DiffEqCallbacks.SavingCallback(fout_diff,out,saveat=T,
                                        save_everystep=false,
                                        save_start = false)

    prob = OrdinaryDiffEq.ODEProblem(f, x0, (T[1], T[end]), p)

    full_cb = OrdinaryDiffEq.CallbackSet(callback, scb)

    sol = OrdinaryDiffEq.solve(prob, alg;
            reltol=1.0e-6,
            abstol=1.0e-8,
            save_everystep = false,
            save_start = false,
            save_end = false,
            callback=full_cb,
            kwargs...)

    out.t, out.saveval

end


Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)


"""
    meanfield_module.steady_state()
"""
function steady_state(f::Function, state0::S, p::Tuple, fout::Function;
                    alg::SteadyStateDiffEq.SteadyStateDiffEqAlgorithm = SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.AutoVern7(OrdinaryDiffEq.RadauIIA5())),
                    callback = nothing, kwargs...) where S

    if isa(state0, Vector{<:Real})
        x0 = state0
        N = length(state0)
        fout_diff = (u, t, integrator) -> fout(t, deepcopy(u))
    else
        x0 = state0.data
        N = state0.N
        fout_diff = (u, t, integrator) -> fout(t, S(N, deepcopy(u)))
    end

    prob = SteadyStateDiffEq.SteadyStateProblem(f, x0, p)

    sol = SteadyStateDiffEq.solve(prob, alg;
            reltol=1.0e-6,
            abstol=1.0e-8,
            tspan=Inf,
            kwargs...)

    return sol

end



end # module