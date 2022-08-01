module mpc_module

using QuantumOpticsBase, LinearAlgebra
using ..interaction_module, ..AtomicArrays, ..quantum_module

import ..integrate_base

import OrdinaryDiffEq, DiffEqCallbacks, SteadyStateDiffEq
import Sundials
import Symbolics

try
    using Optim
    global optimize = Optim.optimize
catch e
    if typeof(e) == ArgumentError
        println("Optim package not available. (Needed for calculation of squeezing parameter)")
    else
        rethrow(e)
    end
end

import ..meanfield_module: densityoperator

export MPCState, densityoperator


const spinbasis = SpinBasis(1//2)
const sigmax_ = dense(sigmax(spinbasis))
const sigmay_ = dense(sigmay(spinbasis))
const sigmaz_ = dense(sigmaz(spinbasis))
const sigmap_ = dense(sigmap(spinbasis))
const sigmam_ = dense(sigmam(spinbasis))

"""
Class describing a MPC state (Product state + Correlations).
The data layout is vector that in matrix form looks like
Cxx Cxy
Cyy Cxz
Czz Cyz
where the Cij are the appropriate correlation matrices.
The expectation values sx, sy and sz are the diagonals of
the matrices Cxx, Cyy and Czz, respectively.
# Arguments
* `N`: Number of spins.
* `data`: Vector of length (3*N)*(2*N+1).
"""
mutable struct MPCState{T1<:Int,T2<:Real}
    N::T1
    data::Vector{T2}
end

"""
    mpc.MPCState(N)
MPC state with all Pauli expectation values and correlations equal to zero.
"""
MPCState(N::T) where T<:Int = MPCState(N, zeros(float(T), (3*N)*(2*N+1)))

"""
    mpc.MPCState(data)
MPC state created from real valued vector of length (3*N)*(2*N+1).
"""
MPCState(data::Vector{<:Real}) = MPCState(dim(data), data)

"""
    mpc.MPCState(rho)
Create MPC state from density operator.
"""
function MPCState(rho::AbstractOperator)
    basis = rho.basis_l
    N = length(basis.bases)
    state = MPCState(N)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state)
    f(ind, op) = real(expect(embed(basis, ind, op), rho))
    for k=1:N
        sx[k] = f(k, sigmax_)
        sy[k] = f(k, sigmay_)
        sz[k] = f(k, sigmaz_)
        for l=1:N
            if k==l
                continue
            end
            Cxx[k,l] = f([k,l], [sigmax_, sigmax_])
            Cyy[k,l] = f([k,l], [sigmay_, sigmay_])
            Czz[k,l] = f([k,l], [sigmaz_, sigmaz_])
            Cxy[k,l] = f([k,l], [sigmax_, sigmay_])
            Cxz[k,l] = f([k,l], [sigmax_, sigmaz_])
            Cyz[k,l] = f([k,l], [sigmay_, sigmaz_])
        end
    end
    return state
end

"""
    mpc.blochstate(phi, theta[, N=1])
Product state of `N` single spin Bloch states.
All spins have the same azimuthal angle `phi` and polar angle `theta`.
"""
function blochstate(phi::Vector{T1}, theta::Vector{T2}) where {T1<:Real,T2<:Real}
    N = length(phi)
    @assert length(theta)==N
    state = MPCState(N)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state)
    for k=1:N
        sx[k] = cos(phi[k])*sin(theta[k])
        sy[k] = sin(phi[k])*sin(theta[k])
        sz[k] = cos(theta[k])
    end
    for k=1:N, l=1:N
        if k==l
            continue
        end
        Cxx[k,l] = sx[k]*sx[l]
        Cyy[k,l] = sy[k]*sy[l]
        Czz[k,l] = sz[k]*sz[l]
        Cxy[k,l] = sx[k]*sy[l]
        Cxz[k,l] = sx[k]*sz[l]
        Cyz[k,l] = sy[k]*sz[l]
    end
    return state
end

function blochstate(phi::Real, theta::Real, N::Int=1)
    state = MPCState(N)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state)
    for k=1:N
        sx[k] = cos(phi)*sin(theta)
        sy[k] = sin(phi)*sin(theta)
        sz[k] = cos(theta)
    end
    for k=1:N, l=1:N
        if k==l
            continue
        end
        Cxx[k,l] = sx[k]*sx[l]
        Cyy[k,l] = sy[k]*sy[l]
        Czz[k,l] = sz[k]*sz[l]
        Cxy[k,l] = sx[k]*sy[l]
        Cxz[k,l] = sx[k]*sz[l]
        Cyz[k,l] = sy[k]*sz[l]
    end
    return state
end

function integersqrt(N::Int)
    n = sqrt(N)
    if abs(trunc(Int, n)-n)>10*eps(n)
        error("N is not a square of an integer.")
    end
    return trunc(Int, n)
end

"""
    mpc.dim(state)
Number of spins described by this state.
"""
function dim(state::Vector{T}) where T<:Real
    x, rem = divrem(length(state), 3)
    @assert rem==0
    N, rem = divrem(-1 + integersqrt(1+8*x), 4)
    @assert rem==0
    return N
end

"""
    mpc.splitstate(N, data)
    mpc.splitstate(state)
Returns sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz.
"""
function splitstate(N::Int, data::Vector{<:Real})
    data = reshape(data, 3*N, 2*N+1)
    sx = view(data, 1:1*N, 2*N+1)
    sy = view(data, 1*N+1:2*N, 2*N+1)
    sz = view(data, 2*N+1:3*N, 2*N+1)
    Cxx = view(data, 1:1*N, 1:1*N)
    Cyy = view(data, 1*N+1:2*N, 1:1*N)
    Czz = view(data, 2*N+1:3*N, 1:1*N)
    Cxy = view(data, 1:1*N, 1*N+1:2*N)
    Cxz = view(data, 1*N+1:2*N, 1*N+1:2*N)
    Cyz = view(data, 2*N+1:3*N, 1*N+1:2*N)
    return sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz
end
splitstate(state::MPCState) = splitstate(state.N, state.data)

function covarianceoperator(productstate::Vector{<:DenseOpType}, operators::Vector{<:DenseOpType}, indices::Vector{Int})
    x = DenseOpType[(i in indices ? operators[something(findfirst(isequal(i), indices), 0)] : productstate[i]) for i=1:length(productstate)]
    return tensor(x...)
end

"""
    mpc.correlation2covariance(corstate)
Convert a MPCState from correlation form into covariance form.
Basically it just calculates Cov_ab = <s_a s_b> - <s_a> <s_b>.
"""
function correlation2covariance(corstate::MPCState)
    N = corstate.N
    covstate = MPCState(N)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(corstate)
    covsx, covsy, covsz, Covxx, Covyy, Covzz, Covxy, Covxz, Covyz = splitstate(covstate)
    for k=1:N
        covsx[k] = sx[k]
        covsy[k] = sy[k]
        covsz[k] = sz[k]
    end
    for k=1:N, l=1:N
        if k==l
            continue
        end
        Covxx[k,l] = Cxx[k,l] - sx[k]*sx[l]
        Covyy[k,l] = Cyy[k,l] - sy[k]*sy[l]
        Covzz[k,l] = Czz[k,l] - sz[k]*sz[l]
        Covxy[k,l] = Cxy[k,l] - sx[k]*sy[l]
        Covxz[k,l] = Cxz[k,l] - sx[k]*sz[l]
        Covyz[k,l] = Cyz[k,l] - sy[k]*sz[l]
    end
    return covstate
end

"""
    mpc.covariance2correlation(covstate)
Convert a MPCState from covariance form into correlation form.
Basically it just calculates <s_a s_b> = Cov_ab + <s_a> <s_b>.
"""
function covariance2correlation(covstate::MPCState)
    N = covstate.N
    corstate = MPCState(N)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(corstate)
    covsx, covsy, covsz, Covxx, Covyy, Covzz, Covxy, Covxz, Covyz = splitstate(covstate)
    for k=1:N
        sx[k] = covsx[k]
        sy[k] = covsy[k]
        sz[k] = covsz[k]
    end
    for k=1:N, l=1:N
        if k==l
            continue
        end
        Cxx[k,l] = Covxx[k,l] + sx[k]*sx[l]
        Cyy[k,l] = Covyy[k,l] + sy[k]*sy[l]
        Czz[k,l] = Covzz[k,l] + sz[k]*sz[l]
        Cxy[k,l] = Covxy[k,l] + sx[k]*sy[l]
        Cxz[k,l] = Covxz[k,l] + sx[k]*sz[l]
        Cyz[k,l] = Covyz[k,l] + sy[k]*sz[l]
    end
    return corstate
end

"""
    mpc.densityoperator(state)
Create density operator from MPCState.
"""
function densityoperator(state::MPCState)
    N = state.N
    covstate = correlation2covariance(state)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(covstate)
    productstate = DenseOpType[densityoperator(sx[k], sy[k], sz[k]) for k=1:N]
    C(op1,op2,index1,index2) = covarianceoperator(productstate, [op1,op2], [index1,index2])
    ρ = reduce(tensor, productstate)
    for k=1:N, l=k+1:N
        ρ += 0.25*(
              Cxx[k,l]*C(sigmax_,sigmax_,k,l) + Cxy[l,k]*C(sigmay_,sigmax_,k,l) + Cxz[l,k]*C(sigmaz_,sigmax_,k,l)
            + Cxy[k,l]*C(sigmax_,sigmay_,k,l) + Cyy[k,l]*C(sigmay_,sigmay_,k,l) + Cyz[l,k]*C(sigmaz_,sigmay_,k,l)
            + Cxz[k,l]*C(sigmax_,sigmaz_,k,l) + Cyz[k,l]*C(sigmay_,sigmaz_,k,l) + Czz[k,l]*C(sigmaz_,sigmaz_,k,l))
    end
    return ρ
end

"""
    mpc.sx(state)
Sigma x expectation values of state.
"""
sx(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 0*x.N+1:1*x.N, 2*x.N+1)

"""
    mpc.sy(state)
Sigma y expectation values of state.
"""
sy(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 1*x.N+1:2*x.N, 2*x.N+1)

"""
    mpc.sz(state)
Sigma z expectation values of state.
"""
sz(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 2*x.N+1:3*x.N, 2*x.N+1)

"""
    mpc.Cxx(state)
Sigmax-Sigmax correlation values of MPCState.
"""
Cxx(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 0*x.N+1:1*x.N, 0*x.N+1:1*x.N)
"""
    mpc.Cyy(state)
Sigmay-Sigmay correlation values of MPCState.
"""
Cyy(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 1*x.N+1:2*x.N, 0*x.N+1:1*x.N)
"""
    mpc.Czz(state)
Sigmaz-Sigmaz correlation values of MPCState.
"""
Czz(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 2*x.N+1:3*x.N, 0*x.N+1:1*x.N)
"""
    mpc.Cxy(state)
Sigmax-Sigmay correlation values of MPCState.
"""
Cxy(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 0*x.N+1:1*x.N, 1*x.N+1:2*x.N)
"""
    mpc.Cxz(state)
Sigmax-Sigmaz correlation values of MPCState.
"""
Cxz(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 1*x.N+1:2*x.N, 1*x.N+1:2*x.N)
"""
    mpc.Cyz(state)
Sigmay-Sigmaz correlation values of MPCState.
"""
Cyz(x::MPCState) = view(reshape(x.data, 3*x.N, 2*x.N+1), 2*x.N+1:3*x.N, 1*x.N+1:2*x.N)

"""
    mpc.correlation(s1, s2, s3, C12, C13, C23)
3-spin correlation value.
"""
function correlation(s1::T, s2::T, s3::T, C12::T, C13::T, C23::T) where T<:Real
    return -2.0*s1*s2*s3 + s1*C23 + s2*C13 + s3*C12
end

"""
    mpc.timeevolution(T, S::SpinCollection, state0[; fout])
MPC time evolution.
# Arguments
* `T`: Points of time for which output will be generated.
* `S`: SpinCollection describing the system.
* `state0`: Initial MPCState.
* `fout` (optional): Function with signature fout(t, state) that is called
    whenever output should be generated.
"""
function timeevolution(T, S::SpinCollection, state0::MPCState; fout=nothing, kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Ω = interaction_module.OmegaMatrix(S)
    Γ = interaction_module.GammaMatrix(S)

    function f(dy, y, p, t)
        sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(N, y)
        dsx, dsy, dsz, dCxx, dCyy, dCzz, dCxy, dCxz, dCyz = splitstate(N, dy)
        @inbounds for k=1:N
            dsx[k] = -0.5*S.gammas[k]*sx[k]
            dsy[k] = -0.5*S.gammas[k]*sy[k]
            dsz[k] = -S.gammas[k]*(1.0+sz[k])
            for j=1:N
                if j==k
                    continue
                end
                dsx[k] += Ω[k,j]*Cyz[j,k] + 0.5*Γ[k,j]*Cxz[j,k]
                dsy[k] += -Ω[k,j]*Cxz[j,k] + 0.5*Γ[k,j]*Cyz[j,k]
                dsz[k] += Ω[k,j]*(Cxy[j,k]-Cxy[k,j]) - 0.5*Γ[k,j]*(Cxx[j,k] + Cyy[j,k])
            end
        end
        @inbounds for k=1:N, l=1:N
            if k==l
                continue
            end
            dCxx[k,l] = -sqrt(S.gammas[k]*S.gammas[l])*Cxx[k,l] + Γ[k,l]*(Czz[k,l]+0.5*sz[k]+0.5*sz[l])
            dCyy[k,l] = -sqrt(S.gammas[k]*S.gammas[l])*Cyy[k,l] + Γ[k,l]*(Czz[k,l]+0.5*sz[k]+0.5*sz[l])
            dCzz[k,l] = -2*sqrt(S.gammas[k]*S.gammas[l])*Czz[k,l] - sqrt(S.gammas[k]*S.gammas[l])*(sz[k]+sz[l]) + Γ[k,l]*(Cyy[k,l]+Cxx[k,l])
            dCxy[k,l] = Ω[k,l]*(sz[k]-sz[l]) - sqrt(S.gammas[k]*S.gammas[l])*Cxy[k,l]
            dCxz[k,l] = Ω[k,l]*sy[l] - 1.5*sqrt(S.gammas[k]*S.gammas[l])*Cxz[k,l] - S.gammas[k]*sx[k] - Γ[k,l]*(Cxz[l,k]+0.5*sx[l])
            dCyz[k,l] = -Ω[k,l]*sx[l] - 1.5*sqrt(S.gammas[k]*S.gammas[l])*Cyz[k,l] - S.gammas[k]*sy[k] - Γ[k,l]*(Cyz[l,k]+0.5*sy[l])
            for j=1:N
                if j==l || j==k
                    continue
                end
                Cxxx = correlation(sx[k], sx[l], sx[j], Cxx[k,l], Cxx[k,j], Cxx[l,j])
                Cxxy = correlation(sx[k], sx[l], sy[j], Cxx[k,l], Cxy[k,j], Cxy[l,j])
                Cxyx = correlation(sx[k], sy[l], sx[j], Cxy[k,l], Cxx[k,j], Cxy[j,l])
                Cxyy = correlation(sx[k], sy[l], sy[j], Cxy[k,l], Cxy[k,j], Cyy[l,j])
                Cxzx = correlation(sx[k], sz[l], sx[j], Cxz[k,l], Cxx[k,j], Cxz[j,l])
                Cxzy = correlation(sx[k], sz[l], sy[j], Cxz[k,l], Cxy[k,j], Cyz[j,l])
                Cyxx = correlation(sy[k], sx[l], sx[j], Cxy[l,k], Cxy[j,k], Cxx[l,j])
                Cyxy = correlation(sy[k], sx[l], sy[j], Cxy[l,k], Cyy[k,j], Cxy[l,j])
                Cyyx = correlation(sy[k], sy[l], sx[j], Cyy[k,l], Cxy[j,k], Cxy[j,l])
                Cyyy = correlation(sy[k], sy[l], sy[j], Cyy[k,l], Cyy[k,j], Cyy[l,j])
                Cyzx = correlation(sy[k], sz[l], sx[j], Cyz[k,l], Cxy[j,k], Cxz[j,l])
                Cyzy = correlation(sy[k], sz[l], sy[j], Cyz[k,l], Cyy[k,j], Cyz[j,l])
                Czxx = correlation(sz[k], sx[l], sx[j], Cxz[l,k], Cxz[j,k], Cxx[l,j])
                Czxy = correlation(sz[k], sx[l], sy[j], Cxz[l,k], Cyz[j,k], Cxy[l,j])
                Czyx = correlation(sz[k], sy[l], sx[j], Cyz[l,k], Cxz[j,k], Cxy[j,l])
                Czyy = correlation(sz[k], sy[l], sy[j], Cyz[l,k], Cyz[j,k], Cyy[l,j])
                Czzx = correlation(sz[k], sz[l], sx[j], Czz[k,l], Cxz[j,k], Cxz[j,l])
                Czzy = correlation(sz[k], sz[l], sy[j], Czz[k,l], Cyz[j,k], Cyz[j,l])

                dCxx[k,l] += Ω[k,j]*Czxy + Ω[l,j]*Cxzy + 0.5*Γ[k,j]*Czxx + 0.5*Γ[l,j]*Cxzx
                dCyy[k,l] += -Ω[k,j]*Czyx - Ω[l,j]*Cyzx + 0.5*Γ[k,j]*Czyy + 0.5*Γ[l,j]*Cyzy
                dCzz[k,l] += (Ω[k,j]*(Cyzx-Cxzy) + Ω[l,j]*(Czyx-Czxy)
                                - 0.5*Γ[k,j]*(Cxzx+Cyzy) - 0.5*Γ[l,j]*(Czxx+Czyy))
                dCxy[k,l] += Ω[k,j]*Czyy - Ω[l,j]*Cxzx + 0.5*Γ[k,j]*Czyx + 0.5*Γ[l,j]*Cxzy
                dCxz[k,l] += (Ω[k,j]*Czzy + Ω[l,j]*(Cxyx-Cxxy)
                                + 0.5*Γ[k,j]*Czzx - 0.5*Γ[l,j]*(Cxxx+Cxyy))
                dCyz[k,l] += (-Ω[k,j]*Czzx + Ω[l,j]*(Cyyx-Cyxy)
                                + 0.5*Γ[k,j]*Czzy - 0.5*Γ[l,j]*(Cyxx+Cyyy))
            end
        end
    end

    if isa(fout, Nothing)
        fout_ = (t, state) -> deepcopy(state)
    else
        fout_ = fout
    end

    return integrate(T, f, state0, fout_; kwargs...)
end

function axisangle2rotmatrix(axis::Vector{T}, angle::Real) where T<:Real
    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    t = 1-c
    R = zeros(T, 3, 3)
    R[1,1] = t*x^2 + c
    R[1,2] = t*x*y - z*s
    R[1,3] = t*x*z + y*s
    R[2,1] = t*x*y + z*s
    R[2,2] = t*y^2 + c
    R[2,3] = t*y*z - x*s
    R[3,1] = t*x*z - y*s
    R[3,2] = t*y*z + x*s
    R[3,3] = t*z^2 + c
    return R
end

"""
    mpc.rotate(axis, angles, state)
Rotations on the Bloch sphere for the given [`MPCState`](@ref).
# Arguments
* `axis`: Rotation axis.
* `angles`: Rotation angle(s).
* `state`: [`MPCState`](@ref) that should be rotated.
"""
function rotate(axis::Vector{T1}, angles::Vector{T2}, state::MPCState) where {T1<:Real, T2<:Real}
    N = state.N
    @assert length(axis)==3
    @assert length(angles)==N
    w = axis/norm(axis)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state)
    rotstate = deepcopy(state)
    sx_rot, sy_rot, sz_rot, Cxx_rot, Cyy_rot, Czz_rot, Cxy_rot, Cxz_rot, Cyz_rot = splitstate(rotstate)
    S_rot = splitstate(rotstate)
    R = [axisangle2rotmatrix(w, angle) for angle=angles]
    for k=1:N
        sx_rot[k], sy_rot[k], sz_rot[k] = R[k]*[sx[k], sy[k], sz[k]]
    end
    for k=1:N,l=1:N
        if k==l
            continue
        end
        S2 = [Cxx[k,l],Cxy[k,l],Cxz[k,l],
              Cxy[l,k],Cyy[k,l],Cyz[k,l],
              Cxz[l,k],Cyz[l,k],Czz[k,l]]
        S2_rot = kron(R[k], R[l])*S2
        Cxx_rot[k,l] = S2_rot[1]
        Cyy_rot[k,l] = S2_rot[5]
        Czz_rot[k,l] = S2_rot[9]
        Cxy_rot[k,l] = S2_rot[2]
        Cxz_rot[k,l] = S2_rot[3]
        Cyz_rot[k,l] = S2_rot[6]
    end
    return rotstate
end

rotate(axis::Vector{T}, angle::Real, state::MPCState) where {T<:Real} = rotate(axis, ones(float(T), state.N)*angle, state)

"""
    mpc.var_Sx(state0)
Variance of the total Sx operator for the given MPCState.
"""
function var_Sx(state0::MPCState)
    N = state0.N
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state0)
    exp_Sx_2 = 0.
    exp_Sx2 = 0.
    for k=1:N, l=1:N
        exp_Sx_2 += sx[k]*sx[l]
        if k==l
            continue
        end
        exp_Sx2 += Cxx[k,l]
    end
    return 1.0/N + 1.0/N^2*exp_Sx2 - 1.0/N^2*exp_Sx_2
end

"""
    mpc.var_Sy(state)
Variance of the total Sy operator for the given MPCState.
"""
function var_Sy(state0::MPCState)
    N = state0.N
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state0)
    exp_Sy_2 = 0.
    exp_Sy2 = 0.
    for k=1:N, l=1:N
        exp_Sy_2 += sy[k]*sy[l]
        if k==l
            continue
        end
        exp_Sy2 += Cyy[k,l]
    end
    return 1.0/N + 1.0/N^2*exp_Sy2 - 1.0/N^2*exp_Sy_2
end

"""
    mpc.var_Sz(state)
Variance of the total Sz operator for the given MPCState.
"""
function var_Sz(state0::MPCState)
    N = state0.N
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state0)
    exp_Sz_2 = 0.
    exp_Sz2 = 0.
    for k=1:N, l=1:N
        exp_Sz_2 += sz[k]*sz[l]
        if k==l
            continue
        end
        exp_Sz2 += Czz[k,l]
    end
    return 1.0/N + 1.0/N^2*exp_Sz2 - 1.0/N^2*exp_Sz_2
end

"""
    mpc.squeeze_sx(χT, state)
Spin squeezing along sx.
# Arguments
* `χT`: Squeezing strength.
* `state0`: MPCState that should be squeezed.
"""
function squeeze_sx(χT::Real, state0::MPCState)
    T = [0,1.]
    N = state0.N
    χeff = 4.0*χT/N^2
    function f(dy, y, p, t)
        sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(N, y)
        dsx, dsy, dsz, dCxx, dCyy, dCzz, dCxy, dCxz, dCyz = splitstate(N, dy)
        for k=1:N
            dsx[k] = 0.
            dsy[k] = 0.
            dsz[k] = 0.
            for j=1:N
                if j==k
                    continue
                end
                dsy[k] += -χeff*Cxz[j,k]
                dsz[k] += χeff*Cxy[j,k]
            end
        end
        for k=1:N, l=1:N
            if k==l
                continue
            end
            dCxx[k,l] = 0.
            dCyy[k,l] = 0.
            dCzz[k,l] = 0.
            dCxy[k,l] = -χeff*sz[l]
            dCxz[k,l] = χeff*sy[l]
            dCyz[k,l] = 0.
            for j=1:N
                if j==l || j==k
                    continue
                end
                Cxyx = correlation(sx[k], sy[l], sx[j], Cxy[k,l], Cxx[k,j], Cxy[j,l])
                Cxzx = correlation(sx[k], sz[l], sx[j], Cxz[k,l], Cxx[k,j], Cxz[j,l])
                Cyyx = correlation(sy[k], sy[l], sx[j], Cyy[k,l], Cxy[j,k], Cxy[j,l])
                Cyzx = correlation(sy[k], sz[l], sx[j], Cyz[k,l], Cxy[j,k], Cxz[j,l])
                Czyx = correlation(sz[k], sy[l], sx[j], Cyz[l,k], Cxz[j,k], Cxy[j,l])
                Czzx = correlation(sz[k], sz[l], sx[j], Czz[k,l], Cxz[j,k], Cxz[j,l])

                dCyy[k,l] += -χeff*(Czyx+Cyzx)
                dCzz[k,l] += χeff*(Cyzx+Czyx)
                dCxy[k,l] += -χeff*Cxzx
                dCxz[k,l] += χeff*Cxyx
                dCyz[k,l] += χeff*(Czzx-Cyyx)
            end
        end
    end

    fout_(t, state::MPCState) = deepcopy(state)
    time_out, state_out = integrate(T, f, state0, fout_)

    return state_out[end]
end

"""
    mpc.squeeze(axis, χT, state0)
Spin squeezing along an arbitrary axis.
# Arguments
* `axis`: Squeezing axis.
* `χT`: Squeezing strength.
* `state0`: MPCState that should be squeezed.
"""
function squeeze(axis::Vector{T}, χT::Real, state0::MPCState) where T<:Real
    @assert length(axis)==3
    axis = axis/norm(axis)
    # Rotation into sigma_x
    w = float(T)[0., axis[3], -axis[2]]
    if norm(w)<1e-5
        state_squeezed = squeeze_sx(χT, state0)
    else
        α = acos(axis[1])
        state_rot = rotate(w, α, state0)
        state_rot_squeezed = squeeze_sx(χT, state_rot)
        state_squeezed = rotate(w, -α, state_rot_squeezed)
    end
    return state_squeezed
end

"""
    mpc.orthogonal_vectors(n)
Create 3 orthonormal vectors where one is in the given direction `n`.
"""
function orthogonal_vectors(n::Vector{<:Real})
    n = n/norm(n)
    v = (n[1]<n[2] ? [1.,0.,0.] : [0.,1.,0.])
    e1 = v - dot(n,v)*n
    e1 = e1/norm(e1)
    e2 = cross(n, e1)
    e2 = e2/norm(e2)
    return e1, e2
end

"""
    mpc.squeezing_parameter(state)
Calculate squeezing parameter for the given `state`.
"""
function squeezingparameter(state::MPCState)
    N = state.N
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = map(sum, splitstate(state))
    n = 1.0/N*[sx, sy, sz]
    e1, e2 = orthogonal_vectors(n)
    function f(phi)
        nphi = cos(phi)*e1 + sin(phi)*e2
        nx = nphi[1]
        ny = nphi[2]
        nz = nphi[3]
        Sphi2 = 1.0/N^2*(N+nx*nx*Cxx + ny*ny*Cyy + nz*nz*Czz +
                2*nx*ny*Cxy + 2*nx*nz*Cxz + 2*ny*nz*Cyz)
        return Sphi2
    end
    varSmin = Optim.optimize(f, 0., 2*pi).f_minimum
    return sqrt(N*varSmin)/norm(n)
end


"""
    mpc_module.f(dy, y, p, t)

MPC equations
"""
function f(dy, y, p, t)
    N, gammas, Delta, Ω, Γ, reOmR, imOmR = p
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(N, y)
    dsx, dsy, dsz, dCxx, dCyy, dCzz, dCxy, dCxz, dCyz = splitstate(N, dy)
    @inbounds for k=1:N
        dsx[k] = (-Delta[k]*sy[k]-0.5*gammas[k]*sx[k] 
                - 2*sz[k]*imOmR[k])
        dsy[k] = (Delta[k]*sx[k]-0.5*gammas[k]*sy[k] 
                + 2*sz[k]*reOmR[k])
        dsz[k] = (-gammas[k]*(1.0+sz[k]) + 2*sx[k]*imOmR[k] 
                                           - 2*sy[k]*reOmR[k])
        for j=1:N
            if j==k
                continue
            end
            dsx[k] += Ω[k,j]*Cyz[j,k] + 0.5*Γ[k,j]*Cxz[j,k]
            dsy[k] += -Ω[k,j]*Cxz[j,k] + 0.5*Γ[k,j]*Cyz[j,k]
            dsz[k] += Ω[k,j]*(Cxy[j,k]-Cxy[k,j]) - 0.5*Γ[k,j]*(Cxx[j,k] + Cyy[j,k])
        end
    end
    @inbounds for k=1:N, l=1:N
        if k==l
            continue
        end
        dCxx[k,l] = (-Delta[k]*Cxy[l,k]-Delta[l]*Cxy[k,l]
            -sqrt(gammas[k]*gammas[l])*Cxx[k,l] + Γ[k,l]*(Czz[k,l]+0.5*sz[k]+0.5*sz[l]) - 2.0 * (imOmR[k]*Cxz[l,k] + 
                                        imOmR[l]*Cxz[k,l]))
        dCyy[k,l] = (Delta[k]*Cxy[k,l]+Delta[l]*Cxy[l,k]
            -sqrt(gammas[k]*gammas[l])*Cyy[k,l] + Γ[k,l]*(Czz[k,l]+0.5*sz[k]+0.5*sz[l]) + 2.0 * (reOmR[k]*Cyz[l,k] + 
                                        reOmR[l]*Cyz[k,l]))
        dCzz[k,l] = (-2*sqrt(gammas[k]*gammas[l])*Czz[k,l] - sqrt(gammas[k]*gammas[l])*(sz[k]+sz[l]) + Γ[k,l]*(Cyy[k,l]+Cxx[k,l])
                - 2.0 * (reOmR[k]*Cyz[k,l] + reOmR[l]*Cyz[l,k])
                + 2.0 * (imOmR[k]*Cxz[k,l] + imOmR[l]*Cxz[l,k]))
        dCxy[k,l] = (Delta[l]*Cxx[l,k]-Delta[k]*Cyy[k,l]+
            Ω[k,l]*(sz[k]-sz[l]) - sqrt(gammas[k]*gammas[l])*Cxy[k,l] 
                + 2.0*reOmR[l]*Cxz[k,l] - 2.0*imOmR[k]*Cyz[l,k])
        dCxz[k,l] = (-Delta[k]*Cyz[k,l] + 
            Ω[k,l]*sy[l] - 1.5*sqrt(gammas[k]*gammas[l])*Cxz[k,l] - gammas[k]*sx[k] - Γ[k,l]*(Cxz[l,k]+0.5*sx[l]) 
                - 2.0*reOmR[l]*Cxy[k,l] 
                - 2.0*(imOmR[k]*Czz[k,l] - imOmR[l]*Cxx[k,l]))
        dCyz[k,l] = (Delta[k]*Cxz[k,l]
            -Ω[k,l]*sx[l] - 1.5*sqrt(gammas[k]*gammas[l])*Cyz[k,l] - gammas[k]*sy[k] - Γ[k,l]*(Cyz[l,k]+0.5*sy[l])
                + 2.0*(reOmR[k]*Czz[k,l] - reOmR[l]*Cyy[k,l])
                + 2.0*imOmR[l]*Cxy[l,k])
        for j=1:N
            if j==l || j==k
                continue
            end
            Cxxx = correlation(sx[k], sx[l], sx[j], Cxx[k,l], Cxx[k,j], Cxx[l,j])
            Cxxy = correlation(sx[k], sx[l], sy[j], Cxx[k,l], Cxy[k,j], Cxy[l,j])
            Cxyx = correlation(sx[k], sy[l], sx[j], Cxy[k,l], Cxx[k,j], Cxy[j,l])
            Cxyy = correlation(sx[k], sy[l], sy[j], Cxy[k,l], Cxy[k,j], Cyy[l,j])
            Cxzx = correlation(sx[k], sz[l], sx[j], Cxz[k,l], Cxx[k,j], Cxz[j,l])
            Cxzy = correlation(sx[k], sz[l], sy[j], Cxz[k,l], Cxy[k,j], Cyz[j,l])
            Cyxx = correlation(sy[k], sx[l], sx[j], Cxy[l,k], Cxy[j,k], Cxx[l,j])
            Cyxy = correlation(sy[k], sx[l], sy[j], Cxy[l,k], Cyy[k,j], Cxy[l,j])
            Cyyx = correlation(sy[k], sy[l], sx[j], Cyy[k,l], Cxy[j,k], Cxy[j,l])
            Cyyy = correlation(sy[k], sy[l], sy[j], Cyy[k,l], Cyy[k,j], Cyy[l,j])
            Cyzx = correlation(sy[k], sz[l], sx[j], Cyz[k,l], Cxy[j,k], Cxz[j,l])
            Cyzy = correlation(sy[k], sz[l], sy[j], Cyz[k,l], Cyy[k,j], Cyz[j,l])
            Czxx = correlation(sz[k], sx[l], sx[j], Cxz[l,k], Cxz[j,k], Cxx[l,j])
            Czxy = correlation(sz[k], sx[l], sy[j], Cxz[l,k], Cyz[j,k], Cxy[l,j])
            Czyx = correlation(sz[k], sy[l], sx[j], Cyz[l,k], Cxz[j,k], Cxy[j,l])
            Czyy = correlation(sz[k], sy[l], sy[j], Cyz[l,k], Cyz[j,k], Cyy[l,j])
            Czzx = correlation(sz[k], sz[l], sx[j], Czz[k,l], Cxz[j,k], Cxz[j,l])
            Czzy = correlation(sz[k], sz[l], sy[j], Czz[k,l], Cyz[j,k], Cyz[j,l])

            dCxx[k,l] += Ω[k,j]*Czxy + Ω[l,j]*Cxzy + 0.5*Γ[k,j]*Czxx + 0.5*Γ[l,j]*Cxzx
            dCyy[k,l] += -Ω[k,j]*Czyx - Ω[l,j]*Cyzx + 0.5*Γ[k,j]*Czyy + 0.5*Γ[l,j]*Cyzy
            dCzz[k,l] += (Ω[k,j]*(Cyzx-Cxzy) + Ω[l,j]*(Czyx-Czxy)
                            - 0.5*Γ[k,j]*(Cxzx+Cyzy) - 0.5*Γ[l,j]*(Czxx+Czyy))
            dCxy[k,l] += Ω[k,j]*Czyy - Ω[l,j]*Cxzx + 0.5*Γ[k,j]*Czyx + 0.5*Γ[l,j]*Cxzy
            dCxz[k,l] += (Ω[k,j]*Czzy + Ω[l,j]*(Cxyx-Cxxy)
                            + 0.5*Γ[k,j]*Czzx - 0.5*Γ[l,j]*(Cxxx+Cxyy))
            dCyz[k,l] += (-Ω[k,j]*Czzx + Ω[l,j]*(Cyyx-Cyxy)
                            + 0.5*Γ[k,j]*Czzy - 0.5*Γ[l,j]*(Cyxx+Cyyy))
        end
    end
end


"""
    mpc_module.f_sym(dy, y, p, t)

MPC equations for symbolic enhancement
"""
function f_sym(dy, y, p, t)
    N, gammas, Delta, Ω, Γ, reOmR, imOmR = p
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate_sym(N, y)
    dsx, dsy, dsz, dCxx, dCyy, dCzz, dCxy, dCxz, dCyz = splitstate_sym(N, dy)
    @inbounds for k=1:N
        dsx[k] = (-Delta[k]*sy[k]-0.5*gammas[k]*sx[k] 
                - 2*sz[k]*imOmR[k])
        dsy[k] = (Delta[k]*sx[k]-0.5*gammas[k]*sy[k] 
                + 2*sz[k]*reOmR[k])
        dsz[k] = (-gammas[k]*(1.0+sz[k]) + 2*sx[k]*imOmR[k] 
                                           - 2*sy[k]*reOmR[k])
        for j=1:N
            if j==k
                continue
            end
            dsx[k] += Ω[k,j]*Cyz[j,k] + 0.5*Γ[k,j]*Cxz[j,k]
            dsy[k] += -Ω[k,j]*Cxz[j,k] + 0.5*Γ[k,j]*Cyz[j,k]
            dsz[k] += Ω[k,j]*(Cxy[j,k]-Cxy[k,j]) - 0.5*Γ[k,j]*(Cxx[j,k] + Cyy[j,k])
        end
    end
    @inbounds for k=1:N, l=1:N
        if k==l
            continue
        end
        dCxx[k,l] = (-Delta[k]*Cxy[l,k]-Delta[l]*Cxy[k,l]
            -sqrt(gammas[k]*gammas[l])*Cxx[k,l] + Γ[k,l]*(Czz[k,l]+0.5*sz[k]+0.5*sz[l]) - 2.0 * (imOmR[k]*Cxz[l,k] + 
                                        imOmR[l]*Cxz[k,l]))
        dCyy[k,l] = (Delta[k]*Cxy[k,l]+Delta[l]*Cxy[l,k]
            -sqrt(gammas[k]*gammas[l])*Cyy[k,l] + Γ[k,l]*(Czz[k,l]+0.5*sz[k]+0.5*sz[l]) + 2.0 * (reOmR[k]*Cyz[l,k] + 
                                        reOmR[l]*Cyz[k,l]))
        dCzz[k,l] = (-2*sqrt(gammas[k]*gammas[l])*Czz[k,l] - sqrt(gammas[k]*gammas[l])*(sz[k]+sz[l]) + Γ[k,l]*(Cyy[k,l]+Cxx[k,l])
                - 2.0 * (reOmR[k]*Cyz[k,l] + reOmR[l]*Cyz[l,k])
                + 2.0 * (imOmR[k]*Cxz[k,l] + imOmR[l]*Cxz[l,k]))
        dCxy[k,l] = (Delta[l]*Cxx[l,k]-Delta[k]*Cyy[k,l]+
            Ω[k,l]*(sz[k]-sz[l]) - sqrt(gammas[k]*gammas[l])*Cxy[k,l] 
                + 2.0*reOmR[l]*Cxz[k,l] - 2.0*imOmR[k]*Cyz[l,k])
        dCxz[k,l] = (-Delta[k]*Cyz[k,l] + 
            Ω[k,l]*sy[l] - 1.5*sqrt(gammas[k]*gammas[l])*Cxz[k,l] - gammas[k]*sx[k] - Γ[k,l]*(Cxz[l,k]+0.5*sx[l]) 
                - 2.0*reOmR[l]*Cxy[k,l] 
                - 2.0*(imOmR[k]*Czz[k,l] - imOmR[l]*Cxx[k,l]))
        dCyz[k,l] = (Delta[k]*Cxz[k,l]
            -Ω[k,l]*sx[l] - 1.5*sqrt(gammas[k]*gammas[l])*Cyz[k,l] - gammas[k]*sy[k] - Γ[k,l]*(Cyz[l,k]+0.5*sy[l])
                + 2.0*(reOmR[k]*Czz[k,l] - reOmR[l]*Cyy[k,l])
                + 2.0*imOmR[l]*Cxy[l,k])
        for j=1:N
            if j==l || j==k
                continue
            end
            Cxxx = correlation(sx[k], sx[l], sx[j], Cxx[k,l], Cxx[k,j], Cxx[l,j])
            Cxxy = correlation(sx[k], sx[l], sy[j], Cxx[k,l], Cxy[k,j], Cxy[l,j])
            Cxyx = correlation(sx[k], sy[l], sx[j], Cxy[k,l], Cxx[k,j], Cxy[j,l])
            Cxyy = correlation(sx[k], sy[l], sy[j], Cxy[k,l], Cxy[k,j], Cyy[l,j])
            Cxzx = correlation(sx[k], sz[l], sx[j], Cxz[k,l], Cxx[k,j], Cxz[j,l])
            Cxzy = correlation(sx[k], sz[l], sy[j], Cxz[k,l], Cxy[k,j], Cyz[j,l])
            Cyxx = correlation(sy[k], sx[l], sx[j], Cxy[l,k], Cxy[j,k], Cxx[l,j])
            Cyxy = correlation(sy[k], sx[l], sy[j], Cxy[l,k], Cyy[k,j], Cxy[l,j])
            Cyyx = correlation(sy[k], sy[l], sx[j], Cyy[k,l], Cxy[j,k], Cxy[j,l])
            Cyyy = correlation(sy[k], sy[l], sy[j], Cyy[k,l], Cyy[k,j], Cyy[l,j])
            Cyzx = correlation(sy[k], sz[l], sx[j], Cyz[k,l], Cxy[j,k], Cxz[j,l])
            Cyzy = correlation(sy[k], sz[l], sy[j], Cyz[k,l], Cyy[k,j], Cyz[j,l])
            Czxx = correlation(sz[k], sx[l], sx[j], Cxz[l,k], Cxz[j,k], Cxx[l,j])
            Czxy = correlation(sz[k], sx[l], sy[j], Cxz[l,k], Cyz[j,k], Cxy[l,j])
            Czyx = correlation(sz[k], sy[l], sx[j], Cyz[l,k], Cxz[j,k], Cxy[j,l])
            Czyy = correlation(sz[k], sy[l], sy[j], Cyz[l,k], Cyz[j,k], Cyy[l,j])
            Czzx = correlation(sz[k], sz[l], sx[j], Czz[k,l], Cxz[j,k], Cxz[j,l])
            Czzy = correlation(sz[k], sz[l], sy[j], Czz[k,l], Cyz[j,k], Cyz[j,l])

            dCxx[k,l] += Ω[k,j]*Czxy + Ω[l,j]*Cxzy + 0.5*Γ[k,j]*Czxx + 0.5*Γ[l,j]*Cxzx
            dCyy[k,l] += -Ω[k,j]*Czyx - Ω[l,j]*Cyzx + 0.5*Γ[k,j]*Czyy + 0.5*Γ[l,j]*Cyzy
            dCzz[k,l] += (Ω[k,j]*(Cyzx-Cxzy) + Ω[l,j]*(Czyx-Czxy)
                            - 0.5*Γ[k,j]*(Cxzx+Cyzy) - 0.5*Γ[l,j]*(Czxx+Czyy))
            dCxy[k,l] += Ω[k,j]*Czyy - Ω[l,j]*Cxzx + 0.5*Γ[k,j]*Czyx + 0.5*Γ[l,j]*Cxzy
            dCxz[k,l] += (Ω[k,j]*Czzy + Ω[l,j]*(Cxyx-Cxxy)
                            + 0.5*Γ[k,j]*Czzx - 0.5*Γ[l,j]*(Cxxx+Cxyy))
            dCyz[k,l] += (-Ω[k,j]*Czzx + Ω[l,j]*(Cyyx-Cyxy)
                            + 0.5*Γ[k,j]*Czzy - 0.5*Γ[l,j]*(Cyxx+Cyyy))
        end
    end
end


"""
    mpc_module.timeevolution_field(T, S::SpinCollection, Om_R::Vector ,state0[; fout])
MPC time evolution.
# Arguments
* `T`: Points of time for which output will be generated.
* `S`: SpinCollection describing the system.
* `E`: Field inpinging the system
* `state0`: Initial MPCState.
* `fout` (optional): Function with signature fout(t, state) that is called
    whenever output should be generated.
"""
function timeevolution_field(T, S::SpinCollection, Om_R::Vector{ComplexF64}, state0::MPCState; fout=nothing, kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Delta = [spin.delta for spin in S.spins] 
    gammas = [gamma for gamma in S.gammas] 
    Ω = real(interaction_module.OmegaMatrix(S))
    Γ = real(interaction_module.GammaMatrix(S))
    reOmR = real(Om_R)
    imOmR = imag(Om_R)
    p = (N, gammas, Delta, Ω, Γ, reOmR, imOmR)

    if isa(fout, Nothing)
        fout_ = (t, state) -> deepcopy(state)
    else
        fout_ = fout
    end

    return integrate(T, f, state0, p, fout_; kwargs...)
end


"""
    mpc_module.steady_state_field(T, S::SpinCollection, Om_R::Vector ,state0[; fout])
MPC steady state.
# Arguments
* `T`: Points of time for which output will be generated.
* `S`: SpinCollection describing the system.
* `E`: Field inpinging the system
* `state0`: Initial MPCState.
* `fout` (optional): Function with signature fout(t, state) that is called
    whenever output should be generated.
"""
function steady_state_field(T, S::SpinCollection, Om_R::Vector{ComplexF64}, state0::MPCState; fout=nothing, kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Delta = [spin.delta for spin in S.spins] 
    gammas = [gamma for gamma in S.gammas] 
    Ω = real(interaction_module.OmegaMatrix(S))
    Γ = real(interaction_module.GammaMatrix(S))
    reOmR = real(Om_R)
    imOmR = imag(Om_R)
    p = (N, gammas, Delta, Ω, Γ, reOmR, imOmR)

    if isa(fout, Nothing)
        fout_ = (t, state) -> deepcopy(state)
    else
        fout_ = fout
    end

    return steady_state(f, state0, p, fout_; kwargs...)
end


"""
    mpc_module.state_from_mf(state_mf, phi, theta, N)

Creates the state for MPC solver using the MF state (sx, sy, sz)

Correlation functions of all spins have the same azimuthal angle `phi` and polar angle `theta`.
"""
function state_from_mf(state_mf::AtomicArrays.meanfield_module.ProductState, phi::Real, theta::Real, N::Int=1)
    sx_mf, sy_mf, sz_mf = AtomicArrays.meanfield_module.splitstate(state_mf)
    state = blochstate(phi, theta, N)
    sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = splitstate(state)
    for k=1:N
        sx[k] = sx_mf[k]
        sy[k] = sy_mf[k]
        sz[k] = sz_mf[k]
    end
    return state
end


"""
    mpc_module.mapexpect(op, states, num)

Expectation values for a operator of a spin collection.
# Arguments
* `op`: operator
* `states`: solution of the meanfield equation
* `num`: number of atom in the collection
"""
mapexpect(op, states::Array{MPCState{Int64, Float64}, 1}, num::Int) = map(s->(op(s)[num]), states)


"""
    mpc_module.sigma_matrices(states, t_ind)

Expectation values for sx, sy, sz, sm, sp of a spin collection.
# Arguments
* `states`: solution of the meanfield equation
* `t_ind`: index in the time array
"""
function sigma_matrices(states::Array{MPCState{Int64, Float64}, 1}, t_ind::Int)
    n = states[1].N
    σx = [mapexpect(sx, states, i)[t_ind] for i=1:n]
    σy = [mapexpect(sy, states, i)[t_ind] for i=1:n]
    σz = [mapexpect(sz, states, i)[t_ind] for i=1:n]
    σm = 0.5.*(σx .- 1im.*σy)
    σp = 0.5.*(σx .+ 1im.*σy)
    return [σx, σy, σz, σm, σp]
end


"""
    mpc_module.splitstate_sym(N, sym_data)
Returns sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz.
"""
function splitstate_sym(N::Int, sym_data::Vector{Symbolics.Num})
    data = reshape(sym_data, 3*N, 2*N+1)
    sx = view(data, 1:1*N, 2*N+1)
    sy = view(data, 1*N+1:2*N, 2*N+1)
    sz = view(data, 2*N+1:3*N, 2*N+1)
    Cxx = view(data, 1:1*N, 1:1*N)
    Cyy = view(data, 1*N+1:2*N, 1:1*N)
    Czz = view(data, 2*N+1:3*N, 1:1*N)
    Cxy = view(data, 1:1*N, 1*N+1:2*N)
    Cxz = view(data, 1*N+1:2*N, 1*N+1:2*N)
    Cyz = view(data, 2*N+1:3*N, 1*N+1:2*N)
    return sx, sy, sz, Cxx, Cyy, Czz, Cxy, Cxz, Cyz
end


"""
    mpc_module.integrate()
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
    mpc_module.steady_state()
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
            kwargs...)

    return sol

end


end # module