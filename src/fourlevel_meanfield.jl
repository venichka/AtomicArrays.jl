module fourlevel_meanfield

#TODO: add functions creating the state from density matrix and as a blochstate
#TODO: add timeevolution, expectation values for Complex ProductState

export ProductState, ProductState_Complex

import ..integrate_base

import OrdinaryDiffEq, DiffEqCallbacks, SteadyStateDiffEq
import NonlinearSolve, SparseDiffTools, ForwardDiff, LinearSolve, SparseArrays, ADTypes, SparseConnectivityTracer
import Sundials

using ..interaction, ..AtomicArrays

"""
Class describing a Meanfield state (complex Product state) for 4-level atoms.
The data layout is [s_m1_1 s_0_1 s_p1_1 ... s_m1_m1_1 s_m1_0_1 s_m1_p1_1 
                                            s_0_m1_1 s_0_0_1 s_0_p1_1 
                                            s_p1_m1_1 s_p1_0_1 s_p1_p1_1 ...]
# Arguments
* `N`: Number of atoms.
* `data`: Vector of length 12*N = 3*N + 9*N.
"""
mutable struct ProductState_Complex{T1<:Int,T2<:Complex}
    N::T1
    data::Vector{T2}
end

"""
    fourlevel_meanfield.ProductState_Complex(N)
Meanfield state with all Pauli expectation values equal to zero.
"""
ProductState_Complex(N::T) where T<:Int = ProductState_Complex(N,
                                            zeros(complex(float(T)), 12*N))

"""
    fourlevel_meanfield.ProductState_Complex(data)
Meanfield state created from complex valued vector of length 12*atomnumber.
"""
ProductState_Complex(data::Vector{<:Complex}) = ProductState_Complex(
                                                                dim(data), data)

"""
Class describing a Meanfield state (real Product state) for 4-level atoms.
The data layout is [re(s_m1_1) re(s_0_1) re(s_p1_1) ... 
                    im(s_m1_1) im(s_0_1) im(s_p1_1) ...
                    re(s_m1_m1_1) re(s_m1_0_1) re(s_m1_p1_1) 
                    re(s_0_m1_1) re(s_0_0_1) re(s_0_p1_1)
                    re(s_p1_m1_1) re(s_p1_0_1) re(s_p1_p1_1) ...
                    same for imaginary part ...]
# Arguments
* `N`: Number of atoms.
* `data`: Vector of length 24*N = 3*N + 3*N + 9*N + 9*N.
"""
mutable struct ProductState{T1<:Int,T2<:Real}
    N::T1
    data::Vector{T2}
end

"""
    fourlevel_meanfield.ProductState(N)
Meanfield state with all transition and population operators expectation values equal to zero.
"""
ProductState(N::T) where T<:Int = ProductState(N, zeros(float(T), 24*N))

"""
    fourlevel_meanfield.ProductState(data)
Meanfield state created from real valued vector of length 24*atomnumber.
"""
ProductState(data::Vector{<:Real}) = ProductState(dim(data), data)

"""
    fourlevel_meanfield.dim(state)
Number of spins described by this state.
"""
function dim(state::Vector{T}) where T<:Real
    N, rem = divrem(length(state), 24)
    @assert rem==0
    return N
end
function dim(state::Vector{T}) where T<:Complex
    N, rem = divrem(length(state), 12)
    @assert rem==0
    return N
end

"""
    fourlevel_meanfield.splitstate(N, data)
    fourlevel_meanfield.splitstate(state)
Split state into sm_m, smm parts for complex state
and for re(sm_m), im(sm_m), re(smm), im(smm) for real state.
"""
splitstate(N::Int, data::Vector{<:Real}) = (reshape(view(data, 1:3*N), (3,N)),
                                    reshape(view(data, 3*N+1:6*N), (3,N)),
                                    reshape(view(data, 6*N+1:15*N), (3,3,N)),
                                    reshape(view(data, 15*N+1:24*N), (3,3,N)))
splitstate(state::ProductState) = splitstate(state.N, state.data)
splitstate(N::Int, data::Vector{<:Complex}) = (reshape(view(data, 1:3*N),(3,N)),
                                    reshape(view(data, 3*N+1:12*N),(3,3,N)))
splitstate(state::ProductState_Complex) = splitstate(state.N, state.data)


"""
    fourlevel_meanfield.sm(state)
Sigma_m expectation values of state.
"""
sm(x::ProductState) = reshape(view(x.data, 1:3*x.N) .+
                              1.0im*view(x.data, 3*x.N+1:6*x.N),
                              (3, x.N))
sm(x::ProductState_Complex) = reshape(view(x.data, 1:3*x.N), (3, x.N))

"""
    fourlevel_meanfield.smm(state)
Sigma_mm expectation values of state.
"""
smm(x::ProductState) = reshape(view(x.data, 6*x.N+1:15*x.N) .+
                              1.0im*view(x.data, 15*x.N+1:24*x.N),
                              (3, 3, x.N))
smm(x::ProductState_Complex) = reshape(view(x.data, 3*x.N+1:12*x.N), (3,3,x.N))

"""
    fourlevel_meanfield.mapexpect(op, states, num)

Expectation values for a operator of a spin collection.
# Arguments
* `op`: operator
* `states`: solution of the meanfield equation
* `num`: number of atom in the collection
* `m`: number of sublevel (1: m=-1, 2: m=0, 3: m=+1) in s_m
    * `m1`: 1 sublevel in s_mm
    * `m2`: 2 sublevel in s_mm
"""
mapexpect(op, states::Array{ProductState{Int64, Float64}, 1}, num::Int, m::Int) = map(s->(op(s)[m, num]), states)
mapexpect(op, states::Array{ProductState{Int64, Float64}, 1}, num::Int, m1::Int, m2::Int) = map(s->(op(s)[m1, m2, num]), states)
mapexpect(op, states::Array{ProductState{Int64, Float64}, 1}, num::Int) = map(s->(op(s)[:, num]), states)

"""
    meanfield.sigma_matrices(states, t_ind)

Expectation values for sm and smm in atom collection
# Arguments
* `states`: solution of the meanfield equation
* `t_ind`: index in the time array
"""
function sigma_matrices(states::Array{ProductState{Int64, Float64}, 1}, t_ind::Int)
    N = length(states[1].data)÷24
    σm = [mapexpect(sm, states, n, m)[t_ind] for m=1:3, n=1:N]
    σmm = [mapexpect(smm, states, n, m1, m2)[t_ind] for m1=1:3, m2=1:3, n=1:N]
    return σm, σmm
end

"""
    fourlevel_meanfield.timeevolution(T, A::FourLevelAtomCollection, 
                                      OmR::Array{Complex{Float64}, 2}, 
                                      B_z::Real,
                                      state0[; fout])
Meanfield time evolution.
# Arguments
* `T`: Points of time for which output will be generated.
* `A`: [`FourLevelAtomCollection`](@ref) describing the system.
* `Om_R`: interaction constant with external electric field
* `B_z`: g_J μ_B B, external uniform magnetic field
* `state0`: Initial ProductState.
* `fout` (optional): Function with signature `fout(t, state)` that is called whenever output
    should be generated.
"""
function timeevolution(T, A::FourLevelAtomCollection, OmR::Array{Complex{Float64}, 2}, B_z::Real, state0::ProductState; fout=nothing, kwargs...)
    N = length(A.atoms)
    @assert N==state0.N
    Omega = interaction.OmegaTensor_4level(A)
    Gamma = interaction.GammaTensor_4level(A)
    w = [A.atoms[n].delta + B_z*m for m = -1:1, n = 1:N]
    OmR_r = real(OmR); OmR_i = imag(OmR)
    Omega_r = real(Omega); Omega_i = imag(Omega)
    Gamma_r = real(Gamma); Gamma_i = imag(Gamma)

    function f(du, u, p, t)
        xm, ym, xmm, ymm = splitstate(N, u)
        dxm, dym, dxmm, dymm = splitstate(N, du)
        # sigma equations
        @inbounds for n=1:N
            for m=1:3
                dxm[m,n] = w[m,n]*ym[m,n]
                dym[m,n] = -w[m,n]*xm[m,n]
                for m1 = 1:3
                    dxm[m,n] -= 0.5*(Gamma_r[n,n,m,m1]*xm[m1,n] -
                                    Gamma_i[n,n,m,m1]*ym[m1,n])
                    dym[m,n] -= 0.5*(Gamma_r[n,n,m,m1]*ym[m1,n] +
                                    Gamma_i[n,n,m,m1]*xm[m1,n])
                    if m1 == m
                        x_bar = xmm[m1,m,n]-1+xmm[1,1,n]+xmm[2,2,n]+xmm[3,3,n]
                    else
                        x_bar = xmm[m1,m,n]
                    end
                    dxm[m,n] -= OmR_i[m1,n]*x_bar - OmR_r[m1,n]*ymm[m1,m,n]
                    dym[m,n] -= OmR_r[m1,n]*x_bar + OmR_i[m1,n]*ymm[m1,m,n]
                    for m2 = 1:3
                        for n2 = 1:N
                            if n2 == n
                                continue
                            end
                            dxm[m,n]+=((0.5*Gamma_r[n,n2,m1,m2]-
                                        Omega_i[n,n2,m1,m2])*
                                        (x_bar*xm[m2,n2] - ymm[m1,m,n]*ym[m2,n2]) -
                                    (0.5*Gamma_i[n,n2,m1,m2]+
                                        Omega_r[n,n2,m1,m2])*
                                        (x_bar*ym[m2,n2] + ymm[m1,m,n]*xm[m2,n2]))
                            dym[m,n]+=((0.5*Gamma_i[n,n2,m1,m2]+
                                        Omega_r[n,n2,m1,m2])*
                                        (x_bar*xm[m2,n2] - ymm[m1,m,n]*ym[m2,n2]) +
                                    (0.5*Gamma_r[n,n2,m1,m2]-
                                        Omega_i[n,n2,m1,m2])*
                                        (x_bar*ym[m2,n2] + ymm[m1,m,n]*xm[m2,n2]))
                        end
                    end
                end
            end
        end
        # population equations
        @inbounds for n = 1:N
            for m = 1:3
                for m1 = 1:3
                    dxmm[m1,m,n] = (-(w[m1,n]-w[m,n])*ymm[m1,m,n]
                                    +OmR_r[m1,n]*ym[m,n]+OmR_i[m1,n]*xm[m,n]
                                    +OmR_r[m,n]*ym[m1,n]+OmR_i[m,n]*xm[m1,n])
                    dymm[m1,m,n] = ((w[m1,n]-w[m,n])*xmm[m1,m,n]
                                    -OmR_r[m1,n]*xm[m,n]+OmR_i[m1,n]*ym[m,n]
                                    +OmR_r[m,n]*xm[m1,n]-OmR_i[m,n]*ym[m1,n])
                    for m2 = 1:3
                        dxmm[m1,m,n] -= 0.5*(Gamma_r[n,n,m2,m1]*xmm[m2,m,n] -
                                            Gamma_i[n,n,m2,m1]*ymm[m2,m,n] +
                                            Gamma_r[n,n,m,m2]*xmm[m1,m2,n] -
                                            Gamma_i[n,n,m,m2]*ymm[m1,m2,n])
                        dymm[m1,m,n] -= 0.5*(Gamma_r[n,n,m2,m1]*ymm[m2,m,n] +
                                            Gamma_i[n,n,m2,m1]*xmm[m2,m,n] +
                                            Gamma_r[n,n,m,m2]*ymm[m1,m2,n] +
                                            Gamma_i[n,n,m,m2]*xmm[m1,m2,n])
                        for n1 = 1:N
                            if n1 == n
                                continue
                            end
                            dxmm[m1,m,n] -=((0.5*Gamma_r[n1,n,m2,m1]+
                                            Omega_i[n1,n,m2,m1])*
                                            (xm[m2,n1]*xm[m,n]+ym[m2,n1]*ym[m,n])-
                                            (0.5*Gamma_i[n1,n,m2,m1]-
                                            Omega_r[n1,n,m2,m1])*
                                            (xm[m2,n1]*ym[m,n]-ym[m2,n1]*xm[m,n])+
                                            (0.5*Gamma_r[n,n1,m,m2]-
                                            Omega_i[n,n1,m,m2])*
                                            (xm[m1,n]*xm[m2,n1]+ym[m1,n]*ym[m2,n1])-
                                            (0.5*Gamma_i[n,n1,m,m2]+
                                            Omega_r[n,n1,m,m2])*
                                            (xm[m1,n]*ym[m2,n1]-ym[m1,n]*xm[m2,n1]))
                            dymm[m1,m,n] -=((0.5*Gamma_i[n1,n,m2,m1]-
                                            Omega_r[n1,n,m2,m1])*
                                            (xm[m2,n1]*xm[m,n]+ym[m2,n1]*ym[m,n])+
                                            (0.5*Gamma_r[n1,n,m2,m1]+
                                            Omega_i[n1,n,m2,m1])*
                                            (xm[m2,n1]*ym[m,n]-ym[m2,n1]*xm[m,n])+
                                            (0.5*Gamma_i[n,n1,m,m2]+
                                            Omega_r[n,n1,m,m2])*
                                            (xm[m1,n]*xm[m2,n1]+ym[m1,n]*ym[m2,n1])+
                                            (0.5*Gamma_r[n,n1,m,m2]-
                                            Omega_i[n,n1,m,m2])*
                                            (xm[m1,n]*ym[m2,n1]-ym[m1,n]*xm[m2,n1]))
                        end
                    end
                end
            end
        end
        @inbounds for m = 1:3
            for m1 = m:3
                dxmm[m, m1, :] .= dxmm[m1, m, :]
                dymm[m, m1, :] .= -dymm[m1, m, :]
                if m1 == m
                    dymm[m, m1, :] .= 0.0
                end
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
    fourlevel_meanfield.f(dy, y, p, t)

Meanfield equations
"""
function f(du, u, p, t)
    N = Int(length(u) / 24)
    w, OmR, Omega, Gamma = p
    OmR_r = real(OmR); OmR_i = imag(OmR)
    Omega_r = real(Omega); Omega_i = imag(Omega)
    Gamma_r = real(Gamma); Gamma_i = imag(Gamma)
    xm, ym, xmm, ymm = splitstate(N, u)
    dxm, dym, dxmm, dymm = splitstate(N, du)
    # sigma equations
    @inbounds for n=1:N
        for m=1:3
            dxm[m,n] = w[m,n]*ym[m,n]
            dym[m,n] = -w[m,n]*xm[m,n]
            for m1 = 1:3
                dxm[m,n] -= 0.5*(Gamma_r[n,n,m,m1]*xm[m1,n] -
                                 Gamma_i[n,n,m,m1]*ym[m1,n])
                dym[m,n] -= 0.5*(Gamma_r[n,n,m,m1]*ym[m1,n] +
                                 Gamma_i[n,n,m,m1]*xm[m1,n])
                if m1 == m
                    x_bar = xmm[m1,m,n]-1+xmm[1,1,n]+xmm[2,2,n]+xmm[3,3,n]
                else
                    x_bar = xmm[m1,m,n]
                end
                dxm[m,n] -= OmR_i[m1,n]*x_bar - OmR_r[m1,n]*ymm[m1,m,n]
                dym[m,n] -= OmR_r[m1,n]*x_bar + OmR_i[m1,n]*ymm[m1,m,n]
                for m2 = 1:3
                    for n2 = 1:N
                        if n2 == n
                            continue
                        end
                        dxm[m,n]+=((0.5*Gamma_r[n,n2,m1,m2]-
                                    Omega_i[n,n2,m1,m2])*
                                    (x_bar*xm[m2,n2] - ymm[m1,m,n]*ym[m2,n2]) -
                                   (0.5*Gamma_i[n,n2,m1,m2]+
                                    Omega_r[n,n2,m1,m2])*
                                    (x_bar*ym[m2,n2] + ymm[m1,m,n]*xm[m2,n2]))
                        dym[m,n]+=((0.5*Gamma_i[n,n2,m1,m2]+
                                    Omega_r[n,n2,m1,m2])*
                                    (x_bar*xm[m2,n2] - ymm[m1,m,n]*ym[m2,n2]) +
                                   (0.5*Gamma_r[n,n2,m1,m2]-
                                    Omega_i[n,n2,m1,m2])*
                                    (x_bar*ym[m2,n2] + ymm[m1,m,n]*xm[m2,n2]))
                    end
                end
            end
        end
    end
    # population equations
    @inbounds for n = 1:N
        for m = 1:3
            for m1 = 1:3
                dxmm[m1,m,n] = (-(w[m1,n]-w[m,n])*ymm[m1,m,n]
                                +OmR_r[m1,n]*ym[m,n]+OmR_i[m1,n]*xm[m,n]
                                +OmR_r[m,n]*ym[m1,n]+OmR_i[m,n]*xm[m1,n])
                dymm[m1,m,n] = ((w[m1,n]-w[m,n])*xmm[m1,m,n]
                                -OmR_r[m1,n]*xm[m,n]+OmR_i[m1,n]*ym[m,n]
                                +OmR_r[m,n]*xm[m1,n]-OmR_i[m,n]*ym[m1,n])
                for m2 = 1:3
                    dxmm[m1,m,n] -= 0.5*(Gamma_r[n,n,m2,m1]*xmm[m2,m,n] -
                                         Gamma_i[n,n,m2,m1]*ymm[m2,m,n] +
                                         Gamma_r[n,n,m,m2]*xmm[m1,m2,n] -
                                         Gamma_i[n,n,m,m2]*ymm[m1,m2,n])
                    dymm[m1,m,n] -= 0.5*(Gamma_r[n,n,m2,m1]*ymm[m2,m,n] +
                                         Gamma_i[n,n,m2,m1]*xmm[m2,m,n] +
                                         Gamma_r[n,n,m,m2]*ymm[m1,m2,n] +
                                         Gamma_i[n,n,m,m2]*xmm[m1,m2,n])
                    for n1 = 1:N
                        if n1 == n
                            continue
                        end
                        dxmm[m1,m,n] -=((0.5*Gamma_r[n1,n,m2,m1]+
                                        Omega_i[n1,n,m2,m1])*
                                        (xm[m2,n1]*xm[m,n]+ym[m2,n1]*ym[m,n])-
                                        (0.5*Gamma_i[n1,n,m2,m1]-
                                        Omega_r[n1,n,m2,m1])*
                                        (xm[m2,n1]*ym[m,n]-ym[m2,n1]*xm[m,n])+
                                        (0.5*Gamma_r[n,n1,m,m2]-
                                        Omega_i[n,n1,m,m2])*
                                        (xm[m1,n]*xm[m2,n1]+ym[m1,n]*ym[m2,n1])-
                                        (0.5*Gamma_i[n,n1,m,m2]+
                                        Omega_r[n,n1,m,m2])*
                                        (xm[m1,n]*ym[m2,n1]-ym[m1,n]*xm[m2,n1]))
                        dymm[m1,m,n] -=((0.5*Gamma_i[n1,n,m2,m1]-
                                        Omega_r[n1,n,m2,m1])*
                                        (xm[m2,n1]*xm[m,n]+ym[m2,n1]*ym[m,n])+
                                        (0.5*Gamma_r[n1,n,m2,m1]+
                                        Omega_i[n1,n,m2,m1])*
                                        (xm[m2,n1]*ym[m,n]-ym[m2,n1]*xm[m,n])+
                                        (0.5*Gamma_i[n,n1,m,m2]+
                                        Omega_r[n,n1,m,m2])*
                                        (xm[m1,n]*xm[m2,n1]+ym[m1,n]*ym[m2,n1])+
                                        (0.5*Gamma_r[n,n1,m,m2]-
                                        Omega_i[n,n1,m,m2])*
                                        (xm[m1,n]*ym[m2,n1]-ym[m1,n]*xm[m2,n1]))
                    end
                end
            end
        end
    end
    @inbounds for m = 1:3
        for m1 = m:3
            dxmm[m, m1, :] .= dxmm[m1, m, :]
            dymm[m, m1, :] .= -dymm[m1, m, :]
            if m1 == m
                dymm[m, m1, :] .= 0.0
            end
        end
    end
end


"""
    fourlevel_meanfield.f_sym(dy, y, p, t)

Meanfield equations for symbolic computations
"""
function f_sym(du, u, p, t)
    N = Int(length(u) / 12)
    w, OmR, Omega, Gamma = p
    sm = reshape(view(u, 1:3*N), (3, N))
    smm = reshape(view(u, (3*N)+1:12*N), (3, 3, N))
    dsm = reshape(view(du, 1:3*N), (3, N))
    dsmm = reshape(view(du, (3*N)+1:12*N), (3, 3, N))
    # sigma equations
    @inbounds for n=1:N
        for m=1:3
            dsm[m,n] = ((-1im*w[m,n] - 0.5*Gamma[n,n,m,m])*sm[m,n])
            for m1 = 1:3
                if m1 == m
                    s_bar = smm[m1,m,n]-1+smm[1,1,n]+smm[2,2,n]+smm[3,3,n]
                else
                    s_bar = smm[m1,m,n]
                end
                dsm[m,n] += -1im*conj(OmR[m1,n])*s_bar
                for m2 = 1:3
                    for n2 = 1:N
                        if n2 == n
                            continue
                        end
                        dsm[m,n] += (1im*Omega[n,n2,m1,m2]+
                                     0.5*Gamma[n,n2,m1,m2])*s_bar*sm[m2,n2] 
                    end
                end
            end
        end
    end
    # population equations
    @inbounds for n = 1:N
        for m = 1:3
            for m1 = 1:3
                dsmm[m1,m,n] = (1im*(w[m1,n] - w[m,n])*smm[m1,m,n] -
                                1im*OmR[m1,n]*sm[m,n] +
                                1im*conj(OmR[m,n])*sm[m1,n]')
                for m2 = 1:3
                    dsmm[m1,m,n] += - (0.5*Gamma[n,n,m2,m1]*smm[m2,m,n] +
                                     0.5*Gamma[n,n,m,m2]*smm[m1,m2,n])
                    for n1 = 1:N
                        if n1 == n
                            continue
                        end
                        dsmm[m1,m,n] += ((1im*Omega[n1,n,m2,m1] -
                                          0.5*Gamma[n1,n,m2,m1])*
                                          sm[m2,n1]'*sm[m,n] +
                                         (-1im*Omega[n,n1,m,m2] -
                                          0.5*Gamma[n,n1,m,m2] )*
                                          sm[m1,n]'*sm[m2,n1])
                    end
                end
            end
        end
    end
end

"""
    fourlevel_meanfield.steady_state(A::FourLevelAtomCollection,
                    OmR::Array{Complex{Float64}, 2}, B_z::Real, state0[; fout])
MPC steady state.
# Arguments
* `A`: FourLevelAtomCollection describing the system.
* `Om_R`: Rabi constant
* `B_z`: g_J μ_B B, external uniform magnetic field
* `state0`: Initial MPCState.
* `fout` (optional): Function with signature fout(t, state) that is called
    whenever output should be generated.
"""
function steady_state(A::FourLevelAtomCollection,
                Om_R::Array{Complex{Float64}, 2}, B_z::Real,
                state0::ProductState; 
                fout=nothing,
                alg::SteadyStateDiffEq.SteadyStateDiffEqAlgorithm=SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.VCABM()),
                kwargs...)
    N = length(A.atoms)
    @assert N==state0.N
    Omega = interaction.OmegaTensor_4level(A)
    Gamma = interaction.GammaTensor_4level(A)
    w = [A.atoms[n].delta + B_z*m for m = -1:1, n = 1:N]
    p = (w, Om_R, Omega, Gamma)

    if isa(fout, Nothing)
        fout_ = (t, state) -> deepcopy(state)
    else
        fout_ = fout
    end

    return AtomicArrays.meanfield.steady_state(f, state0, p, fout_;
                                               alg=alg, kwargs...)
end

# Cache for Jacobian sparsity patterns
const jacobian_sparsity_cache = Dict{Tuple{Int, Int}, SparseArrays.SparseMatrixCSC{Float64, Int}}()

"""
    fourlevel_meanfield.steady_state_nonlinear(A::FourLevelAtomCollection, 
                 Om_R::Matrix{ComplexF64},
                 B_z::Real, state0::ProductState; kwargs...)

Solves the meanfield steady-state problem using NonlinearSolve and a sparse Jacobian with caching.
"""
function steady_state_nonlinear(A::FourLevelAtomCollection, 
    Om_R::Array{ComplexF64,2}, B_z::Real, state0::ProductState; 
    abstol=1e-8, reltol=1e-8, maxiters=100)
    
    N = state0.N
    Omega = interaction.OmegaTensor_4level(A)
    Gamma = interaction.GammaTensor_4level(A)
    w = [A.atoms[n].delta + B_z * m for m = -1:1, n = 1:N]
    p = (w, Om_R, Omega, Gamma)

    function f_steady!(du, u, p)
        f(du, u, p, 0.0)
    end

    function jacobian!(J, u, p)
        SparseDiffTools.forwarddiff_color_jacobian!(J, (du, u) -> f_steady!(du, u, p), u)
    end

    u0 = copy(state0.data)
    cache_key = (length(u0), length(p))
    sparsity = get!(jacobian_sparsity_cache, cache_key) do
        detector = SparseConnectivityTracer.TracerSparsityDetector()
        ADTypes.jacobian_sparsity((du, u) -> f_steady!(du, u, p), u0, u0, detector)
    end

    nlfun = NonlinearSolve.NonlinearFunction(f_steady!, jac=jacobian!, jac_prototype=sparsity)
    prob = NonlinearSolve.NonlinearProblem(nlfun, u0, p)
    linsolve = LinearSolve.KrylovJL_GMRES()
    sol = NonlinearSolve.solve(prob, NonlinearSolve.NewtonRaphson(linsolve=linsolve); 
                abstol, reltol, maxiters)

    state0.data .= sol.u
    return state0
end

end # module