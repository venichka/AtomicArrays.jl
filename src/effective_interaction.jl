module effective_interaction

using ..AtomicArrays

include("interaction.jl")
using .interaction


"""
effective_interaction.effective_interactions(S::SpinCollection{T,P,G}) where {T,P,G}
Function computing Ω_eff and Γ_eff for an array of spins at the position of the central spin.
Arguments:
* S: spin collection
"""
function effective_interactions(S::SpinCollection{T,P,G}) where {T,P,G}
    omega_eff::float(G) = 0.
    gamma_eff::float(G) = 0.
    spins = S.spins
    mu = S.polarizations
    gamma = S.gammas
    N = length(spins)
    idx = 1#Int32((N - sqrt(N)) ÷ 2)
    origin = spins[idx].position
    for i = 1:N
        omega_eff += interaction.Omega(origin, spins[i].position, 
        mu[idx], mu[i], gamma[idx], gamma[i], 
        spins[idx].delta + 2π, spins[i].delta + 2π)
        gamma_eff += interaction.Gamma(origin, spins[i].position, 
        mu[idx], mu[i], gamma[idx], gamma[i],
        spins[idx].delta + 2π, spins[i].delta + 2π)
    end
    return omega_eff, gamma_eff
end


"""
effective_interaction.collective_shift_1array(d::Real, Delt::Real, delt::Real, N::Int)
Function for computing collective shifts for a square array.
Arguments:
* d: lattice period of the basic array
* Delt: detuning ("Delta = omega_0 - omega_L")
* delt: difference between lattice constants
* N: number of spins in one dimension (N_tot = N^2)
"""
function collective_shift_1array(d::Real, Delt::Real, delt::Real, N::Int)
    Nx = N
    Ny = N
    pos = geometry.rectangle(d+delt, d+delt; Nx=Nx, Ny=Ny)
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0im, 0.0] for i = 1:Nx*Ny]
    γ_e = [1e-2 for i = 1:Nx*Ny]
    S = SpinCollection(pos,μ; gammas=γ_e, deltas=Delt)
    Omega, Gamma = effective_interaction.effective_interactions(S)
end


"""
effective_interaction.effective_constants(d::Real, Delt::Real, gamma::Real, N::Int)
Function for computing collective shifts for a square array.
Arguments:
* d: lattice period of the basic array
* Delt: detuning ("Delta = omega_0 - omega_L")
* gamma: spontaneous decay rate of an individual atom
* N: number of spins in one dimension (N_tot = N^2)
"""
function effective_constants(d::Real, Delt::Real, gamma::Real, N::Int)
    Nx = N
    Ny = N
    pos = geometry.rectangle(d, d; Nx=Nx, Ny=Ny)
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0im, 0.0] for i = 1:Nx*Ny]
    γ_e = [gamma for i = 1:Nx*Ny]
    S = SpinCollection(pos,μ; gammas=γ_e, deltas=Delt)
    Omega, Gamma = effective_interaction.effective_interactions(S)
end


# Finite symmetric systems

#TODO: fix these constants
const e_z = [0,0,1]
const gamma = 1



function triangle_orthogonal(a::T) where T<:Real
    positions = Vector{float(T)}[[a,0,0], [a/2, sqrt(3. /4)*a,0]]
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function square_orthogonal(a::T) where T<:Real
    positions = Vector{float(T)}[[a,0,0], [0,a,0], [a,a,0]]
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function polygon_orthogonal(N::Int, a::T) where T<:Real
    @assert N>2
    dα = 2*pi/N
    R = a/(2*sin(dα/2))
    positions = Vector{float(T)}[]
    for i=1:(N-1)
        x = R*cos(i*dα)
        y = R*sin(i*dα)
        push!(positions, float(T)[x-R,y,0])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function rectangle_orthogonal(a::U, b::T) where {U<:Real,T<:Real}
    positions = Vector{float(T)}[[a,0,0], [0,b,0], [a,b,0]]
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function cube_orthogonal(a::T) where T<:Real
    positions = Vector{float(T)}[]
    for ix=0:1, iy=0:1, iz=0:1
        if ix==0 && iy==0 && iz==0
            continue
        end
        push!(positions, [ix*a, iy*a, iz*a])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function box_orthogonal(a::T, b::U, c::V) where {T<:Real,U<:Real,V<:Real}
   positions = Vector{float(T)}[]
    for ix=0:1, iy=0:1, iz=0:1
        if ix==0 && iy==0 && iz==0
            continue
        end
        push!(positions, [ix*a, iy*b, iz*c])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end


# Infinite 1D symmetric systems

function chain(a::T, Θ, N::Int) where T<:Real
    positions = Vector{float(T)}[]
    for ix=-N:N
        if ix==0
            continue
        end
        push!(positions, [ix*a, 0., 0.])
    end
    S = SpinCollection(positions, Float64[cos(Θ), 0., sin(Θ)]; gammas=gamma)
    return effective_interactions(S)
end

function chain_orthogonal(a::T, N::Int) where T<:Real
    positions = Vector{float(T)}[]
    for ix=-N:N
        if ix==0
            continue
        end
        push!(positions, [ix*a, 0., 0.])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end


# Infinite 2D symmetric systems

function squarelattice_orthogonal(a::T, N::Int) where {T<:Real}
    positions = Vector{float(T)}[]
    for ix=-N:N, iy=-N:N
        if ix==0 && iy==0
            continue
        end
        push!(positions, [ix*a, iy*a, 0.])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function hexagonallattice_orthogonal(a::T, N::Int) where {T<:Real}
    positions = Vector{float(T)}[]
    ax = sqrt(3.0/4)*a
    for iy=1:N
        push!(positions, [0, iy*a, 0])
        push!(positions, [0, -iy*a, 0])
    end
    for ix=[-1:-2:-N; 1:2:N]
        Ny = div(2*N+1-abs(ix),2)
        for iy=0:Ny-1
            push!(positions, [ax*ix, (0.5+iy)*a, 0])
            push!(positions, [ax*ix, -(0.5+iy)*a, 0])
        end
    end
    for ix=[-2:-2:-N; 2:2:N]
        Ny = div(2*N-abs(ix),2)
        push!(positions, [ax*ix, 0, 0])
        for iy=1:Ny
            push!(positions, [ax*ix, iy*a, 0])
            push!(positions, [ax*ix, -iy*a, 0])
        end
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end


# Infinite 3D symmetric systems

function cubiclattice_orthogonal(a::T, N::Int) where {T<:Real}
    positions = Vector{float(T)}[]
    for ix=-N:N, iy=-N:N, iz=-N:N
        if ix==0 && iy==0 && iz==0
            continue
        end
        push!(positions, [ix*a, iy*a, iz*a])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function tetragonallattice_orthogonal(a::T, b::U, N::Int) where {T<:Real,U<:Real}
    positions = Vector{float(T)}[]
    for ix=-N:N, iy=-N:N, iz=-N:N
        if ix==0 && iy==0 && iz==0
            continue
        end
        push!(positions, [ix*a, iy*a, iz*b])
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end

function hexagonallattice3d_orthogonal(a::T, b::U, N::Int) where {T<:Real,U<:Real}
    positions = Vector{float(T)}[]
    ax = sqrt(3.0/4)*a
    for iz=-N:N
        for iy=1:N
            push!(positions, [0, iy*a, iz*b])
            push!(positions, [0, -iy*a, iz*b])
        end
        for ix=[-1:-2:-N; 1:2:N]
            Ny = div(2*N+1-abs(ix),2)
            for iy=0:Ny-1
                push!(positions, [ax*ix, (0.5+iy)*a, iz*b])
                push!(positions, [ax*ix, -(0.5+iy)*a, iz*b])
            end
        end
        for ix=[-2:-2:-N; 2:2:N]
            Ny = div(2*N-abs(ix),2)
            push!(positions, [ax*ix, 0, iz*b])
            for iy=1:Ny
                push!(positions, [ax*ix, iy*a, iz*b])
                push!(positions, [ax*ix, -iy*a, iz*b])
            end
        end
    end
    S = SpinCollection(positions, e_z; gammas=gamma)
    return effective_interactions(S)
end


end #module
