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


end #module