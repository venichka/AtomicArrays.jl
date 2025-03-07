module interaction

using ..AtomicArrays

using LinearAlgebra

export GreenTensor, OmegaMatrix, GammaMatrix, OmegaTensor_4level, GammaTensor_4level

const N_sublevels = 3

"""
    interaction.F(ri::Vector, rj::Vector, µi::Vector, µj::Vector, ki::Real, kj::Real)
General F function for arbitrary positions and dipole orientations.
Arguments:
* ri: Position of first spin
* rj: Position of second spin
* µi: Dipole orientation of first spin.
* µj: Dipole orientation of second spin.
* ki: module of k vector (ω₁/c) of first spin.
* kj: module of k vector (ω₂/c) of second spin.
"""
function F(ri::Vector, rj::Vector, µi::Vector, µj::Vector, 
           ki::Real, kj::Real)
    rij = ri - rj
    rij_norm = norm(rij)
    rijn = rij./rij_norm
    μi_ = normalize(μi)
    μj_ = normalize(μj)
    T = float(promote_type(eltype(rij),eltype(μi_),eltype(μj_)))
    if rij_norm == 0
        T(2/3.)
    else
        ξ = 0.5*(ki+kj)*rij_norm
        T(dot(µi_, µj_)*(sin(ξ)/ξ + cos(ξ)/ξ^2 - sin(ξ)/ξ^3) + dot(µi_, rijn)*dot(rijn, µj_)*(-sin(ξ)/ξ - 3*cos(ξ)/ξ^2 + 3*sin(ξ)/ξ^3))
    end
end


"""
    interaction.G(ri::Vector, rj::Vector, µi::Vector, µj::Vector, ki::Real, kj::Real)
General G function for arbitrary positions and dipole orientations.
Arguments:
* ri: Position of first spin
* rj: Position of second spin
* µi: Dipole orientation of first spin.
* µj: Dipole orientation of second spin.
* ki: k number (ω₁/c) of first spin.
* kj: k number (ω₂/c) of second spin.
"""
function G(ri::Vector, rj::Vector, µi::Vector, µj::Vector,
           ki::Real, kj::Real)
    rij = ri - rj
    rij_norm = norm(rij)
    rijn = rij./rij_norm
    μi_ = normalize(μi)
    μj_ = normalize(μj)
    T = float(promote_type(eltype(rij),eltype(μi_),eltype(μj_)))
    if rij_norm == 0
        zero(T)
    else
        ξ = 0.5*(ki+kj)*rij_norm
        T(dot(µi_, µj_)*(-cos(ξ)/ξ + sin(ξ)/ξ^2 + cos(ξ)/ξ^3) + dot(µi_, rijn)*dot(rijn, µj_)*(cos(ξ)/ξ - 3*sin(ξ)/ξ^2 - 3*cos(ξ)/ξ^3))
    end
end


"""
    interaction.Omega(ri::Vector, rj::Vector, µi::Vector, µj::Vector, γi::Real=1, γj::Real=1)
Arguments:
* ri: Position of first spin
* rj: Position of second spin
* µi: Dipole orientation of first spin.
* µj: Dipole orientation of second spin.
* γi: Decay rate of first spin.
* γj: Decay rate of second spin.
Note that the dipole moments `μi` and `μj` are normalized internally. To account
for dipole moments with different lengths you need to scale the decay rates
`γi` and `γj`, respectively.
"""
function Omega(ri::Vector, rj::Vector, µi::Vector, µj::Vector, γi::Real=1, γj::Real=1, ki::Real=2π, kj::Real=2π)
    return 0.75*sqrt(γi*γj)*G(ri, rj, µi, µj, ki, kj)
end


"""
    interaction.Gamma(ri::Vector, rj::Vector, µi::Vector, µj::Vector, γi::Real=1, γj::Real=1)
Arguments:
* ri: Position of first spin
* rj: Position of second spin
* µi: Dipole orientation of first spin.
* µj: Dipole orientation of second spin.
* γi: Decay rate of first spin.
* γj: Decay rate of second spin.
Note that the dipole moments `μi` and `μj` are normalized internally. To account
for dipole moments with different lengths you need to scale the decay rates
`γi` and `γj`, respectively.
"""
function Gamma(ri::Vector, rj::Vector, µi::Vector, µj::Vector, γi::Real=1, γj::Real=1, ki::Real=2π, kj::Real=2π)
    return 1.5*sqrt(γi*γj)*F(ri, rj, µi, µj, ki, kj)
end


"""
    interaction.OmegaMatrix(S::SpinCollection)
Matrix of the dipole-dipole interaction for a given SpinCollection.
"""
function OmegaMatrix(S::SpinCollection)
    spins = S.spins
    mu = S.polarizations
    gamma = S.gammas
    N = length(spins)
    Ω = zeros(Float64, N, N)
    for i=1:N, j=1:N
        if i==j
            continue
        end
        Ω[i,j] = Omega(spins[i].position, spins[j].position, mu[i], mu[j], gamma[i], gamma[j],spins[i].delta+2π,spins[j].delta+2π)
    end
    return Ω
end


"""
    interaction.GammaMatrix(S::SpinCollection)
Matrix of the collective decay rate for a given SpinCollection.
"""
function GammaMatrix(S::SpinCollection)
    spins = S.spins
    mu = S.polarizations
    gamma = S.gammas
    N = length(spins)
    return [
        Gamma(spins[i].position, spins[j].position, mu[i], mu[j], gamma[i], gamma[j],spins[i].delta+2π,spins[j].delta+2π)
        for i=1:N, j=1:N
    ]
end


"""
    interaction.GreenTensor(r::Vector, k::Number=2π)
Calculate the Green's Tensor at position r for wave number k defined by
```math
G = e^{ikr}\\Big[\\left(\\frac{1}{kr} + \\frac{i}{(kr)^2} - \\frac{1}{(kr)^3}\\right)*I -
    \\textbf{r}\\textbf{r}^T\\left(\\frac{1}{kr} + \\frac{3i}{(kr)^2} - \\frac{3}{(kr)^3}\\right)\\Big]
```
Choosing `k=2π` corresponds to the position `r` being given in units of the
wavelength associated with the dipole transition.
Returns a 3×3 complex Matrix.
"""
function GreenTensor(r::Vector{<:Number},k::Real=2π)
    n = norm(r)
    rn = r./n
    return exp(im*k*n)*(
        (1/(k*n) + im/(k*n)^2 - 1/(k*n)^3).*Matrix(I,3,3) +
        -(1/(k*n) + 3im/(k*n)^2 - 3/(k*n)^3).*(rn*rn')
    )
end

"""
    interaction.OmegaTensor_4level
Dimensions: NxNx3x3 -- atom i, atom j, polarization k, polarization m
"""
function OmegaTensor_4level(A::FourLevelAtomCollection)
    atoms = A.atoms
    mu = A.polarizations
    gamma = A.gammas
    N = length(atoms)
    return [
        (i == j && k != m) ? 0.0 : Omega(atoms[i].position, atoms[j].position, mu[k,:,i], mu[m,:,j], gamma[k, i], gamma[m, j], atoms[i].delta+2π, atoms[j].delta+2π)
        for i=1:N, j=1:N, k=1:N_sublevels, m=1:N_sublevels
    ]
end

# TODO: check the correctnes of flattening

"""
    interaction.OmegaMatrix_4level
Dimensions: 3Nx3N -- including atoms and all 3 transitions
"""
function OmegaMatrix_4level(A::FourLevelAtomCollection)
    atoms = A.atoms
    mu = A.polarizations
    gamma = A.gammas
    N = length(atoms)
    M = N_sublevels * N  # matrix rank (3 transitions, N atoms)
    Omega_matrix = zeros(ComplexF64, M, M)
    for n = 1:N
        for m = 1:N_sublevels
            i = N_sublevels*(n - 1) + m
            for nprime = 1:N
                for mprime = 1:N_sublevels
                    j = N_sublevels * (nprime - 1) + mprime
                    Omega_matrix[i,j] = (n==nprime && m!=mprime) ? 0.0 : Omega(
                        atoms[n].position, atoms[nprime].position,
                        mu[m,:,n], mu[mprime,:,nprime],
                        gamma[m, n], gamma[mprime, nprime],
                        atoms[n].delta+2π, atoms[nprime].delta+2π)
                end
            end
        end
    end
    return Omega_matrix
end

"""
    interaction.GammaTensor_4level
Dimensions: NxNx3x3 -- atom i, atom j, polarization k, polarization m
"""
function GammaTensor_4level(A::FourLevelAtomCollection)
    atoms = A.atoms
    mu = A.polarizations
    gamma = A.gammas
    N = length(atoms)
    return [
        (i == j && k != m) ? 0.0 : Gamma(atoms[i].position, atoms[j].position, mu[k,:,i], mu[m,:,j], gamma[k, i], gamma[m, j], atoms[i].delta+2π, atoms[j].delta+2π)
        for i=1:N, j=1:N, k=1:N_sublevels, m=1:N_sublevels
    ]
end

"""
    interaction.GammaMatrix_4level
Dimensions: 3Nx3N -- including atoms and all 3 transitions
"""
function GammaMatrix_4level(A::FourLevelAtomCollection)
    atoms = A.atoms
    mu = A.polarizations
    gamma = A.gammas
    N = length(atoms)
    M = N_sublevels * N  # matrix rank (3 transitions, N atoms)
    Gamma_matrix = zeros(ComplexF64, M, M)
    for n = 1:N
        for m = 1:N_sublevels
            i = N_sublevels*(n - 1) + m
            for nprime = 1:N
                for mprime = 1:N_sublevels
                    j = N_sublevels * (nprime - 1) + mprime
                    Gamma_matrix[i,j] = (n==nprime && m!=mprime) ? 0.0 : Gamma(
                        atoms[n].position, atoms[nprime].position,
                        mu[m,:,n], mu[mprime,:,nprime],
                        gamma[m, n], gamma[mprime, nprime],
                        atoms[n].delta+2π, atoms[nprime].delta+2π)
                end
            end
        end
    end
    return Gamma_matrix
end

end # module