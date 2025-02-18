using LinearAlgebra

"""
Abstract base class for all systems defined in this library.
Currently there are the following concrete systems:
* Spin
* SpinCollection
* CavityMode
* CavitySpinCollection
* FourLevelAtom
* FourLevelAtomCollection
* CavityFourLevelAtomCollection
"""
abstract type System end


"""
A class representing a single spin.
A spin is defined by its position and its detuning to a main
frequency.
# Arguments
* `position`: A vector defining a point in R3.
* `delta`: Detuning.
"""
struct Spin{T1,T2} <: System
    position::T1
    delta::T2
end
Spin(position::Vector; delta::Real=0) = Spin(position, delta)


"""
A class representing a system consisting of many spins.
# Arguments
* `spins`: Vector of spins.
* `polarizations`: Unit vectors defining the directions of the spins.
* `gammas`: Decay rates.
"""
struct SpinCollection{S<:Spin,P<:Vector,G<:Real} <: System
    spins::Vector{S}
    polarizations::Vector{P}
    gammas::Vector{G}
    function SpinCollection{S,P,G}(spins::Vector{S}, polarizations::Vector{P}, gammas::Vector{G}) where {S,P<:Vector{<:Number},G}
        @assert length(polarizations)==length(spins)
        @assert length(gammas)==length(spins)
        new(spins,normalize.(polarizations),gammas)
    end
end
SpinCollection(spins::Vector{S}, polarizations::Vector{P}, gammas::Vector{G}) where {S,P,G} = SpinCollection{S,P,G}(spins, polarizations, gammas)
SpinCollection(spins::Vector{<:Spin}, polarizations::Vector{<:Vector{<:Number}}, gammas::Number) = SpinCollection(spins, polarizations, [gammas for i=1:length(spins)])
SpinCollection(spins::Vector{<:Spin}, polarizations::Vector{<:Number}, args...) = SpinCollection(spins, [polarizations for i=1:length(spins)], args...)
SpinCollection(spins::Vector{<:Spin}, polarizations; gammas=ones(length(spins))) = SpinCollection(spins, polarizations, gammas)

"""
Create a SpinCollection without explicitly creating single spins.
# Arguments
* `positions`: Vector containing the positions of all single spins.
* `polarizations`: Unit vectors defining the directions of the spins.
* `deltas=0`: Detunings.
* `gammas=1`: Decay rates.
"""
function SpinCollection(positions::Vector{<:Vector{<:Real}}, args...; deltas::Union{T,Vector{T}}=zeros(length(positions)), kwargs...) where T<:Real
    if length(deltas)==1
        SpinCollection([Spin(positions[i]; delta=deltas[1]) for i=1:length(positions)], args...; kwargs...)
    else
        SpinCollection([Spin(positions[i]; delta=deltas[i]) for i=1:length(positions)], args...; kwargs...)
    end
end


"""
A class representing a single mode in a cavity.
# Arguments
* `cutoff`: Number of included Fock states.
* `delta=0` Detuning.
* `eta=0`: Pump strength.
* `kappa=0`: Decay rate.
"""
struct CavityMode{C<:Int,T1<:Number,T2<:Number,T3<:Number} <: System
    cutoff::C
    delta::T1
    eta::T2
    kappa::T3
end
CavityMode(cutoff::Int; delta::Number=0, eta::Number=0, kappa::Number=0) = CavityMode(cutoff,delta,eta,kappa)


"""
A class representing a cavity coupled to many spins.
# Arguments
* `cavity`: A CavityMode.
* `spincollection`: A SpinCollection.
* `g`: A vector defing the coupling strengths between the i-th spin and
    the cavity mode. Alternatively a single number can be given for
    identical coupling for all spins.
"""
struct CavitySpinCollection{C<:CavityMode,S<:SpinCollection,G<:Number} <: System
    cavity::C
    spincollection::S
    g::Vector{G}
    function CavitySpinCollection{C,S,G}(cavity::C, spincollection::S, g::Vector{G}) where {C,S,G}
        @assert length(g) == length(spincollection.spins)
        new(cavity, spincollection, g)
    end
end
CavitySpinCollection(cavity::C, spincollection::S, g::Vector{G}) where {C,S,G} = CavitySpinCollection{C,S,G}(cavity, spincollection, g)
CavitySpinCollection(cavity::CavityMode, spincollection::SpinCollection, g::Real) = CavitySpinCollection(cavity, spincollection, [g for i=1:length(spincollection.spins)])


# -------------------------------------------------------------------
# 1) Single 4-level atom
# -------------------------------------------------------------------
"""
A single 4-level atom.

# Fields
- `position::Vector{<:Real}`: 3D position.
- `Eg::Real`: Ground-state energy reference.
- `Ee::Real`: Reference excited-state energy.
- `deltas::Vector{<:Real}`: Must have length = 3 (for the 3 excited sublevels: m = -1, 0, 1).
"""
struct FourLevelAtom{P<:AbstractVector{<:Real},T<:Real} <: System
    position::P
    delta::T
end

"""
Constructor
"""
function FourLevelAtom(position::Vector{<:Real};
                       delta::Real=0.0)
    return FourLevelAtom(position, delta)
end


# -------------------------------------------------------------------
# 2) FourLevelAtomCollection with complex polarizations and per-transition gammas
# -------------------------------------------------------------------
"""
A collection of 4-level atoms, each having 3 excited-state transitions. We store:

- `atoms`: Vector of `FourLevelAtom` (length N).
- `polarizations`: A 3D array of shape (3, 3, N) with element type `ComplexF64` (or Complex generally).
  * dimension 1: the transition index µ ∈ {1,2,3} (-1, 0, 1) 
  * dimension 2: the Cartesian coordinate c ∈ {1,2,3} (x, y, z) 
  * dimension 3: the atom index n ∈ {1..N}

  So `polarizations[μ, c, n]` is the c-th component of the polarization vector
  for transition μ of atom n. Each 3D vector can be complex, and is automatically 
  normalized to unit magnitude in the constructor.

- `gammas`: A 2D array of shape (3, N) of real decay rates. 
  `gammas[μ, n]` is the decay rate for transition μ in atom n.

In the constructor, we:
1) Check array sizes match (3 in dimension 1, 3 in dimension 2, N in dimension 3).
2) Check `gammas` is shape (3, N).
3) Normalize each polarization vector in place using its complex norm.
"""
struct FourLevelAtomCollection{A<:FourLevelAtom,PT<:Complex,GT<:Real} <: System
    atoms::Vector{A}
    polarizations::Array{PT,3}  # shape (3,3,N)
    gammas::Array{GT,2}         # shape (3,N)

    function FourLevelAtomCollection{A,PT,GT}(
        atoms::Vector{A}, 
        polarizations::Array{PT,3}, 
        gammas::Array{GT,2}
    ) where {A<:FourLevelAtom,PT<:Complex,GT<:Real}

        N = length(atoms)
        # Check polarizations shape
        @assert size(polarizations,1) == 3  "polarizations must have 3 transitions along dim 1."
        @assert size(polarizations,2) == 3  "polarizations must have 3 coordinates (x,y,z) along dim 2."
        @assert size(polarizations,3) == N  "polarizations must match number of atoms along dim 3."

        # Check gammas shape
        @assert size(gammas,1) == 3 "gammas must have 3 transitions along dim 1."
        @assert size(gammas,2) == N "gammas must match number of atoms along dim 2."

        # Normalize each polarization vector
        @assert N > 0 "Must have at least one atom."
        for n in 1:N
            for μ in 1:3
                @views pvec = polarizations[μ, :, n]  # This is a (3)-element slice
                normp = sqrt(sum(abs2, pvec))         # complex norm
                @assert normp > 1e-14 "Polarization vector is too close to zero to normalize."
                pvec .= pvec ./ normp
            end
        end

        new(atoms, polarizations, gammas)
    end
end

"""
Convenience constructor that accepts:
- `atoms`
- optional `polarizations` (a 3×3×N complex array)
- optional `gammas` (a 3×N real array)

If `polarizations` or `gammas` is `nothing`, we create defaults.
"""
function FourLevelAtomCollection(
    atoms::Vector{<:FourLevelAtom};
    polarizations::Union{Nothing,Array{<:Complex,3}}=nothing,
    gammas::Union{Nothing,Array{<:Real,2}}=nothing
)
    N = length(atoms)
    # If user gave no polarizations, build a default shape (3,3,N) array
    pols = if polarizations === nothing
        # e.g. fill each transition with real(1,0,0) or random complex
        tmp = Array{ComplexF64,3}(undef, 3, 3, N)
        for n in 1:N
            # Example: random complex vectors
            for μ in 1:3
                for c in 1:3
                    tmp[μ, c, n] = rand() + im*rand()
                end
            end
        end
        tmp
    else
        convert(Array{ComplexF64,3}, polarizations)
    end

    # If user gave no gammas, default to 1.0 for each transition
    gam = if gammas === nothing
        ones(3, N)  # shape (3,N)
    else
        convert(Array{Float64,2}, gammas)
    end

    return FourLevelAtomCollection{typeof(atoms[1]),ComplexF64,Float64}(atoms, pols, gam)
end

"""
Alternate constructor that also builds `atoms` from positions.
You can pass:
- positions::Vector of 3D vectors
- Eg, Ee, deltas
- optional polarizations, gammas

We create the `atoms` array, then forward to the main constructor.
"""
function FourLevelAtomCollection(
    positions::Vector{<:AbstractVector{<:Real}};
    # Eg::Real=0.0,
    # Ee::Real=1.0,
    # deltas::Vector{<:AbstractVector{<:Real}} = [[0.0,0.0,0.0]],
    deltas::Vector{<:Real} = [0.0],
    polarizations::Union{Nothing,Array{<:Complex,3}}=nothing,
    gammas::Union{Nothing,Array{<:Real,2}}=nothing
)
    if length(deltas) == 1
        atoms = [FourLevelAtom(pos; delta=deltas[1]) for pos in positions]
    else
        @assert length(deltas) == length(positions) "Size of deltas must be equal to positions size."
        atoms = [FourLevelAtom(positions[i]; delta=deltas[i]) for i in eachindex(positions)]
    end
    return FourLevelAtomCollection(
        atoms; polarizations=polarizations, gammas=gammas
    )
end


# -------------------------------------------------------------------
# 3) Cavity + FourLevelAtomCollection
# -------------------------------------------------------------------
""""
A system representing a cavity coupled to a collection of 4-level atoms.

- `cavity`: A `CavityMode` instance
- `atomcollection`: A `FourLevelAtomCollection`
- `g`: An Array{T,2}, where T<:Number (real or complex)

We have an inner constructor that enforces shape checks:
 - `size(g,1) == 3` for the 3 transitions
 - `size(g,2) == Natom` for the number of atoms
"""
struct CavityFourLevelAtomCollection{C<:CavityMode, A<:FourLevelAtomCollection, T<:Number} <: System
    cavity::C
    atomcollection::A
    g::Array{T,2}          # 2D array, can be real or complex

    function CavityFourLevelAtomCollection{C,A,T}(cavity::C, coll::A, g::Array{T,2}) where
        {C<:CavityMode, A<:FourLevelAtomCollection, T<:Number}
        
        Natom = length(coll.atoms)
        @assert size(g,1) == 3      "For a 2D g, size(g,1) must be 3 (number of transitions)."
        @assert size(g,2) == Natom "For a 2D g, size(g,2) must match number of atoms."
        
        new{C,A,T}(cavity, coll, g)
    end
end

"""
Construct a `CavityFourLevelAtomCollection` from a 2D array `g`, which
can be real or complex. The shape must be `(3, Natom)`, where Natom
is the number of atoms in `coll`.

Example usage:
    M = [0.1 0.1 0.1;
         0.2 0.2 0.2;
         0.3 0.3 0.3]   # shape (3,3)
    sys = CavityFourLevelAtomCollection(cavity, coll, M)
"""
function CavityFourLevelAtomCollection(
    cavity::CavityMode,
    coll::FourLevelAtomCollection,
    g::AbstractMatrix{<:Number}
)
    # Let T be the element type of g
    T = eltype(g)
    # Call the inner constructor with explicit {C, A, T}
    return CavityFourLevelAtomCollection{typeof(cavity), typeof(coll), T}(cavity, coll, g)
end

"""
If the user provides a single scalar for `g`, replicate it into a (3, Natom) array.
"""
function CavityFourLevelAtomCollection(
    cavity::CavityMode,
    coll::FourLevelAtomCollection,
    g::Number
)
    Natom = length(coll.atoms)
    # Fill a 3×N matrix with the scalar
    gmat = fill(g, (3, Natom))
    return CavityFourLevelAtomCollection(cavity, coll, gmat)
end