module field

using ..AtomicArrays
import LinearAlgebra


"""
Abstract base class for all fields defined in this library.
Currently there are the following concrete field profiles:
* Gauss
* Plane
"""
abstract type Field end


"""
A class representing an EM field impinging the system.
# Arguments
* `amplitude`: Field amplitude.
* `module_k`: Module of a wave vector.
* `angle_k`: Angle between Z axis and X axis, and Y axis, [θ (z-x), φ (z-y)]
* `polarisation`: Polarisation of the field: [eₓ, e_y, e_z].
* `position_0=[0,0,0]`: Position of '0'.
* `waist_radius=0`: Waist radius for Gaussian beam
"""
struct EMField{T1<:Number,T2<:Number,
               V1<:Vector,V2<:Vector,V3<:Vector,V4<:Vector,
               T3<:Number} <: Field
    amplitude::T1
    module_k::T2
    angle_k::V1
    polarisation::V2
    position_0::V3
    k_vector::V4
    waist_radius::T3
end
function EMField(amplitude::Number, module_k::Number,
                 angle_k::AbstractVector{<:Real},
                 polarisation::AbstractVector{<:Number};
                 position_0::AbstractVector{<:Real} = [0.0, 0.0, 0.0],
                 waist_radius::Number = 0.1)
                               
    normed_pol = polarisation ./ LinearAlgebra.norm(polarisation)
    rotated_pol = vec_rotate(normed_pol, angle_k[1], angle_k[2])
    k_vec = module_k * [sin(angle_k[1]),
                        cos(angle_k[1]) * sin(angle_k[2]),
                        cos(angle_k[1]) * cos(angle_k[2])]
                               
    return EMField(amplitude, module_k, angle_k, rotated_pol,
                   position_0, k_vec, waist_radius)
end



"""
field.vec_rotate(vec::Vector, θ::Number, φ::Number)
* Rotate vector by an angle θ about y-axis and by an angle φ about x-axis
* θ is in a x-y plane
* φ is in a y-z plane
* At (θ, φ) = (0, 0), a vector = (0, 0, 1)
                 x|
                  |
                  |  /
                  | / +θ
                  |/_)________
                 / +φ         z
                /
               /
              /y
"""
function vec_rotate(vec::Vector, θ::Number, φ::Number)
    R_y = [cos(θ)  0 sin(θ);
           0       1      0;
           -sin(θ) 0 cos(θ)]
    R_x = [1       0      0;
           0  cos(φ) sin(φ);
           0 -sin(φ) cos(φ)]
    return R_x*R_y*vec
end


"""
field.gauss(r_vec::Vector, E::Field)
Impinging field propagating along Z axis at angle φ i Y-Z plane and angle
θ in X-Z plane.

Compute a Gaussian beam field at `r_vec` from field parameters `E`.
Assumes propagation at angles θ and φ, centered at `position_0`.
# Arguments
* `r_vec`: vector {x,y,z} at which we calculate the field
* `E`: EM field impinging the system
"""
function gauss(r_vec::Vector, E::Field)
    x, y, z = r_vec
    x0, y0, z0 = E.position_0
    A = E.amplitude
    K = E.module_k
    θ, φ = E.angle_k
    w0 = E.waist_radius
    polarization = E.polarisation

    # Local beam coordinates
    x1 = (x - x0) * cos(θ) - (z - z0) * cos(φ) * sin(θ)
    y1 = (y - y0) * cos(φ) - (z - z0) * sin(φ)
    z1 = (x - x0) * sin(θ) + (z - z0) * cos(θ)

    zR = 0.5 * K * w0^2
    wz = w0 * sqrt(1 + (z1 / zR)^2)
    Rz_inv = z1 / (z1^2 + zR^2)
    phi_z = atan(z1 / zR)

    envelope = A * (w0 / wz) * exp(-((x1^2 + y1^2) / wz^2))
    phase = exp(1im * K * z1) * exp(-1im * phi_z) * exp(1im * K * (x1^2 + y1^2) / 2 * Rz_inv)

    return envelope * phase .* polarization
end
gauss(r_vecs::Vector{<:AbstractVector}, E::Field) = gauss.(r_vecs, Ref(E))



"""
field.plane(r_vec::Vector, k_vec::Vector,  A::ComplexF64, pos_vec::Vector,
                            polarisation::Vector)
Impinging field propagating along k_vec
# Arguments
* `r_vec`: vector {x,y,z} at which we calculate the field
* `k_vec`: vector {k_x, k_y, k_z} -- k-vector
* `A`: field amplitude
* `pos_vec`: vector {x,y,z} at which the field impidges the system
* `polarisation`: unit vector {d_x, d_y, d_z}, field polarisation
"""
function plane(r_vec::Vector, E::Field)
    A = E.amplitude
    pos0 = E.position_0
    pol = E.polarisation

    k_vec = E.k_vector
    phase = exp(1im * LinearAlgebra.dot(k_vec, r_vec .- pos0))

    return A * phase .* pol
end
plane(r_vecs::Vector{<:AbstractVector}, E::Field) = plane.(r_vecs, Ref(E))



"""
field.rabi(E_vec::Vector, polarisation::Vector)
Rabi frequency of atoms interacting with the incident field:
```math
Ω_R = d E^*/ħ
```
# Arguments
* `E_vec`: vector {{E_x,E_y,E_z},...} -- incident field at the atoms coordinates
* `polarisation`: {{d_x,d_y,d_z},...} -- atomic polarisations
# Output
* `Ω_R`: complex vector
"""
function rabi(E_vec::Vector, polarisation::Vector)
    μ = polarisation
    n = length(μ)
    Ω_R = [(conj(μ[i]')*conj(E_vec[i]))/LinearAlgebra.norm(μ[i]) for i=1:n]
    return Ω_R
end
function rabi(E_vec::Vector, atom_coll::SpinCollection)
    μ = atom_coll.polarizations
    n = length(atom_coll.spins)
    Ω_R = [(conj(μ[j]')*conj(E_vec[j])) for j=1:n]
    return Ω_R
end
function rabi(E::Field, field_function::Function,
              atom_coll::SpinCollection)
    atoms = atom_coll.spins
    n = length(atoms)
    μ = atom_coll.polarizations
    Ω_R = [conj(μ[j]')*conj(field_function(atoms[j].position, E)) 
           for j=1:n]
    return Ω_R
end
function rabi(E_vec::Vector, atom_coll::FourLevelAtomCollection)
    μ = atom_coll.polarizations
    n = length(atom_coll.atoms)
    Ω_R = [(conj(μ[i, :, j]')*conj(E_vec[j])) for i=1:3, j=1:n]
    return Ω_R
end
function rabi(E::Field, field_function::Function,
              atom_coll::FourLevelAtomCollection)
    atoms = atom_coll.atoms
    n = length(atoms)
    μ = atom_coll.polarizations
    Ω_R = [conj(μ[i, :, j]')*conj(field_function(atoms[j].position, E)) 
           for i=1:3, j=1:n]
    return Ω_R
end


function intensity(E::AbstractVector)
    return 0.5*real(E' * E)
end


"""
    field.scattered_field(r::Vector, S::SpinCollection, σ, k_field::Real=2π)
Compute the field scattered by a collection of atoms/spins at point `r`.
Supports both `SpinCollection` and `FourLevelAtomCollection`.
# Arguments
* `r`: Position [r_x, r_y, r_z]
* `S`: Spin collection.
* `σ`: Vector of steady-state values of σⱼ for each spin
* `k_field`: 2π
# Output
* `E_{sc}`: Vector of total field at r
"""
function scattered_field(r::AbstractVector, collection, σ, k_field::Real=2π)
    _scattered_field(r, collection, σ, k_field)
end
scattered_field(r_vecs::Vector{<:AbstractVector},
                collection, σ, k::Real=2π) = scattered_field.(
                r_vecs, Ref(collection),Ref(σ), Ref(k))


"""
field.total_field(inc_wave_function::Function, r::Vector,
                         E::Field, S::SpinCollection, σ::Vector)
Compute the total field at point `r` as a sum of incident and scattered fields.
# Arguments
* `inc_wave_function`: Function that computes incident wave at point r.
* `r`: Position [r_x, r_y, r_z]
* `E`: Incident EM field parameters.
* `S`: Spin collection.
* `σ`: Vector of steady-state values of σⱼ for each spin or Matrix for 4level
# Output
* `E_{tot}`: Vector of total field at r
"""
function total_field(inc_wave_function::Function, r::AbstractVector,
                     E::Field, collection, σ, k::Real=2π)
    E_inc = inc_wave_function(r, E)
    E_scat = scattered_field(r, collection, σ, k)
    return E_inc + E_scat
end
total_field(inc_wave_function::Function, r_vecs::Vector{<:AbstractVector},
            E::Field, collection, σ, k::Real=2π) = total_field.(
                Ref(inc_wave_function), r_vecs, Ref(E), Ref(collection),
                Ref(σ), Ref(k))


"""
field.forward_scattering(r_lim::Number, E::Field, S::SpinCollection,
                                sigmam::Vector)
Computes forward scattering in k_vec of E_in direction.
Note that forward scattering is normalized.
# Arguments
* `r_lim`: Distance at which we observe scattering (must be >> λ)
* `E`: Incident EM field parameters.
* `S`: Spin collection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin
# Output
* `σ_{tot}`, Float64: total scattering (due to optical theorem) from the system
             (must be a positive number)
"""
function forward_scattering(r_lim::Number, E::Field,
                            S::SpinCollection,
                            sigmam::Vector)
    # TODO: k in Green's tensor and in scattered field
    K = E.module_k
    E_0 = E.amplitude
    k_vec = E.k_vector
    polar = E.polarisation
    r = k_vec / K * r_lim
    E_sc = scattered_field(r, S, sigmam, K)
    return (4π/K * imag(r_lim/E_0/exp(im*k_vec'*(r-E.position_0)) .* polar'*E_sc))[1]
end


"""
field.forward_scattering_1particle(r_lim::Number, E::Field,
                                            γ::Number; Δ::Number = 0.0)
Computes forward scattering in k_vec of E_in direction of 1 atom.
Note that forward scattering is normalized.
# Arguments
* `r_lim`: Distance at which we observe scattering (must be >> λ)
* `E`: Incident EM field parameters.
* `γ`: spontaneous decay rate of an atom.
* `Δ`: detuning
# Output
* `σ_{tot}`, Float64: total scattering (due to optical theorem) from the system
             (must be a positive number)
"""
function forward_scattering_1particle(r_lim::Number, E::Field,
                                      γ::Number; Δ::Number = 0.0)
    # TODO: k in Green's tensor and in scattered field
    K = E.module_k
    θ, φ = E.angle_k
    E_0 = E.amplitude
    k_vec = E.k_vector
    polar = E.polarisation
    r = k_vec / K * r_lim
    # atom
    r_0 = [0., 0., 0.]  # coordinates of an atom
    Ω_R = rabi([plane(r_0, E)], [polar])[1]
    σ = 2.0im * (γ - 2.0im*Δ)*conj(Ω_R) / (γ^2 + 4*Δ^2 + 8*abs(Ω_R)^2)
    C = 3.0/4.0.*γ   # (k_0/4.0/pi) * 3.0*pi/k_0.*γ_e./(norm.(μ).^2)
                     # took into account factor k/(4π) in Green Tensor
    # Scattered field
    E_sc = C*σ*GreenTensor(r - r_0, K)*polar
    return (4π/K * imag(r_lim/E_0/exp(im*k_vec'*(r-E.position_0)) .* polar'*E_sc))[1]
end


"""
field.objective(scatt_dir_1, scatt_dir_2)
Computes the objective function for optimisation problem:
# Arguments
* `scatt_dir_1`: Number or array of any dimension of total scattering for direction 1.
* `scatt_dir_2`: Number or array of any dimension of total scattering for direction 2.
# Output
* `M`: Number or array
"""
function objective(scatt_dir_1, scatt_dir_2)
    s1 = scatt_dir_1
    s2 = scatt_dir_2
    Norm = maximum([s1 s2])
    M = (1/1 * max.(s1, s2) .* abs.((s1 .- s2) ./ (s1 .+ s2))).^1
    return M
end


"""
field.transmission_rand(E::Field, inc_wave_function::Function,
                          S::SpinCollection, sigmam::Vector)
Function computes transmission in +z or -z direction in a hemisphere limited by
angles [dθ, dφ], points are located randomly  with uniform distribution on
a hemisphere.
T = |∫e E_{tot}dS / ∫e E_{in}|^2
# Arguments
* `E`: Incident EM field parameters.
* `inc_wave_function`: Function that computes incident wave at point r.
* `S`: Spin collection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin
# Optional arguments
* `samples = 100`: Number of points on a hemisphere
* `zlim = 10`: length of a radius-vector
* `angle = [π, π]`: [dθ, dφ] -- dθ in x-z plane, dφ in y-z plane, [0,0] is along z
# Output
* `T`: transmission, a real number
* `r`: a vector of points on a hemisphere
"""
function transmission_rand(E::Field, inc_wave_function::Function,
                      S::SpinCollection, sigmam::Vector;
                      samples::Int=100, zlim::Real=1000.0,angle::Vector=[π,π])
    # TODO: debug, check the correctness of transmission computation
    L = (E.angle_k[1] >= π/2) ? S.spins[1].position[3] : S.spins[end].position[3]
    dθ, dφ = angle
    zlim = (E.angle_k[1] >= π/2) ? -zlim : zlim
    θ = asin.((sin(dθ/2) - sin(-dθ/2)) * (rand(Float64, samples) .- 0.5))
    φ = dφ * (rand(Float64, samples) .- 0.5)
    E_in = 0.0im
    E_out = 0.0im
    r = Vector{Vector{Float64}}(undef, samples)
    for j in 1:samples
        r_j = zlim * [sin(θ[j]),
                      cos(θ[j])*sin(φ[j]),
                      cos(θ[j])*cos(φ[j]) + L/zlim]
        r[j] = r_j
        E_in += E.polarisation'*inc_wave_function(r_j, E)
        E_out += E.polarisation'*total_field(inc_wave_function, r_j, E, S, sigmam)
    end
    return [abs.(E_out./E_in).^2, r]
end


"""
field.transmission_reg(E::Field, inc_wave_function::Function,
                          S::SpinCollection, sigmam::Vector)
Function computes transmission in +z or -z direction on a hemisphere at the
distance zlim. Points are distributed on a hemisphere by Fibonacci sequence.
T = Σ|E_{tot} * (E_{in})'^*|^2 / Σ|E_{in} * (E_{in})'^*|^2
# Arguments
* `E`: Incident EM field parameters.
* `inc_wave_function`: Function that computes incident wave at point r.
* `S`: Spin collection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin
# Optional arguments
* `samples = 100`: Number of points on a hemisphere
* `zlim = 10`: length of a radius-vector
* `angle = [π, π]`: rudiment angle, just for backward compatibility

Returns:
* `T`: transmission coefficient (real number)
* `r`: sampling points on the hemisphere (Vector of 3D positions)
"""
function transmission_reg(E::Field, inc_wave_function::Function,
                          S::Union{SpinCollection, FourLevelAtomCollection}, sigmam::AbstractArray;
                          samples::Int=50, zlim::Real=1000.0, angle::Vector=[π, π])

    # Extract front/back position L depending on wave vector
    L = get_L_position(E, S)
    zlim = (E.angle_k[1] >= π/2) ? -zlim : zlim

    # Generate hemisphere points via Fibonacci method
    θ, φ = fibonacci_angles(samples)
    r = [zlim * [sin(θ[j]) * cos(φ[j]),
                 sin(θ[j]) * sin(φ[j]),
                 cos(θ[j]) + L / zlim] for j in 1:samples]

    # Precompute incoming and total fields
    E_in = inc_wave_function(r, E)  # vector{vector}
    total_fields = total_field(inc_wave_function, r, E, S, sigmam)

    # Compute normalization
    E_in2 = sum(2*intensity.(E_in))

    # Compute transmission
    E_out2 = 0.0
    for j in 1:samples
        E_in_j = E_in[j]
        tf_j = total_fields[j]
        for k in 1:samples
            tf_k = total_fields[k]
            E_in_k = E_in[k]
            E_out2 += abs(conj(E_in_j') * (tf_j' * tf_k) * conj(E_in_k))
        end
    end

    E_in4 = abs(E_in2)^2
    return [E_out2 / E_in4, r]
        # alternative way
    # for j in 1:samples
    #     r_j = zlim * [sin(θ[j])*cos(φ[j]),
    #                   sin(θ[j])*sin(φ[j]),
    #                   cos(θ[j]) + L/zlim]
    #     r[j] = r_j
    #     E_in = inc_wave_function(r_j, E)
    #     E_out2 += abs(E_in'*
    #     total_field(inc_wave_function,r_j, E, S, sigmam)*total_field(inc_wave_function,r_j, E, S, sigmam)'*E_in)
    #     E_in2 += abs(E_in'*E_in)^2
    # end
end

"""
field.transmission_plane(E::Field, inc_wave_function::Function,
                          S::SpinCollection, sigmam::Vector)
Function computes transmission in +z or -z direction on a square plate at the
distance zlim.
T = Σ|E_{tot} * (E_{in})'^*|^2 / Σ|E_{in} * (E_{in})'^*|^2
# Arguments
* `E`: Incident EM field parameters.
* `inc_wave_function`: Function that computes incident wave at point r.
* `S`: Spin collection or FourLevelAtomCollection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin, or Matrix
# Optional arguments
* `samples = 100`: Number of points on a plane
* `zlim = 40`: length of a radius-vector
* `size = [5, 5]`: width (x) and height (y) of the plate
# Output
* `T`: transmission coefficient (real number)
* `r`: sampling points on the plane (Vector of 3D positions)
"""
function transmission_plane(E::Field, inc_wave_function::Function,
                            S::Union{SpinCollection, FourLevelAtomCollection}, sigmam::AbstractArray;
                            samples::Int=100, zlim::Real=50.0, size::Vector=[5.0, 5.0])

    # Front or back depending on propagation direction
    L = get_L_position(E, S)
    zlim = (E.angle_k[1] >= π/2) ? -zlim : zlim

    # Generate uniformly spaced square grid of points
    r = [ [
        -0.5 * size[1] + size[1] * (j - 1) / (samples - 1),
        -0.5 * size[2] + size[2] * (j - 1) / (samples - 1),
        zlim + L
    ] for j in 1:samples ]

    # Precompute incoming fields and total fields
    E_in = inc_wave_function(r, E)  # vector{vector}
    total_fields = total_field(inc_wave_function, r, E, S, sigmam)

    # Normalize factor (denominator)
    E_in2 = sum(2*intensity.(E_in))

    # Transmission numerator
    E_out2 = 0.0
    for j in 1:samples
        E_in_j = E_in[j]
        tf_j = total_fields[j]
        for k in 1:samples
            tf_k = total_fields[k]
            E_in_k = E_in[k]
            E_out2 += abs(conj(E_in_j') * (tf_j' * tf_k) * conj(E_in_k))
        end
    end

    E_in4 = abs(E_in2)^2
    return [E_out2 / E_in4, r]
end

"""
    field.transmission_reflection(E::EMField,
                            collection::Union{SpinCollection, FourLevelAtomCollection},
                            σ::AbstractArray;
                            beam::Symbol = :plane,
                            surface::Symbol = :hemisphere,
                            polarization = nothing,
                            samples::Int = 50,
                            zlim::Real = 1000.0,
                            size::Vector = [5.0, 5.0],
                            return_positions::Bool = false)

Computes the transmission and reflection coefficients by integrating the output power 
over a specified surface in forward and backward directions.

# Arguments
- `E::EMField`: Incident electromagnetic field (plane wave or Gaussian beam).
- `collection`: Atomic collection (e.g. `FourLevelAtomCollection`) describing positions, detunings, and polarizations.
- `σ`: Array of expectation values for the system (e.g., `σm` or `σmm`).

# Keyword Arguments
- `beam::Symbol`: `:plane` or `:gauss`. Defines the beam profile used for the incident field.
- `surface::Symbol`: `:hemisphere` or `:plane`. Defines the surface over which power is integrated.
- `polarization`: Optional polarization vector (default is taken from `E.polarisation`).
- `samples::Int`: Number of sample points for integration (angular or grid).
- `zlim::Real`: Distance from the atomic plane to the integration surface.
- `size::Vector`: `[x, y]` dimensions for the planar integration surface (only used when `surface = :plane`).
- `return_positions::Bool`: If true, also returns vectors of sampled points `r_fwd` and `r_bwd`.

# Returns
- `T`: Transmission coefficient.
- `R`: Reflection coefficient.
- Optionally: `r_fwd`, `r_bwd` if `return_positions=true`.

# Example
```julia
T, R = transmission_reflection(E, coll, σ; beam=:gauss, surface=:hemisphere, samples=100)
T, R, rf, rb = transmission_reflection(E, coll, σ; return_positions=true)
"""
function transmission_reflection(E::AtomicArrays.field.EMField,
                         collection::Union{SpinCollection, FourLevelAtomCollection},
                         σ::AbstractArray;
                         beam::Symbol      = :plane,
                         surface::Symbol   = :hemisphere,
                         polarization      = nothing,
                         samples::Int      = 50,
                         zlim::Real        = 1_000.0,
                         size::Vector      = [5.0,5.0],
                         return_positions::Bool = false)
                                
    # ---------- 1.  Set helpers ----------
    inc_wave = beam === :plane ? plane : gauss
    pol      = polarization === nothing ? E.polarisation :
                polarization ./ LinearAlgebra.norm(polarization) # ensure unit‑norm
    k̂        = E.k_vector / LinearAlgebra.norm(E.k_vector) # propagation direction
                                
    # analytic incident intensity  (c = ε₀ = 1)
    intensity(v) = abs2( LinearAlgebra.dot(pol, v) )/2
    I_inc = abs2(E.amplitude)/2
    P_inc = (π*E.waist_radius^2/2) * I_inc
                                
    # ---------- 2.  Make integration grids ----------
    if surface === :hemisphere
        θ, φ = fibonacci_angles(samples)
        # forward (+k̂) & backward (−k̂) hemispheres
        r_fwd =  Vector{Vector{Float64}}(undef, samples)
        r_bwd =  Vector{Vector{Float64}}(undef, samples)
        for j in eachindex(θ)
            rot_vec = [sin(θ[j])*cos(φ[j]),
                       sin(θ[j])*sin(φ[j]),
                       cos(θ[j])]
            # build orthonormal basis with k̂ as new +z
            # fast Gram–Schmidt
            zax = k̂
            xax = abs(zax[3]) < 0.9 ? LinearAlgebra.normalize(
                                        LinearAlgebra.cross(zax,[0,0,1])) :
                                      LinearAlgebra.normalize(
                                        LinearAlgebra.cross(zax,[0,1,0]))
            yax = LinearAlgebra.cross(zax,xax)
            R   = hcat(xax,yax,zax)          # 3×3 rotation matrix
            dir = R*rot_vec
            r_fwd[j] =  zlim* dir           # along +k̂
            r_bwd[j] = -zlim* dir           # along −k̂
        end
        ΔΩ = 2π/samples                     # equal‑area weight
        area_factor = zlim^2                # R² term in ∫ I dΩ
    else  # :plane
        # square grid centred on optical axis
        x = range(-0.5*size[1], stop=0.5*size[1], length=samples)
        y = range(-0.5*size[2], stop=0.5*size[2], length=samples)
        r_fwd = [ [xx,yy, zlim] for yy in y, xx in x ] |> vec
        r_bwd = [ [xx,yy,-zlim] for yy in y, xx in x ] |> vec
        dA   = (size[1]/(samples-1))*(size[2]/(samples-1))
    end
                                
    # ---------- 3.  Pre‑compute fields ----------
    # (broadcasting works because inc_wave / total_field already have dot‑methods)
    E_in_fwd  = inc_wave(r_fwd, E)
    E_sc_fwd = scattered_field(r_fwd, collection, σ)
    E_tot_fwd = E_in_fwd .+ E_sc_fwd
    E_in_bwd  = inc_wave(r_bwd, E)
    E_sc_bwd = scattered_field(r_bwd, collection, σ)
    E_tot_bwd = E_in_bwd .+ E_sc_bwd
                                
    # ---------- 4.  Integrate power ----------
    if surface === :hemisphere
        # projected |E|² sum  (∝ intensity); pol·E selects co‑polarised power
        P_fwd = ΔΩ*area_factor * sum( intensity.(E_tot_fwd) )
        P_bwd = ΔΩ*area_factor * sum( intensity.(E_sc_bwd) )
        P_inc = beam === :plane ? I_inc*2*pi*zlim^2 : P_inc # analytic = I_inc*π zlim²
        # P_inc = ΔΩ*area_factor * sum( intensity.(E_in_fwd) )
    else  # plane
        P_fwd = dA * sum( intensity.(E_tot_fwd) )
        P_bwd = dA * sum( intensity.(E_sc_bwd) )
        P_inc = I_inc * size[1] * size[2]*2  # analytic = I_inc*size[1]*size[2]
        # P_inc = dA * sum( intensity.(E_in_fwd) )
    end
                                
    # ---------- 5.  Coefficients ----------
    T = P_fwd / P_inc
    R = P_bwd / P_inc
    return return_positions ? (T, R, r_fwd, r_bwd) : (T, R)
end

# ---------- Helper functions ----------

# Extract front or back Z-position from collection
function get_L_position(E::Field, S::SpinCollection)
    return (E.angle_k[1] >= π/2) ? S.spins[1].position[3] : S.spins[end].position[3]
end

function get_L_position(E::Field, S::FourLevelAtomCollection)
    return (E.angle_k[1] >= π/2) ? S.atoms[1].position[3] : S.atoms[end].position[3]
end

# Generate θ, φ using Fibonacci method
function fibonacci_angles(samples::Int)
    golden_ratio = 0.5 * (1.0 + sqrt(5.0))
    itr = range(0, samples - 1, length=samples)
    θ = acos.(1.0 .- (itr .+ 0.5) ./ samples)
    φ = 2π .* itr ./ golden_ratio
    return θ, φ
end

function _scattered_field(r::AbstractVector, S::SpinCollection,
                          sigmam::Vector, k::Real)
    n = length(S.spins)
    C = 3.0/4.0 .* S.gammas ./ LinearAlgebra.norm.(S.polarizations).^2
    return sum(C[i] * sigmam[i] *
               GreenTensor(r .- S.spins[i].position, k) *
               S.polarizations[i] for i in 1:n)
end

function _scattered_field(r::AbstractVector, A::FourLevelAtomCollection,
                          sigmas_m::AbstractMatrix, k::Real)
    M, N = size(A.gammas)
    C = 3.0/4.0 .* A.gammas
    return sum(C[m, n] * sigmas_m[m, n] *
               GreenTensor(r .- A.atoms[n].position, k) *
               A.polarizations[m, :, n] for m in 1:M, n in 1:N)
end


end  # module field
