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
               V1<:Vector,V2<:Vector,V3<:Vector,
               T3<:Number} <: Field
    amplitude::T1
    module_k::T2
    angle_k::V1
    polarisation::V2
    position_0::V3
    waist_radius::T3
end
EMField(amplitude::Number, module_k::Number, angle_k::Vector,
        polarisation::Vector; position_0::Vector=[0.0,0.0,0.0],
        waist_radius::Number=0.0) = EMField(amplitude,
                                            module_k, angle_k,
                                            vec_rotate(polarisation,
                                                       angle_k[1],
                                                       angle_k[2]) ./ 
                                            LinearAlgebra.norm(polarisation),
                                            position_0,
                                            waist_radius)


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
# Arguments
* `r_vec`: vector {x,y,z} at which we calculate the field
* `E`: EM field impinging the system
"""
function gauss(r_vec::Vector, E::Field)
    # TODO: fix the dependence on φ
    x, y, z = r_vec
    x0, y0, z0 = E.position_0
    A = E.amplitude
    K = E.module_k
    θ, φ = E.angle_k
    w0 = E.waist_radius
    # reference frame of the beam
    x1, y1, z1 = [(x-x0)*cos(θ)-(z-z0)*cos(φ)*sin(θ), #- (y-y0)*sin(θ)*sin(φ),
                  (y-y0)*cos(φ) - (z-z0)*sin(φ),
                  (x-x0)*sin(θ) + (z-z0)*cos(θ)]
    polarisation = E.polarisation
    #k_vec = K.*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]
    zR = 0.5*K*w0^2
    wz1 = w0*sqrt(1.0+(z1/zR)^2)
    invRz1 = z1 / (z1^2 + zR^2)
    phiz1 = atan(z1/zR)
    return (A*w0/wz1*exp(1.0im*K*z1)*
        exp(-1.0im*phiz1)*exp(-(x1^2 + y1^2)/wz1^2)*
        exp(1.0im*K*(x1^2 + y1^2)/2.0*invRz1)).*polarisation
end


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
    pos_vec = E.position_0
    A = E.amplitude
    K = E.module_k
    θ, φ = E.angle_k
    k_vec = K*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]
    polarisation = E.polarisation
    return A*exp(1.0im*k_vec'*(r_vec-pos_vec)).*polarisation
end


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
    Ω_R = [(conj(μ[i]')*conj(E_vec[i]))/sqrt(sum(μ[i][j]*conj(μ[i][j]) for j=1:3)) for i=1:n]
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


"""
field.total_field(inc_wave_function::Function, r::Vector,
                         E::Field, S::SpinCollection, sigmam::Vector)
Function computes the total field at position r:
# Arguments
* `inc_wave_function`: Function that computes incident wave at point r.
* `r`: Position [r_x, r_y, r_z]
* `E`: Incident EM field parameters.
* `S`: Spin collection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin
# Output
* `E_{tot}`: Vector of total field at r
"""
function total_field(inc_wave_function::Function, r::Vector, E::Field,
                     S::SpinCollection, sigmam::Vector,
                     k_field::Number=2π)
    # TODO: check the constants and dimensions
    n = length(S.gammas)
    E_inc = inc_wave_function(r, E)
    C = 3.0/4.0.*S.gammas./LinearAlgebra.norm.(S.polarizations).^2   # (k_0/4.0/pi) * 3.0*pi/k_0.*γ_e./(norm.(μ).^2)
                                    # took into account factor k/(4π) in Green Tensor
    return E_inc + sum(C[i]* sigmam[i]*GreenTensor(r-S.spins[i].position, k_field)*
                   S.polarizations[i]
                   for i=1:n)
end


"""
field.scattered_field(r::Vector, S::SpinCollection, sigmam::Vector)
Function computes the total field at position r:
# Arguments
* `r`: Position [r_x, r_y, r_z]
* `S`: Spin collection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin
# Output
* `E_{sc}`: Vector of total field at r
"""
function scattered_field(r::Vector, S::SpinCollection,
                         sigmam::Vector, k_field::Number=2π)
    # TODO: k in Green's tensor and in scattered field
    n = length(S.gammas)
    C = 3.0/4.0.*S.gammas./LinearAlgebra.norm.(S.polarizations).^2   # (k_0/4.0/pi) * 3.0*pi/k_0.*γ_e./(norm.(μ).^2)
                                    # took into account factor k/(4π) in Green Tensor
    return sum(C[i]*sigmam[i]*GreenTensor(r-S.spins[i].position, k_field)*
               S.polarizations[i]
               for i=1:n)
end


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
    θ, φ = E.angle_k
    E_0 = E.amplitude
    k_vec = K*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]
    polar = E.polarisation
    r = r_lim*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]
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
    k_vec = K*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]
    polar = E.polarisation
    r = r_lim*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]
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
# Output
* `T`: transmission, a real number
* `r`: a vector of points on a hemisphere
"""
function transmission_reg(E::Field, inc_wave_function::Function,
                      S::SpinCollection, sigmam::Vector;
                      samples::Int=100, zlim::Real=1000.0,angle::Vector=[π,π])
    # TODO: debug, check the correctness of transmission computation
    L = (E.angle_k[1] >= π/2) ? S.spins[1].position[3] : S.spins[end].position[3]
    zlim = (E.angle_k[1] >= π/2) ? -zlim : zlim
    itr = range(0,samples-1,samples)
    golden_ratio = 0.5*(1.0 + sqrt(5.0))
    θ = acos.(1.0 .- (itr .+ 0.5)./samples)
    φ = 2π * itr ./ golden_ratio
    E_in2 = 0.0
    E_out2 = 0.0
    r = Vector{Vector{Float64}}(undef, samples)
    for j in 1:samples 
        r_j = zlim * [sin(θ[j])*cos(φ[j]),
                      sin(θ[j])*sin(φ[j]),
                      cos(θ[j]) + L/zlim]
        r[j] = r_j
        E_in_j = inc_wave_function(r_j, E)
        E_in2 += E_in_j'*E_in_j
        for k in 1:samples
            r_k = zlim * [sin(θ[k])*cos(φ[k]),
                          sin(θ[k])*sin(φ[k]),
                          cos(θ[k]) + L/zlim]
            E_in_k = inc_wave_function(r_k, E)
            E_out2 += abs(conj(E_in_j')*
            (total_field(inc_wave_function, r_j, E, S, sigmam)'*total_field(inc_wave_function, r_k, E, S, sigmam))*conj(E_in_k))
        end
    end
    E_in4 = abs(E_in2)^2
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
    return  [E_out2/E_in4, r]
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
* `S`: Spin collection.
* `sigmam`: Vector of steady-state values of σⱼ for each spin
# Optional arguments
* `samples = 100`: Number of points on a plane
* `zlim = 40`: length of a radius-vector
* `size = [5, 5]`: width (x) and height (y) of the plate
# Output
* `T`: transmission, a real number
* `r`: a vector of points on a plane
"""
function transmission_plane(E::Field, inc_wave_function::Function,
                      S::SpinCollection, sigmam::Vector;
                      samples::Int=100, zlim::Real=50.0,size::Vector=[5.0,5.0])
    L = (E.angle_k[1] >= π/2) ? S.spins[1].position[3] : S.spins[end].position[3]
    zlim = (E.angle_k[1] >= π/2) ? -zlim : zlim
    E_in2 = 0.0
    E_out2 = 0.0
    r = Vector{Vector{Float64}}(undef, samples)
    for j in 1:samples 
        r_j = [-0.5*size[1] + size[1]*(j-1)/(samples-1),
               -0.5*size[2] + size[2]*(j-1)/(samples-1),
               zlim + L]
        r[j] = r_j
        E_in_j = inc_wave_function(r_j, E)
        E_in2 += E_in_j'*E_in_j
        for k in 1:samples
            r_k = [-0.5*size[1] + size[1]*(k-1)/(samples-1),
                   -0.5*size[2] + size[2]*(k-1)/(samples-1),
                   zlim + L]
            E_in_k = inc_wave_function(r_k, E)
            E_out2 += abs(conj(E_in_j')*
            (total_field(inc_wave_function, r_j, E, S, sigmam)'*total_field(inc_wave_function, r_k, E, S, sigmam))*conj(E_in_k))
        end
    end
    E_in4 = abs(E_in2)^2
    # for j in 1:samples, k in 1:samples
    #     r_j = [-0.5*size[1] + size[1]*(j-1)/(samples-1),
    #            -0.5*size[2] + size[2]*(j-1)/(samples-1),
    #            zlim + L]
    #     r[j] = r_j
    #     E_in = inc_wave_function(r_j, E)
    #     E_out2 += abs(conj(E_in')*
    #     (total_field(inc_wave_function,r_j, E, S, sigmam)'*total_field(inc_wave_function,r_j, E, S, sigmam))*conj(E_in))
    #     E_in2 += abs(E_in'*E_in)^2
    # end
    return  [E_out2/E_in4, r]
end


end
