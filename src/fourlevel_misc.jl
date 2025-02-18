module fourlevel_misc

using LinearAlgebra
using ..AtomicArrays

# TODO
export polarizations_cartesian, polarizations_spherical, gammas, U_cart_to_sph, U_sph_to_cart

"""
    fourlevel_misc.polarizations_spherical()
Returns an array with dipole moments (polarizations) for a single atom in spherical coordinates.
m = 0 transition is along z axis.
Dimensions: m × α ({-1,0,+1} × {x, y, z})
"""
function polarizations_spherical()
    d_single = Array{ComplexF64,2}(undef, 3, 3)
    d_single[1, :] = - (1/sqrt(2)) .* [1.0, -im, 0.0]  # m = -1
    d_single[2, :] = [0.0, 0.0, 1.0]  # m = 0
    d_single[3, :] = (1/sqrt(2)) .* [1.0, im, 0.0]  # m = +1
    return d_single
end
function polarizations_spherical(N::Integer)
    d = Array{ComplexF64,3}(undef, 3, 3, N)
    for j = 1:N
        d[1, :, j] = - (1/sqrt(2)) .* [1.0, -im, 0.0]  # m = -1
        d[2, :, j] = [0.0, 0.0, 1.0]  # m = 0
        d[3, :, j] = (1/sqrt(2)) .* [1.0, im, 0.0]  # m = +1
    end
    return d
end

"""
    fourlevel_misc.polarizations_cartesian()
Returns an array with dipole moments (polarizations) for a single atom in cartesian coordinates.
m = 0 transition is along z axis.
- d_x = 1/√2 * (d_{+1} - d_{-1}) = (1 ,0, 0)
- d_y = -i/√2 * (d_{+1} + d_{-1}) = (0, 1, 0)
- d_z = d_0 = (0, 0, 1)

Dimensions: α × m ({x, y, z} × {-1,0,+1})
"""
function polarizations_cartesian()
    d_sph = polarizations_spherical()
    d_car = Array{ComplexF64,2}(undef, 3, 3)
    d_car[1, :] = 1/sqrt(2) * (d_sph[3, :] - d_sph[1, :])  # d_x
    d_car[2, :] = -1im/sqrt(2) * (d_sph[3, :] + d_sph[1, :])  # d_y
    d_car[3, :] = d_sph[2, :]  # d_z
    return d_car
end
function polarizations_cartesian(N::Integer)
    d = Array{ComplexF64,3}(undef, 3, 3, N)
    d_sph = polarizations_spherical()
    for j = 1:N
        d[1, :, j] = 1/sqrt(2) * (d_sph[3, :] - d_sph[1, :])  # d_x
        d[2, :, j] = -1im/sqrt(2) * (d_sph[3, :] + d_sph[1, :])  # d_y
        d[3, :, j] = d_sph[2, :]  # d_z
    end
    return d
end

"""
    fourlevel_misc.gammas(gamma_0::Real = 0.1)
Returns a vector with gammas for m =-1 ,0 ,+1 (or x,y,z) transitions. Normalized to gamma_total = gamma_{-1} + gamma_{0} + gamma_{+1} = gamma_0

For ref. see https://doi.org/10.1103/PhysRevA.47.1336
"""
function gammas(gamma_0::Real=0.1)
    return 1.0/3.0 * [gamma_0, gamma_0, gamma_0]
end

"""
    fourlevel_misc.U_sph_to_cart()
Transformation matrix from spherical basis to cartesian basis: U_{\alpha,m}.
"""
function U_sph_to_cart()
    U = [-1/sqrt(2)   0.0 1/sqrt(2);
         -1im/sqrt(2) 0.0 -1im/sqrt(2);
         0.0          1.0 0.0]
    return U
end

"""
    fourlevel_misc.U_cart_to_sph()
Transformation matrix from cartesian basis to spherical basis: U^*_{m,\alpha}.
"""
function U_cart_to_sph()
    U = U_sph_to_cart()
    return U'
end



end  # module