# Comparison with Shahmoon results. 2D lattice of spins: incident field and evolution

using CollectiveSpins
using QuantumOptics
using PyPlot
using LinearAlgebra

using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices
const mapexpect = AtomicArrays.meanfield_module.mapexpect

dag(x) = conj(transpose(x))

const NMAX = 20
const NMAX_T = 5
transmission = zeros(NMAX)
tran = zeros(NMAX)
tran2 = zeros(NMAX)
d_iter = [0.1 + 0.9*(i - 1)/(NMAX - 1) for i=1:NMAX]


"""Parameters"""

lam_0 = 1.0
c_light = 1.0
om_0 = 2*π*c_light/lam_0
Nx = 26
Ny = 26
Nz = 1  # number of arrays
k_0 = 2*π / lam_0

μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny]
γ_e = [1e-0 for i = 1:Nx*Ny]

# Incident field
om_f = om_0

E_ampl = 0.001 + 0im
E_kvec = om_f/c_light
E_pos0 = [0.0,0.0,0.0]
E_polar = [1.0, 0im, 0.0]
E_angle = [0.0*π/6, 0.0]  # {θ, φ}

em_inc_function = AtomicArrays.field_module.gauss


Threads.@threads for i = 1:NMAX
    print(i, "\n")
    # Parameters
    d = d_iter[i]

    pos = geometry.rectangle(d, d; Nx=Nx, Ny=Ny)
    # shift the origin of the array
    p_x0 = pos[1][1]
    p_xN = pos[end][1]
    p_y0 = pos[1][2]
    p_yN = pos[end][2]
    for i = 1:Nx*Ny
        pos[i][1] = pos[i][1] - 0.5*(p_x0 + p_xN)
        pos[i][2] = pos[i][2] - 0.5*(p_y0 + p_yN)
    end

    S = SpinCollection(pos,μ; gammas=γ_e)

    # Incident field parameters
    E_width = 0.3*d*sqrt(Nx*Ny)
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                    position_0 = E_pos0, waist_radius = E_width)

    """Dynamics: meanfield"""

    # E_field vector for Rabi constant computation
    E_vec = [em_inc_function(S.spins[k].position, E_inc)
             for k = 1:Nx*Ny*Nz]
    Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

    T = [0:2500.0:5000;]
    # Initial state (Bloch state)
    phi = 0.
    theta = pi/1.
    # Meanfield
    state0 = CollectiveSpins.meanfield.blochstate(phi, theta, Nx*Ny*Nz)
    tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S,
                                                                         Om_R,
                                                                         state0)

    t_ind = length(T)
    sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)


    """Transmission"""

    # First approach: Plane at infinity approach
    xlim = 0.0001
    ylim = 0.0001
    zlim = (abs(E_angle[1]) >= π/2) ? -1000.0 : 1000.0
    x_t = range(-xlim, xlim, NMAX_T)
    y_t = range(-ylim, ylim, NMAX_T)
    E_out = sum(E_polar'*AtomicArrays.field_module.total_field(em_inc_function,
                                                               [x_t[i],y_t[j],zlim],
                                                               E_inc, S, sm_mat)
                for i = 1:NMAX_T, j = 1:NMAX_T)
    E_in = sum(E_polar'*em_inc_function([x_t[i],y_t[j],zlim], E_inc)
               for i = 1:NMAX_T, j = 1:NMAX_T)
    transmission[i] = abs.(E_out)^2 ./ abs.(E_in)^2

    # Second approach: regular distribution of points on a hemisphere, power T
    zlim = 1*d*(Nx)
    tran[i], _ = AtomicArrays.field_module.transmission_reg(E_inc,
                                                            em_inc_function,
                                                            S, sm_mat;
                                                            samples=400,
                                                            zlim=zlim,
                                                            angle=[π, π]);

    # Third approach: random distribution of points on a hemisphere, T with polarisation
    zlim = 1*d*(Nx)
    tran2[i], _ = AtomicArrays.field_module.transmission_rand(E_inc,
                                                              em_inc_function,
                                                              S, sm_mat;
                                                              samples=400,
                                                              zlim=zlim,
                                                              angle=[π, π]);

end

PyPlot.figure(figsize=(6,4))
PyPlot.plot(d_iter, transmission, "-o")
PyPlot.plot(d_iter, tran, "-o")
PyPlot.plot(d_iter, tran2, "-o")
PyPlot.xlabel(L"a/\lambda_0")
PyPlot.ylabel("Transmission")



#write("../Data/test.bin", I)
