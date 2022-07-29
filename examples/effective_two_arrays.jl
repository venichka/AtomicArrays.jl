#  Effective model: two arrays are represented as two dipoles with renormalized 
# frequecies and decay rates

using CollectiveSpins
using QuantumOptics
using FFTW
using PyPlot
#pygui(true)
PyPlot.svg(true)

using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const collective_shift_1array = AtomicArrays.effective_interaction_module.collective_shift_1array
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices
const mapexpect = AtomicArrays.meanfield_module.mapexpect

dag(x) = conj(transpose(x))

const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/two_arrays"
const NMAX = 100
const NMAX_T = 5
const DIRECTION = "left"

"""Parameters"""
c_light = 1.0
lam_0 = 1.0
k_0 = 2*π / lam_0
om_0 = 2.0*pi*c_light / lam_0

Nx = 10
Ny = 10
Nz = 2  # number of arrays
M = 1 # Number of excitations
d = 0.1888
delt = 0.0
Delt = 0.055
L = 0.7
#d = 0.48
#delt = 0.256
#Delt = 0.0
#L = 0.338
μ = [(i < 0) ? [1.0, 0, 0.0] : [1.0, 0.0im, 0.0] for i = 1:2]
γ_e = [1e-2 for i = 1:2]
δ_S = [(i < Nx*Ny+1) ? 0.0 : Delt for i = 1:2]


"""Calculate the collective shift depending on the lattice constant"""

Omega_1, Gamma_1 = collective_shift_1array(d, 0.0, 0.0, Nx)
Omega_2, Gamma_2 = collective_shift_1array(d, Delt, delt, Nx)

pos = [[0,0,0], [0,0,L]]
S_1 = Spin(pos[1], delta=Omega_1)
S_2 = Spin(pos[2], delta=Omega_2 + Delt)
S = SpinCollection([S_1, S_2], [[1,0,0],[1,0,0]], 
                    gammas=[Gamma_1, Gamma_2])


# Incident field parameters

om_f = om_0

E_ampl = 1.67e-2 - 0.0im#2.5e-1 + 0im#0.001 + 0im
E_kvec = om_f/c_light
E_width = 0.3*d*sqrt(Nx*Ny)
if (DIRECTION == "right")
    E_pos0 = [0.0,0.0,0.0]
    E_polar = [1.0, 0im, 0.0]
    E_angle = [0.0, 0.0]  # {θ, φ}
elseif (DIRECTION == "left")
    E_pos0 = [0.0,0.0,1.0*L]
    E_polar = [-1.0, 0im, 0.0]
    E_angle = [π, 0.0]  # {θ, φ}
else
    println("DIRECTION wasn't specified")
end


incident_field = EMField(E_ampl, E_kvec, E_angle, E_polar;
                     position_0 = E_pos0, waist_radius = E_width)
#em_inc_function = AtomicArrays.field_module.gauss
em_inc_function = AtomicArrays.field_module.plane


"""Impinging field"""

x = range(-10, 10, NMAX)
y = 0.0
z = range(-10., 10., NMAX)
e_field = Matrix{ComplexF64}(undef, length(x), length(z))
for i = 1:length(x)
    for j = 1:length(z)
        e_field[i,j] = em_inc_function([x[i],y,z[j]], incident_field)[1]
    end
end


fig_0 = PyPlot.figure(figsize=(7,3))
PyPlot.contourf(z, x, real(e_field), 30)
for p in pos
    PyPlot.plot(p[3],p[1],"o",color="w",ms=2)
end
PyPlot.xlabel("z")
PyPlot.ylabel("x")
PyPlot.colorbar(label="Amplitude")
PyPlot.tight_layout()
display(fig_0)


"""Dynamics: meanfield"""

# E_field vector for Rabi constant computation
E_vec = [em_inc_function(S.spins[k].position,incident_field)
         for k = 1:2]

Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

fig_1, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(abs.(Om_R))
axs[1].plot(abs.(Om_R), "o")
axs[1].set_title(L"|\Omega_R|")
axs[2].plot(real(Om_R),"-o")
axs[2].plot(imag(Om_R), "-o")
axs[2].set_title(L"\Re(\Omega_R), \Im(\Omega_R)")
display(fig_1)

const T = [0:10.0:10000;]
# Initial state (Bloch state)
const phi = 0.
const theta = pi/1.
# Meanfield
state0 = CollectiveSpins.meanfield.blochstate(phi, theta, 2)
@time tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0)


# Expectation values

sx_mf = sum([mapexpect(CollectiveSpins.meanfield.sx, state_mf_t, i) for i=1:2]) ./ (2)
sy_mf = sum([mapexpect(CollectiveSpins.meanfield.sy, state_mf_t, i) for i=1:2]) ./ (2)
sz_mf = sum([mapexpect(CollectiveSpins.meanfield.sz, state_mf_t, i) for i=1:2]) ./ (2)
sm_mf = 0.5*(sx_mf - 1im.*sy_mf)
sp_mf = 0.5*(sx_mf + 1im.*sy_mf)

println("[", sx_mf[end], ", ", sy_mf[end], ", ", sz_mf[end])

fig = PyPlot.figure("Average values",figsize=(5,5))
PyPlot.subplot(311)
PyPlot.plot(T, sx_mf, label="mean field")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_x \rangle")
PyPlot.legend()

subplot(312)
PyPlot.plot(T, sy_mf, label="mean field")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_y \rangle")
PyPlot.legend()

PyPlot.subplot(313)
PyPlot.plot(T, sz_mf, label="mean field")
PyPlot.xlabel("Time")
PyPlot.ylabel(L"\langle \sigma_z \rangle")
PyPlot.legend()

t_ind = length(T)
sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)

PyPlot.figure()
PyPlot.plot(abs.(sm_mat),"o")
PyPlot.title(L"\sigma_-(\infty)")

display(fig)


"""Computing the field distribution"""

x = range(-2.5, 2.5, NMAX) #range(-Nx*d, Nx*d, NMAX)
y = 0.0
z = range(-2.5, 2.5, NMAX)
t_ind = length(T)
I = zeros(length(x), length(z))
E_tot = zeros(length(x), length(z))
E_sc = zeros(length(x), length(z))
Threads.@threads for i=1:length(x)
    for j=1:length(z)
        I[i,j] = (norm(AtomicArrays.field_module.total_field(em_inc_function,
                                                             [x[i],y,z[j]],
                                                             incident_field,
                                                             S, sm_mat))^2 /
                    abs(E_ampl)^2)
        E_tot[i,j] = real(AtomicArrays.field_module.total_field(em_inc_function,
                                                             [x[i],y,z[j]],
                                                             incident_field,
                                                             S, sm_mat)[1])/real(E_ampl)
        E_sc[i,j] = real(AtomicArrays.field_module.scattered_field(
                                                             [x[i],y,z[j]],
                                                             S, sm_mat)[1])/real(E_ampl)
        print(i, "  ", j,"\n")
    end
end

# Plot
I_arr = [I[(i - 1)÷length(z) + 1, (i - 1)%length(z)+1]
         for i = 1:length(x)*length(z)]
I_arr_sorted = sort(I_arr, rev=true)#[Nx*Ny*Nz:end]  # sorted and truncated Intensity
I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
I_arr_max = findmax(I_arr)[1]

fig_e_2, axs = plt.subplots(ncols=3, nrows=1, figsize=(10, 3),
                          constrained_layout=true)
levels_f = -1.2:1e-2:1.2
levels_I = 0:1e-2:2
c1 = axs[1].contourf(z, x, I, 30, levels=levels_I)
axs[1].set_xlabel("z")
axs[1].set_ylabel("x")
fig_e_2.colorbar(c1, label=L"|E_{tot}|^2 / |E_0|^2")
c2 = axs[2].contourf(z, x, E_sc, 30, levels=levels_f)
axs[2].set_xlabel("z")
fig_e_2.colorbar(c2, label=L"Re(E_{sc} / E_0)")
c3 = axs[3].contourf(z, x, E_tot, 30, levels=levels_f)
axs[3].set_xlabel("z")
fig_e_2.colorbar(c3, label=L"Re(E_{tot} / E_0)")
for p in pos
    axs[1].plot(p[3],p[1],"o",color="w",ms=4)
    axs[2].plot(p[3],p[1],"o",color="w",ms=4)
    axs[3].plot(p[3],p[1],"o",color="w",ms=4)
end


xf = 0.0
yf = 0.1
zf = range(-5, 5, 2*NMAX)
e_field_x = zeros(ComplexF64, length(zf))
e_tot_x = zeros(ComplexF64, length(zf))
for i = 1:length(zf)
    e_field_x[i] = (em_inc_function([xf,yf,zf[i]], incident_field)[1])
    e_tot_x[i] = (AtomicArrays.field_module.total_field(em_inc_function,
                                                        [xf,yf,zf[i]],
                                                        incident_field,
                                                        S, sm_mat)[1])
end

fig_e_5, axs = plt.subplots(ncols=1, nrows=3, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(zf, real(e_field_x))
axs[1].plot(zf, real(e_tot_x - e_field_x))
axs[1].plot(zf, real(e_tot_x), "--")
axs[1].set_xlabel(L"z/\lambda_0")
axs[1].set_ylabel(L"Re(E_{in}, E_{sc}, E_{tot})")
axs[2].plot(zf, imag(e_field_x))
axs[2].plot(zf, imag(e_tot_x - e_field_x))
axs[2].plot(zf, imag(e_tot_x), "--")
axs[2].set_xlabel(L"z/\lambda_0")
axs[2].set_ylabel(L"Im(E_{in}, E_{sc}, E_{tot})")
axs[3].plot(zf, abs.(e_tot_x), "-")
axs[3].plot(zf, abs.(e_field_x), "-")
axs[3].set_xlabel(L"z/\lambda_0")
axs[3].set_ylabel(L"|E_{in}|, |E_{tot}|")


display(fig_e_2)
display(fig_e_5)


"""Total scattering"""
rlim = 1000.0
σ_tot  = AtomicArrays.field_module.forward_scattering(rlim, incident_field,
                                                      S, sm_mat)








fig.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/dynamics_N20_RL_freq_opt_0.pdf", dpi=300)

fig_e_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/field2D_N20_R_freq_opt_0.png", dpi=300)

fig_5.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/field1D_N20_R_freq_opt_0.pdf", dpi=300)

display(fig)
display(fig_e_2)
display(fig_5)


d_iter = range(1e-1, 1e-0, NMAX)
Omega_arr = zeros(NMAX)
Gamma_arr = zeros(NMAX)

Threads.@threads for i=1:NMAX
    Omega_arr[i], Gamma_arr[i] = collective_shift_1array(d_iter[i],0.0, 0.0,1000)
end


fig_3, axs = plt.subplots(ncols=2, nrows=1, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(d_iter, Omega_arr/γ_e[1])
axs[1].plot(d_iter, zeros(NMAX))
axs[1].plot(0.2*ones(NMAX), range(-3.0, 1.0, NMAX), "--", color="black")
axs[1].plot(0.8*ones(NMAX), range(-3.0, 1.0, NMAX), "--", color="black")
axs[1].set_xlabel(L"a/\lambda_0")
axs[1].set_ylabel(L"\Delta_d")
axs[1].set_title("Collective shift")
axs[1].set_ylim([-3,1])
axs[2].plot(d_iter, Gamma_arr/γ_e[1])
axs[2].set_xlabel(L"a/\lambda_0")
axs[2].set_ylabel(L"\Gamma_d")
axs[2].set_title("Collective decay")
display(fig_3)
