#  Two arrays: total field dynamics
using QuantumOptics
using FFTW
using BenchmarkTools
using PyPlot
using LinearAlgebra
using DifferentialEquations, Sundials
using Revise

using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices
const mapexpect = AtomicArrays.meanfield_module.mapexpect
const mapexpect_mpc = AtomicArrays.mpc_module.mapexpect
const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices

# pygui(true)
gcf()


const PATH_FIG = "/Users/jimi/Google Drive/Work/In process/Projects/\
                  Collective_effects_QMS/Figures/two_arrays"
const NMAX = 100
const NMAX_T = 5
const DIRECTION = "left"

"""Parameters"""
const c_light = 1.0
const lam_0 = 1.0
const k_0 = 2*π / lam_0
const om_0 = 2.0*pi*c_light / lam_0
 
const Nx = 2
const Ny = 2
const Nz = 2  # number of arrays
const N = Nx*Ny*Nz
const M = 1 # Number of excitations

"Key parameters"

d = 0.1 #0.147
delt = 0.0 #0.147
Delt = 0.01
d_1 = d
d_2 = d + delt
L = 0.1#0.7158
E_ampl = 9.7e-4 + 0im#0.001 + 0im

"Spin arrays"

pos_1 = geometry_module.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_1/2, 
                                                           -(Ny-1)*d_1/2,
                                                           -L/2])
pos_2 = geometry_module.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_2/2, 
                                                           -(Ny-1)*d_2/2,
                                                           L/2])
pos = vcat(pos_1, pos_2)
μ = [(i < 0) ? [1, 0, 0.0] : [1.0, 0.0im, 0.0] for i = 1:N]
γ_e = [1e-2 for i = 1:N]
δ_S = [(i < Nx*Ny + 1) ? -0.5*Delt : 0.5*Delt for i = 1:N]
S = SpinCollection(pos,μ; gammas=γ_e, deltas=δ_S)


# Plot arrays (atom distribution)
fig_1 = PyPlot.figure(figsize=(5,5))
x_at = [vec[1] for vec = pos]
y_at = [vec[2] for vec = pos]
z_at = [vec[3] for vec = pos]
PyPlot.scatter3D(x_at, y_at, z_at)

# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)


"Incident field"

om_f = om_0

E_kvec = om_f/c_light
E_width = 0.3*d*sqrt(Nx*Ny)
if (DIRECTION == "right")
    E_pos0 = [0.0,0.0,0.0]
    E_polar = [1.0, 0im, 0.0]
    E_angle = [0.0, 0.0]  # {θ, φ}
elseif (DIRECTION == "left")
    E_pos0 = [0.0,0.0,0.0*L]
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

x = range(-0.5, 0.5, NMAX)
y = 0.5*(pos[1][2] + pos[Nx*Ny][2])
z = range(-1., 1., NMAX)
e_field = Matrix{ComplexF64}(undef, length(x), length(z))
for i in eachindex(x), j in eachindex(z)
    e_field[i,j] = em_inc_function([x[i],y,z[j]], incident_field)[1]
end


fig_0 = PyPlot.figure(figsize=(7,3))
PyPlot.contourf(z, x, real(e_field), 30)
for p in pos
    PyPlot.plot(p[3],p[1],"o",color="w",ms=2)
end
PyPlot.ylabel("x")
PyPlot.xlabel("z")
PyPlot.colorbar(label="Amplitude")
PyPlot.tight_layout()


"""Dynamics: meanfield"""

# E_field vector for Rabi constant computation
E_vec = [em_inc_function(S.spins[k].position,incident_field)
         for k = 1:N]

Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

fig_1, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(abs.(Om_R))
axs[1].plot(abs.(Om_R), "o")
axs[1].set_title(L"|\Omega_R|")
axs[2].plot(real(Om_R),"-o")
axs[2].plot(imag(Om_R), "-o")
axs[2].set_title(L"\Re(\Omega_R), \Im(\Omega_R)")

tmax = 20e5
const T = [0:tmax/100:tmax;]
# Initial state (Bloch state)
const phi = 0.
const theta = pi/1.

# Meanfield
state0 = AtomicArrays.meanfield_module.blochstate(phi, theta, N)
tout, state_mf_t = AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0);
ss_mf = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(AutoVern7(RadauIIA5(), nonstifftol = 5//10)))
ss_mf = AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=SSRootfind())
ss_mf - state_mf_t[end].data
ss_mf_state = AtomicArrays.meanfield_module.ProductState(ss_mf.u)
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=SSRootfind(), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(CVODE_BDF(linear_solver=:GMRES)), dt=0.5);
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(RadauIIA5()), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(Rodas5()), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(CompositeAlgorithm((Vern7(), RadauIIA5()), AutoSwitch(Vern7(), RadauIIA5()))), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(Vern7()));
@btime AtomicArrays.meanfield_module.steady_state_field(T, S, Om_R, state0, 
        alg=DynamicSS(AutoVern7(RadauIIA5(), nonstifftol = 3//10)), reltol=1e-10, abstol=1e-12);



@btime AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0, reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0,
        alg=Rodas5(), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0,
        alg=Vern7(), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0,
        alg=VCABM(), reltol=1e-10, abstol=1e-12);
@btime AtomicArrays.meanfield_module.timeevolution_field(T, S, Om_R, state0,
        alg=CVODE_BDF(linear_solver=:GMRES), reltol=1e-10, abstol=1e-12);


# MPC
# state0_mpc = AtomicArrays.mpc_module.blochstate(phi, theta, N)
state0_mpc = AtomicArrays.mpc_module.state_from_mf(ss_mf_state, phi, theta, N)
# state0 = state_mpc_t[end]
tout, state_mpc_t = AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0_mpc, alg=VCABM(), reltol=1e-10, abstol=1e-12);
ss_mpc = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc, alg=DynamicSS(CVODE_BDF(linear_solver=:GMRES)), dt=0.01, reltol=1e-10,abstol=1e-12);
ss_mpc = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc, alg=DynamicSS(VCABM()), reltol=1e-10,abstol=1e-12);
ss_mpc = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc, alg=DynamicSS(Rodas5()), reltol=1e-10,abstol=1e-12);
using NLsolve
ss_mpc = AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc,
alg=SSRootfind(nlsolve = (f, u0, abstol) -> (NLsolve.nlsolve(f, u0,
                                                autodiff = :forward,
                                                method = :anderson,
                                                iterations = Int(1e5),
                                                ftol = abstol))));

ss_mpc - state_mpc_t[end].data


@btime AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc,
 alg=DynamicSS(CVODE_BDF(linear_solver=:GMRES)), dt=0.5, reltol=1e-6, abstol=1e-8);
@btime AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc,
 alg=DynamicSS(AutoVern7(Rodas4P())),reltol=1e-6, abstol=1e-8);
@btime AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc,
 alg=DynamicSS(AutoVern7(RadauIIA5(), nonstifftol=9//10)),reltol=1e-6, abstol=1e-8);
@btime AtomicArrays.mpc_module.steady_state_field(T, S, Om_R, state0_mpc, alg=SSRootfind(), reltol=1e-6,abstol=1e-8);



@btime AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0_mpc; alg=Rodas5(),maxiters=1e5,reltol=1e-8,abstol=1e-6);
@btime AtomicArrays.mpc_module.timeevolution_field(T, S, Om_R, state0_mpc; alg=Rodas5(),maxiters=1e5,reltol=1e-8,abstol=1e-6);





# Expectation values
begin
    sx_mf = sum([mapexpect(AtomicArrays.meanfield_module.sx, state_mf_t, i) for i=1:N]) ./ (N)
    sy_mf = sum([mapexpect(AtomicArrays.meanfield_module.sy, state_mf_t, i) for i=1:N]) ./ (N)
    sz_mf = sum([mapexpect(AtomicArrays.meanfield_module.sz, state_mf_t, i) for i=1:N]) ./ (N)

    sx_mpc = sum([mapexpect_mpc(AtomicArrays.mpc_module.sx, state_mpc_t, i) for i=1:N]) ./ (N)
    sy_mpc = sum([mapexpect_mpc(AtomicArrays.mpc_module.sy, state_mpc_t, i) for i=1:N]) ./ (N)
    sz_mpc = sum([mapexpect_mpc(AtomicArrays.mpc_module.sz, state_mpc_t, i) for i=1:N]) ./ (N)

    sx_mpc_ss = sum([mapexpect_mpc(AtomicArrays.mpc_module.sx, [AtomicArrays.mpc_module.MPCState(ss_mpc.u)], i) for i=1:N]) ./ (N)
    sy_mpc_ss = sum([mapexpect_mpc(AtomicArrays.mpc_module.sy, [AtomicArrays.mpc_module.MPCState(ss_mpc.u)], i) for i=1:N]) ./ (N)
    sz_mpc_ss = sum([mapexpect_mpc(AtomicArrays.mpc_module.sz, [AtomicArrays.mpc_module.MPCState(ss_mpc.u)], i) for i=1:N]) ./ (N)
end

begin
    fig, axs = PyPlot.subplots(ncols=1, nrows=3, figsize=(6, 9),
                            constrained_layout=true)
    axs[1].plot(T, sx_mf, label="mean field")
    axs[1].plot(T, sx_mpc, label="mpc")
    axs[1].plot(T, [sx_mpc_ss[1] for t in T], "--", color="black")
    axs[1].set_ylim(-0.02451,-0.024485)
    axs[1].set_xscale("linear")
    axs[1].set_ylabel(L"\langle \sigma_x \rangle")
    axs[1].set_title("Average values")
    axs[1].legend()

    axs[2].plot(T, sy_mf, label="mean field")
    axs[2].plot(T, sy_mpc, label="mpc")
    axs[2].plot(T, [sy_mpc_ss[1] for t in T], "--", color="black")
    axs[2].set_ylim(-0.02942,-0.02935)
    axs[2].set_xscale("linear")
    axs[2].set_ylabel(L"\langle \sigma_y \rangle")

    axs[3].plot(T, sz_mf, label="mean field")
    axs[3].plot(T, sz_mpc, label="mpc")
    axs[3].plot(T, [sz_mpc_ss[1] for t in T], "--", color="black")
    axs[3].set_ylim(-0.9995,-0.999)
    axs[3].set_xscale("linear")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel(L"\langle \sigma_z \rangle")
    display(fig)
end

t_ind = length(T)
#sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices(state_mf_t, t_ind)
#sx_mat_c, sy_mat_c, sz_mat_c, sm_mat_c, sp_mat_c = sigma_matrices_mpc(state_mpc_t, t_ind)
sx_mat_mf, sy_mat_mf, sz_mat_mf, sm_mat_mf, sp_mat_mf = sigma_matrices(state_mf_t, t_ind)
sx_mat, sy_mat, sz_mat, sm_mat, sp_mat = sigma_matrices_mpc(state_mpc_t, t_ind)


PyPlot.figure()
PyPlot.plot(abs.(sm_mat),"o")
PyPlot.plot(abs.(sm_mat_mf),"o")
PyPlot.title(L"\sigma_-(\infty)")
display(gcf())

"""Compute the radiation pattern"""

VIEW = "x-z"
distance = 0.3  # in terms of d

if VIEW == "x-y"
# X-Y view
    x = range(pos[Nx*Ny+1][1] - 3d, pos[N][1] + 3d, NMAX)
    y = range(pos[Nx*Ny+1][2] - 3d, pos[N][2] + 3d, NMAX)
    z = -sign(E_angle[1])*(L + (distance+5.3)*d) + L + distance*d
    t_ind = length(T)
    I = zeros(length(x), length(y))
    E_scat = Matrix{Vector{ComplexF64}}(undef, length(x), length(y))
    Threads.@threads for i=1:length(x)
        for j=1:length(x)
            E_t = AtomicArrays.field_module.total_field(em_inc_function,
                                                        [x[i],y[j],z],
                                                        incident_field,
                                                        S, sm_mat)
            I[i,j] = (norm(E_t)^2 / abs(E_ampl)^2)
            E_scat[i,j] = (E_t - em_inc_function([x[i],y[j],z], incident_field))
            print(i, "  ", j,"\n")
        end
    end

    # Plot
    I_arr = [I[(i - 1)%length(x) + 1, (i - 1)÷length(x)+1] for i = 1:length(x)^2]
    I_arr_sorted = sort(I_arr, rev=true)[1:end]  # sorted and truncated Intensity
    I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    I_arr_max = findmax(I_arr)[1]


    fig_2 = PyPlot.figure(figsize=(9,4))
    # Lin scale
    #levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
    #contourf(x./d,y./d,I',30, levels=levels)

    # Log scale
    levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
                          length=100))
    PyPlot.contourf(x./d,y./d,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
    for p in pos
        PyPlot.plot(p[1]./d,p[2]./d,"o",color="w",ms=2)
    end
    PyPlot.xlabel("x/d")
    PyPlot.ylabel("y/d")
    PyPlot.colorbar(label=L"|E_{tot}|^2 / |E_0|^2")
    PyPlot.tight_layout()


    """FFT of the scattered field"""

    E_scat_j = [E_scat[i,j][1] for i = 1:length(x),j = 1:length(y)]
    # fft approach
    I_fft = sqrt(2)*fftshift(fft(ifftshift( E_scat_j )))/sqrt(Nx*Ny)
    x_fft = fftfreq(length(x), 2*pi*NMAX/(x[end]-x[1])) |> fftshift
    y_fft = x_fft

    fig_4, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 3),
                              constrained_layout=true)
    levels_r = exp10.(range(log10(findmin(abs.(E_scat_j))[1]), log10(findmax(abs.(E_scat_j))[1]),
                            length=100))
    levels_fft = exp10.(range(log10(findmin(abs.(I_fft))[1]), log10(findmax(abs.(I_fft))[1]),
                              length=100))
    #contourf(x_fft, y_fft, abs.(I_fft),30, levels=levels, norm=matplotlib.colors.LogNorm())
    axs[1].contourf(x, y, abs.(E_scat_j'),
                    30, levels=levels_r, norm=matplotlib.colors.LogNorm())
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[2].contourf(x_fft, y_fft, abs.(I_fft'),
                    30, levels=levels_fft, norm=matplotlib.colors.LogNorm())
    axs[2].set_xlabel("q_x")
    axs[2].set_ylabel("q_y")


    E_scat_x = [E_scat[i,j][1] for i = 1:length(x),j = 1:length(y)]
    E_scat_y = [E_scat[i,j][2] for i = 1:length(x),j = 1:length(y)]
    E_scat_z = [E_scat[i,j][3] for i = 1:length(x),j = 1:length(y)]

    PyPlot.figure()
    PyPlot.quiver(x,y, abs.(E_scat_x),abs.(E_scat_y), sqrt.(abs.(E_scat_x).+abs.(E_scat_y)))

    """Vector plots using Makie"""

    using GLMakie

    f = GLMakie.Figure(resolution = (1200, 1200), fontsize=46)
    GLMakie.Axis(f[1, 1], backgroundcolor = "black",
                 title=L"\Re(E_{x,y})", xlabel=L"x", ylabel=L"y")
    ps2 = [Point2(x[i], y[j])
           for i in 1:3:NMAX for j in 1:3:NMAX]
    ns2 = [real(Vec2(E_scat_x[i,j], E_scat_y[i,j]))
           for i in 1:3:NMAX for j in 1:3:NMAX]
    strength = norm.(ns2)
    normal = strength / findmax(strength)[1]
    GLMakie.contour!(x, y, I, levels=10, linewidth=4, colormap=:lake)
    GLMakie.arrows!(ps2, ns2,
                    arrowsize = 20.0*normal, colormap=:lake,
                    lengthscale = 40, linewidth = 1,
                    arrowcolor = normal, linecolor = normal)
    f
    #limits!(-1.1,1.1,-1.1,1.1)

    #GLMakie.save(PATH_FIG*"/scattering_asym_d/d_02/reExy_d02_delt01_L045_R.png",
    #             f, px_per_unit = 2)

    # 3D plot
    #ps = [Point3(x[i], y[j], -2.0 + (k-1)*4)
    #      for i in 1:5:NMAX for j in 1:5:NMAX for k in 1:2]
    #ns = [abs.(Vec3(E_scat_x[i,j], E_scat_y[i,j], E_scat_z[i,j]))
    #      for i in 1:5:NMAX for j in 1:5:NMAX for k in 1:2]
    #strength = norm.(ns)
    #arrows(
    #    ps, ns, fxaa=true, # turn on anti-aliasing
    #    color = strength,
    #    arrowsize=Vec3(0.2, 0.2, 0.3),
    #    #normalize=true,
    #    #lengthscale=0.02,
    #    align=:origin,
    #    axis=(type=Axis3,),
    #)
elseif VIEW == "x-z"
# X-Z view
    x = range(-4.5, 4.5, NMAX) #range(-Nx*d, Nx*d, NMAX)
    y = 0#0.5*(pos[1][2]+pos[Nx*Ny][2])
    z = range(-4.5, 4.5, NMAX)
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
    I_arr_sorted = sort(I_arr, rev=true)#[N_t:end]  # sorted and truncated Intensity
    I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    I_arr_max = findmax(I_arr)[1]

    fig_2, axs = plt.subplots(ncols=3, nrows=1, figsize=(10, 3),
                              constrained_layout=true)
    levels_f = -1.2:1e-2:1.2
    levels_I = 0:1e-2:2
    c1 = axs[1].contourf(z, x, I, 30, levels=levels_I)
    axs[1].set_xlabel("z")
    axs[1].set_ylabel("x")
    fig_2.colorbar(c1, label=L"|E_{tot}|^2 / |E_0|^2")
    c2 = axs[2].contourf(z, x, E_sc, 30, levels=levels_f)
    axs[2].set_xlabel("z")
    fig_2.colorbar(c2, label=L"Re(E_{tot} / E_0)")
    c3 = axs[3].contourf(z, x, E_tot, 30, levels=levels_f)
    axs[3].set_xlabel("z")
    fig_2.colorbar(c3, label=L"Re(E_{sc} / E_0)")
    for p in pos
        axs[1].plot(p[3],p[1],"o",color="b",ms=4)
        axs[2].plot(p[3],p[1],"o",color="b",ms=4)
        axs[3].plot(p[3],p[1],"o",color="b",ms=4)
    end
end


"""Field study"""

PyPlot.figure()
PyPlot.plot((VIEW == "x-y") ? y : z, I[length(x)÷2,:])
PyPlot.title(L"|E_{tot}|^2 / |E_0|^2")
display(gcf())

xf = 0.5*(pos[1][1] + pos[Nx*Ny][1])
yf = 0.5*(pos[1][2] + pos[Nx*Ny][2])
zf = range(-10, 10, 2*NMAX)
e_field_x = zeros(ComplexF64, length(zf))
e_tot_x = zeros(ComplexF64, length(zf))
for i = 1:length(zf)
    e_field_x[i] = (em_inc_function([xf,yf,zf[i]], incident_field)[1])
    e_tot_x[i] = (AtomicArrays.field_module.total_field(em_inc_function,
                                                        [xf,yf,zf[i]],
                                                        incident_field,
                                                        S, sm_mat)[1])
end

fig_5, axs = plt.subplots(ncols=1, nrows=3, figsize=(5.7, 3),
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

display(fig_2)
display(fig_5)

"""Transmission"""

xlim = 0.0001
ylim = 0.0001
zlim = (E_angle[1] >= π/2) ? -1000. : 1000. + L
x_t = range(-xlim, xlim, NMAX_T)
y_t = range(-ylim, ylim, NMAX_T)
E_out = sum([norm(AtomicArrays.field_module.total_field(em_inc_function,
                                                           [x_t[i],y_t[j],zlim],
                                                           incident_field,
                                                           S, sm_mat))
            for i = 1:NMAX_T, j = 1:NMAX_T])
E_in = sum([norm(em_inc_function([x_t[i],y_t[j],zlim], incident_field))
            for i = 1:NMAX_T, j = 1:NMAX_T])
transmission = (E_out ./ E_in).^2


zlim = 0.7*(d+delt)*(Nx)
n_samp = 400
@time tran, points = AtomicArrays.field_module.transmission_reg(incident_field, em_inc_function,
                                       S, sm_mat; samples=n_samp, zlim=zlim, angle=[π, π]);
tran


"""Forward scattering"""

zlim2 = 1000
σ_tot  = AtomicArrays.field_module.forward_scattering(zlim2, incident_field,
                                                      S, sm_mat)

# Plot points on a hemi-sphere
fig_11 = PyPlot.figure(figsize=(5,5))
x_p = [points[i][1] for i = 1:length(points)]
y_p = [points[i][2] for i = 1:length(points)]
z_p = [points[i][3] for i = 1:length(points)]
PyPlot.scatter3D(x_p, y_p, z_p)
#write("../Data/test.bin", I)

fig_22 = PyPlot.figure(figsize=(9,4))
# Lin scale
#levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
#PyPlot.contourf(z,x,I,30, levels=levels)

# Log scale
levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
                      length=100))
PyPlot.contourf(z,x,I,30, levels=levels, norm=matplotlib.colors.LogNorm())
for p in pos
    PyPlot.plot(p[3],p[1],"o",color="w",ms=2)
end
for i = 1:100
    if (DIRECTION == "right")
        PyPlot.plot(zlim*cos(-π/2+π*(i-1)/99.)+L, zlim*sin(-π/2+π*(i-1)/99.) ,
                    "o",color="cyan",ms=2)
    elseif (DIRECTION == "left")
        PyPlot.plot(-zlim*cos(-π/2+π*(i-1)/99.), -zlim*sin(-π/2+π*(i-1)/99.) ,
                    "o",color="cyan",ms=2)
    end
end
PyPlot.xlabel("z")
PyPlot.ylabel("x")
PyPlot.colorbar(label=L"|E_{tot}|^2 / |E_0|^2")
PyPlot.tight_layout()


"""Check radiation pattern"""

using GLMakie

tf = AtomicArrays.field_module.total_field
positions = vec([(x_p[i], y_p[i], z_p[i]) for i in 1:length(x_p)])
vals = [norm(tf(em_inc_function,points[ip],incident_field,S, sm_mat))^2/abs(E_ampl)^2
        for ip in 1:length(points)]
fig, ax, pltobj = GLMakie.meshscatter(positions, color = vec(vals),
    marker = FRect3D(Vec3f0(zlim*0.5), Vec3f0(zlim*0.5)), # here, if you use less than 10, you will see smaller squares.
    colormap = :bwr, colorrange = (minimum(vals), maximum(vals)),
    transparency = false, # set to false, if you don't want the transparency.
    shading= true,
    figure = (; resolution = (1200,1200)),
    axis=(; type=Axis3, perspectiveness = 0.5,  azimuth = 7.19, elevation = 0.57,
        xlabel = "x", ylabel = "y", zlabel = "z",
        aspect = (1,1,1/2)))
cbar = GLMakie.Colorbar(fig, pltobj, label = L"|E_{tot}|^2 / |E_0|^2", height = Relative(0.5))
GLMakie.xlims!(ax,-zlim,zlim)
GLMakie.ylims!(ax,-zlim,zlim)
GLMakie.zlims!(ax, (E_angle[1] >= π/2) ? -zlim : L,
               (E_angle[1] >= π/2) ? 0 : L + zlim)
fig[1,2] = cbar
fig
#save(PATH_FIG * "fieldDistSphere_d02_L.png", fig) # here, you save your figure.


fig.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/dynamics_N10_RL_lat.pdf", dpi=300)

fig_2.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/field2D_N10_L_lat.png", dpi=300)

fig_5.savefig("/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_lone/field1D_N10_L_lat.pdf", dpi=300)

display(fig)
display(fig_2)
display(fig_5)