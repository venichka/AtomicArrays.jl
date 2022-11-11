#  2D lattice of spins: eigenstates in 1-photon regime
begin
    if pwd()[end-14:end] == "AtomicArrays.jl"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
end

using Pkg
Pkg.activate(PATH_ENV)

using Revise
using AtomicArrays
using QuantumOptics, FFTW
using CairoMakie, GLMakie
using PyPlot
PyPlot.svg(false)
PyPlot.pygui(true)

const PATH_FIGS, PATH_DATA = AtomicArrays.misc.path()
const LAT_TYPE = "lat"

# Parameters

const c_light = 1.0
const lam_0 = 1.0
const k_0 = 2*π / lam_0
const om_0 = 2.0*pi*c_light / lam_0

const Nx = 10
const Ny = 10
const Nz = 2  # number of arrays
const N = Nx*Ny*Nz
const M = 1 # Number of excitations

d = 0.24444
Delt = 0.0
delt = 0.0
d_1 = d
d_2 = d + delt
L = 0.62222
pos_1 = AtomicArrays.geometry.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_1/2, 
                                                           -(Ny-1)*d_1/2,
                                                           -L/2])
pos_2 = AtomicArrays.geometry.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_2/2, 
                                                           -(Ny-1)*d_2/2,
                                                           L/2])
pos = vcat(pos_1, pos_2)
μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0im, 0.0] for i = 1:N]
γ_e = [1e-2 for i = 1:N]
δ_S = [(i < Nx*Ny+1) ? -0.5*Delt : 0.5*Delt for i = 1:N]
S = SpinCollection(pos,μ; gammas=γ_e, deltas=δ_S)

# Plot arrays (atom distribution)
fig_1 = figure(figsize=(7,3.5))
x_at = [pos[i][1] for i = 1:N]
y_at = [pos[i][2] for i = 1:N]
z_at = [pos[i][3] for i = 1:N]
scatter3D(x_at, y_at, z_at)

# Collective effects
Ωmat = OmegaMatrix(S)
Γmat = GammaMatrix(S)

# Hilbert space
b = ReducedSpinBasis(N,M,M) # Basis from M excitations up to M excitations


# Effective Hamiltonian
spsm = [reducedsigmapsigmam(b, i, j) for i=1:N, j=1:N]
H_eff = dense(sum((Ωmat[i,j] - 0.5im*Γmat[i,j])*spsm[i, j]
                  for i=1:N, j=1:N))

# Find the most subradiant eigenstate
λ, states = eigenstates(H_eff; warning=false)
γ = -2 .* imag.(λ)
s_ind_max = sortperm(γ, rev=true)[9]
s_ind = sortperm(γ)[1]
ψ = states[s_ind]
ψ_max = states[s_ind_max]


"""Plot eigenstates"""

# Parity (even: true, odd: false)
parity_n = [(sign(real(states[n].data[1]))==
    sign(real(states[n].data[Nx*Ny+1]))) ? true : false for n = 1:N]


fig_1_1, axs = plt.subplots(ncols=2, nrows=1, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].scatter(x_at[1:Nx*Ny]./d, y_at[1:Nx*Ny]./d,
               c=(abs.((states[s_ind].data)[1:Nx*Ny])/
                   (findmax(abs.((states[s_ind].data)[1:Nx*Ny]))[1])),
        s=200, cmap="Reds")
axs[1].set_xlabel("x/d")
axs[1].set_ylabel("y/d")
axs[1].set_title("dark state")
axs[2].scatter(x_at[Nx*Ny+1:end]./d, y_at[Nx*Ny+1:end]./d,
               c=(abs.((states[s_ind_max].data)[Nx*Ny+1:end])/
                   (findmax(abs.((states[s_ind_max].data)[Nx*Ny+1:end]))[1])),
        s=200, cmap="Reds")
axs[2].set_xlabel("x/d")
axs[2].set_ylabel("y/d")
axs[2].set_title("bright state")
display(fig_1_1)

figure()
PyPlot.plot((abs.((states[s_ind].data)[1:Nx*Ny])/
                   (findmax(abs.((states[s_ind].data)[1:Nx*Ny]))[1])))
PyPlot.plot((abs.((states[s_ind].data)[1:Nx*Ny])/
                   (findmax(abs.((states[s_ind].data)[1:Nx*Ny]))[1])), "o")
display(gcf())


"""Eigenvalues depending on quasimomentum"""

r_perp = [d.*[(j-1)%(Nx), (j-1)÷Nx] for j = 1:Nx*Ny]
q_perp = [-π/d .+ 2*π.*(r_perp[j]./d) ./ ((Nx)*d) for j = 1:Nx*Ny]
states_f = [sum([sqrt(2)*(states[i].data)[j]*exp(1im*r_perp[j]'*q_perp[k])/sqrt(Nx*Ny)
            for j = 1:Nx*Ny]) for i = 1:N, k = 1:Nx*Ny]
states_f_2D = [([states_f[i, (Nx)*(j-1) + (k-1)%(Ny) + 1] for j=1:Nx, k=1:Ny])
           for i=1:N]

q_quasi = [sum([abs(states_f[i, k])^2*norm(q_perp[k]) for k = 1:Nx*Ny])
               for i = 1:N]

# fft approach
q_perp_x = [-π/d + 2*π*(i-1) / d/(Nx) for i=1:Nx]
states_2D = [[states[i].data[(Nx)*(j-1) + (k-1)%(Ny) + 1]
              for j=1:Nx, k=1:Ny] for i = 1:N]
states_2D_fft = [sqrt(2)*fftshift(fft(states_2D[i]))/sqrt(Nx*Ny) for i=1:N]

q_quasi_fft = [sum([sum([abs(states_2D_fft[i][j,k])^2*
                         norm([q_perp_x[j], q_perp_x[k]])
                         for k = 1:Ny]) for j=1:Nx])
               for i = 1:N]

s_ind = sortperm(q_quasi)[1]

s_ind = sortperm(γ, rev=false)[1]
q_quasi[s_ind]
γ[s_ind]

ψ = states[s_ind]

PyPlot.figure()
PyPlot.contourf(abs.(states_2D[s_ind]))
PyPlot.xlabel("x/d")
PyPlot.ylabel("y/d")
display(gcf())

PyPlot.figure()
x_fft = fftfreq(length(x_at[1:Nx]), 2*pi/d) |> fftshift
y_fft = x_fft
PyPlot.contourf(x_fft, y_fft, abs.(states_2D_fft[s_ind]))
PyPlot.xlabel("q_x")
PyPlot.ylabel("q_y")
display(gcf())

PyPlot.figure()
PyPlot.contourf(q_perp_x, q_perp_x, abs.(states_f_2D[s_ind]))
PyPlot.xlabel("q_x")
PyPlot.ylabel("q_y")
display(gcf())

PyPlot.figure()
PyPlot.contourf(abs.(states_f'))
display(gcf())

PyPlot.figure()
PyPlot.plot(q_quasi)
PyPlot.plot(q_quasi_fft)
display(gcf())

PyPlot.figure()
PyPlot.plot(q_quasi[parity_n]./k_0, γ[parity_n]./γ_e[1], "o", color="red")
PyPlot.plot(q_quasi[.!parity_n]./k_0, γ[.!parity_n]./γ_e[1], "o", color="black")
PyPlot.plot(q_quasi_fft./k_0, γ./γ_e, "o", alpha=0.1)
PyPlot.xscale("log")
PyPlot.yscale("log")
PyPlot.xlabel(L"\bar{q}/k_0")
PyPlot.ylabel(L"\gamma_n/\gamma_e")
display(gcf())

PyPlot.figure()
PyPlot.plot(γ[sortperm(γ)[parity_n]]./γ_e[1], "o", color="red")
PyPlot.plot(γ[sortperm(γ)[.!parity_n]]./γ_e[1], "o", color="black")
PyPlot.xscale("log")
PyPlot.yscale("log")
PyPlot.xlabel(L"n")
PyPlot.ylabel(L"\gamma_n/\gamma_e")
display(gcf())

PyPlot.figure()
PyPlot.plot(γ[sortperm(γ)]./γ_e[1], "o", color="blue")
PyPlot.xscale("log")
PyPlot.yscale("log")
PyPlot.xlabel(L"n")
PyPlot.ylabel(L"\gamma_n/\gamma_e")
display(gcf())

"""Compute the radiation pattern"""

function G(r,i,j) # Green's Tensor overlap
    G_i = GreenTensor(r-pos[i])
    G_j = GreenTensor(r-pos[j])
    return μ[i]' * (G_i'*G_j) * μ[j]
end
function intensity(r, ψ) # The intensity ⟨E⁻(r)⋅E⁺(r)⟩
    real(sum(expect(spsm[i,j], ψ)*G(r,i,j) for i=1:N, j=1:N))
end


function intensity_xy()
    # X-Y view
    x = range(-1d+pos[Nx*Ny+1][1], pos[end][1]+1d, 50)
    y = range(-1d+pos[Nx*Ny+1][1], pos[end][1]+1d, 50)
    z = L + 0.4d
    I = zeros(length(x), length(y))
    I_max = zeros(length(x), length(y))
    Threads.@threads for i=1:length(x)
        for j=1:length(y)
            I[i,j] = intensity([x[i],y[j],z], ψ)
            I_max[i,j] = intensity([x[i],y[j],z], ψ_max)
            print(i, "  ", j,"\n")
        end
    end

    # Plot
    # I_arr = [I[(i - 1)%length(x) + 1, (i - 1)÷length(x)+1] for i = 1:length(x)^2]
    # I_arr_sorted = sort(I_arr, rev=true)[N:end]  # sorted and truncated Intensity
    # I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    # I_arr_max = findmax(I_arr)[1]


    # fig_2 = PyPlot.figure(figsize=(9,4))
    # # Lin scale
    # levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
    # PyPlot.contourf(x./d,y./d,I',30, levels=levels)

    # # Log scale
    # #levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
    # #                      length=100))
    # #contourf(x./d,z./d,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
    # for p in pos
    #     PyPlot.plot(p[1]./d,p[2]./d,"o",color="w",ms=2)
    # end
    # PyPlot.xlabel("x/d")
    # PyPlot.ylabel("y/d")
    # PyPlot.colorbar(label="Intensity",ticks=[])
    # PyPlot.tight_layout()
    return x, y, I, I_max
end

function intensity_xz()
    # X-Z view
    x = range(-1d+pos[Nx*Ny+1][1], pos[end][1]+1d, 50)
    y = 0.0#0.5*(pos[Nx*Ny+7*Nx+1][2] + pos[end][2])
    z = range(-3d+pos[1][3], pos[end][3]+3d, 50)
    I = zeros(length(x), length(z))
    I_max = zeros(length(x), length(z))
    Threads.@threads for i=1:length(x)
        for j=1:length(z)
            I[i,j] = intensity([x[i],y,z[j]], ψ)
            I_max[i,j] = intensity([x[i],y,z[j]], ψ_max)
            print(i, "  ", j,"\n")
        end
    end

    # Plot
    # I_arr = [I[(i - 1)%length(x) + 1, (i - 1)÷length(z)+1]
    #          for i = 1:length(x)*length(z)]
    # I_arr_sorted = sort(I_arr, rev=true)[N:end]  # sorted and truncated Intensity
    # I_arr_av = (sum(I_arr_sorted) / (length(I_arr_sorted)))
    # I_arr_max = findmax(I_arr)[1]


    # fig_2 = PyPlot.figure(figsize=(9,4))
    # # Lin scale
    # levels = 0:I_arr_max*1e-2:I_arr_max*1e-0
    # #levels = 0:1e-4:2e-2
    # PyPlot.contourf(x./d,z./d,I',30, levels=levels)

    # # Log scale
    # #levels = exp10.(range(log10((I_arr_sorted[end])), log10(I_arr_sorted[1]),
    # #                      length=100))
    # #contourf(x./d,z./d,I',30, levels=levels, norm=matplotlib.colors.LogNorm())
    # for p in pos
    #     PyPlot.plot(p[1]./d,p[3]./d,"o",color="w",ms=2)
    # end
    # PyPlot.xlabel("x/d")
    # PyPlot.ylabel("z/d")
    # PyPlot.colorbar(label="Intensity",ticks=[])
    # PyPlot.tight_layout()
    return x, z, I, I_max
end

x_xy, y_xy, I_xy, Imax_xy = intensity_xy()
x_xz, z_xz, I_xz, Imax_xz = intensity_xz()


"""Calculate the collective shift depending on the lattice constant"""

function collective_shift(d, L, N)
    Nx = N
    Ny = N
    Nz = 2  # number of arrays
    delt = 0.0
    d_1 = d
    d_2 = d + delt
    pos_1 = geometry.rectangle(d_1, d_1; Nx=Nx, Ny=Ny, position_0=[0.,0.,0.])
    pos_2 = geometry.rectangle(d_2, d_2; Nx=Nx, Ny=Ny, position_0=[0.,0.,L])
    pos = vcat(pos_1, pos_2)
    # shift the origin of the array
    #p_x0 = pos[1][1]
    #p_xN = pos[Nx*Ny][1]
    #p_y0 = pos[1][2]
    #p_yN = pos[Nx*Ny][2]
    #for i = 1:Nx*Ny
    #    pos[i][1] = pos[i][1] - 0.5*(p_x0 + p_xN)
    #    pos[i][2] = pos[i][2] - 0.5*(p_y0 + p_yN)
    #    #pos[i+Nx*Ny][1] = (pos[i+Nx*Ny][1] - 0.5*(p_x0 + p_xN)) * d_2/d_1
    #    #pos[i+Nx*Ny][2] = (pos[i+Nx*Ny][2] - 0.5*(p_y0 + p_yN)) * d_2/d_1
    #end
    μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0im, 0.0] for i = 1:N]
    γ_e = [1e-0 for i = 1:N]
    S = SpinCollection(pos,μ; gammas=γ_e)
    Omega, Gamma = AtomicArrays.effective_interaction.effective_interactions(S)
end

NMAX = 100
d_iter = range(1e-1, 1e-0, NMAX)
Omega_arr = zeros(NMAX)
Gamma_arr = zeros(NMAX)

Threads.@threads for i=1:NMAX
    Omega_arr[i], Gamma_arr[i] = collective_shift(d_iter[i],0.49,100)
end


fig_3, axs = plt.subplots(ncols=2, nrows=1, figsize=(5.7, 3),
                        constrained_layout=true)
axs[1].plot(d_iter, Omega_arr)
axs[1].set_xlabel(L"a/\lambda_0")
axs[1].set_ylabel(L"\Delta_d")
axs[1].set_title("Collective shift")
axs[1].set_ylim([-3,1])
axs[2].plot(d_iter, Gamma_arr)
axs[2].set_xlabel(L"a/\lambda_0")
axs[2].set_ylabel(L"\Gamma_d")
axs[2].set_title("Collective decay")
display(fig_3)


"""Fidelity"""

# TODO: find the way to project the steady-state of mf model on reduced spin 
#       states

ρ_mf = AtomicArrays.meanfield.densityoperator(state_mf_t[end])



"Publication plots"

function phases_ampl_lattice_plot()
    # GLMakie.activate!()
    CairoMakie.activate!()

    f = Makie.Figure(resolution=(1200, 1000))

    ticks_arg = [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]
    args_num = range(-pi, pi, 5)

    arr_idx = 1:N÷2

    # Plot
    Axis(f[1, 1], title=L"Dark state$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc11 = scatter!(x_at[arr_idx], y_at[arr_idx], markersize = 30,
                    color = (abs.((ψ.data)[arr_idx])/
                    (findmax(abs.((ψ.data)[arr_idx]))[1])),
                    colormap=:BuPu_6)
    Colorbar(f[1, 2], sc11, label = L"$|\psi^D_j|$", height = Relative(1),
             labelsize=28, ticklabelsize=18)
    Axis(f[2, 1],
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc21 = scatter!(x_at[arr_idx], y_at[arr_idx], markersize = 30, colorrange=(-pi, pi),
                    color = angle.(ψ.data[arr_idx]),
                    colormap=:hsv)
    Colorbar(f[2, 2], sc21, label = L"arg$(\psi^D_j)$", height = Relative(1),
             labelsize=28, ticklabelsize=22, ticks=(args_num, ticks_arg))

    Axis(f[1, 3], title=L"Bright state$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc13 = scatter!(x_at[arr_idx], y_at[arr_idx], markersize = 30,
                    color = (abs.((ψ_max.data)[arr_idx])/
                    (findmax(abs.((ψ_max.data)[arr_idx]))[1])),
                    colormap=:BuPu_6)
    Colorbar(f[1, 4], sc13, label = L"$|\psi^B_j|$", height = Relative(1),
             labelsize=28, ticklabelsize=18)

    Axis(f[2, 3],
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    sc23 = scatter!(x_at[arr_idx], y_at[arr_idx], markersize = 30, colorrange=(-pi, pi),
                    color = angle.(ψ_max.data[arr_idx]),
                    colormap=:hsv)
    Colorbar(f[2, 4], sc23, label = L"arg$(\psi^B_j)$", height = Relative(1),
             labelsize=28, ticklabelsize=22, ticks=(args_num, ticks_arg))

    # Adding letters
    ga = f[1, 1] = GridLayout()
    gb = f[1, 3] = GridLayout()
    gc = f[2, 1] = GridLayout()
    gd = f[2, 3] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)",
                                "(c)", "(d)"],
                               [ga, gb, gc, gd])
        Label(layout[1, 1, TopLeft()], label,
              textsize = 30,
              font = "TeX Gyre Heros Bold",
              padding = (0, 20, 0, 0),
              halign = :right)
    end

    colsize!(f.layout, 1, Auto(1.))

    save((PATH_FIGS * "psiDB_abs_arg_" * LAT_TYPE * "_" * string(Nx)
                    * "x" * string(Ny) * ".pdf"), f) # here, you save your figure.
    return f
end


function field_plot()
    # GLMakie.activate!()
    CairoMakie.activate!()
    f = Makie.Figure(resolution=(1050, 1000))
    # Plot
    # X-Y
    Axis(f[1, 1], title=L"Dark state$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    c11 = contourf!(x_xy, y_xy, I_xy, colormap=:BuPu_6)
    sc11 = scatter!(x_at[1:N÷2], y_at[1:N÷2], markersize = 10,
                    color = :black)
    # Colorbar(f[1, 2], c11, label = L"$$Intensity", #width = Relative(1),
    #          labelsize=28, ticklabelsize=18, vertical=true,
    #          ticks=([minimum(I_xy), maximum(I_xy)], ["0", "max"])
    #          )
    # X-Z
    Axis(f[2, 1], #title=L"Dark state$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"z/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    c12 = contourf!(x_xz, z_xz, I_xz, colormap=:BuPu_6)
    sc12 = scatter!(x_at[1:N], z_at[1:N], markersize = 10,
                    color = :black)
    # Colorbar(f[2, 2], c12, label = L"$$Intensity", #width = Relative(1),
    #          labelsize=28, ticklabelsize=18, vertical=true,
    #          ticks=([minimum(I_xz), maximum(I_xz)], ["0", "max"])
    #          )

    # Bright
    # X-Y
    Axis(f[1, 2], title=L"Bright state$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"y/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    c21 = contourf!(x_xy, y_xy, Imax_xy, colormap=:BuPu_6)
    sc21 = scatter!(x_at[1:N÷2], y_at[1:N÷2], markersize = 10,
                    color = :black)
    # Colorbar(f[1, 4], c21, label = L"$$Intensity", #width = Relative(1),
    #          labelsize=28, ticklabelsize=18, vertical=true,
    #          ticks=([minimum(Imax_xy), maximum(Imax_xy)], ["0", "max"])
    #          )
    # X-Z
    Axis(f[2, 2], #title=L"Dark state$$",
        xlabel=L"x/\lambda_0",
        ylabel=L"z/\lambda_0",
        xlabelsize=28, ylabelsize=28,
        titlesize=30,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsize=22,
        yticklabelsize=22)
    c22 = contourf!(x_xz, z_xz, Imax_xz, colormap=:BuPu_6)
    sc22 = scatter!(x_at[1:N], z_at[1:N], markersize = 10,
                    color = :black)
    Colorbar(f[:, 3], c22, label = L"$$Intensity", #width = Relative(1),
             labelsize=30, ticklabelsize=24, vertical=true,
             ticks=([minimum(Imax_xz), maximum(Imax_xz)], ["0", "max"]),
             height=450
             )

    # Adding letters
    ga = f[1, 1] = GridLayout()
    gb = f[1, 2] = GridLayout()
    gc = f[2, 1] = GridLayout()
    gd = f[2, 2] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)",
                                "(c)", "(d)"],
                               [ga, gb, gc, gd])
        Label(layout[1, 1, TopLeft()], label,
              textsize = 30,
              font = "TeX Gyre Heros Bold",
              padding = (0, 40, 20, 0),
              halign = :right)
    end

    colsize!(f.layout, 1, Auto(1.))
    save((PATH_FIGS * "intensityEig_" * LAT_TYPE * "_" * string(Nx)
                    * "x" * string(Ny) * ".pdf"), f) # here, you save your figure.
    return f
end

phases_ampl_lattice_plot()

field_plot()
