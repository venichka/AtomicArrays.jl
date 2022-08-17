using Distributed
addprocs(4)

@everywhere begin
    if pwd()[end-14:end] == "AtomicArrays.jl"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
end

using SharedArrays
using DelimitedFiles

@everywhere using Pkg
@everywhere Pkg.activate(PATH_ENV)

@everywhere begin
    using ProgressMeter
    using QuantumOptics
    using CairoMakie, GLMakie
    using LinearAlgebra, EllipsisNotation
    using LsqFit

    using Revise
    using AtomicArrays

    import EllipsisNotation: Ellipsis
    const .. = Ellipsis()

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()
    const LAT_TYPE = "lat"

    const NMAX = 20

    delt_list = range(0.0, 0.1, NMAX)
    Delt_list = range(-0.1, 0.1, NMAX)
    L_list = range(1.0e-1, 10e-1, NMAX)
    d_list = range(1.0e-1, 10e-1, NMAX)
    N_j_list = 1:20

    """Parameters"""
    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx = 2
    const Ny = 2
    const Nz = 2  # number of arrays
    const N = Nx * Ny * Nz
    const M = 1 # Number of excitations
    const μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:N]
    const γ_e = [1e-2 for i = 1:N]

    d = 0.24444
    L = 0.62222
    Delt = 0.0
    delt = 0.0
    d_1 = d
    d_2 = d + delt

    # Function for computing
    function eigenvalues_var(N_j)

        N = N_j * N_j * Nz
        μ = [(i < 0) ? [0, 0, 1.0] : [1.0, 0.0im, 0.0] for i = 1:N]
        γ_e = [1e-2 for i = 1:N]
        δ_S = [(ind < N_j * N_j) ? -0.5*Delt : 0.5*Delt for ind = 1:N]

        pos_1 = AtomicArrays.geometry_module.rectangle(d_1, d_1; Nx=N_j, Ny=N_j,
            position_0=[-(N_j - 1) * d_1 / 2,
                -(N_j - 1) * d_1 / 2,
                -L / 2])
        pos_2 = AtomicArrays.geometry_module.rectangle(d_2, d_2; Nx=N_j, Ny=N_j,
            position_0=[-(N_j - 1) * d_2 / 2,
                -(N_j - 1) * d_2 / 2,
                L / 2])
        pos = vcat(pos_1, pos_2)
        S = SpinCollection(pos, μ; gammas=γ_e, deltas=δ_S)


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

        # Min/max eigenstates and eigenvalues
        s_ind_max = findmax(γ)[2]
        s_ind_min = sortperm(γ)[1]
        γ_min = γ[s_ind_min]
        γ_max = γ[s_ind_max]
        ψ_min = states[s_ind_min]
        ψ_max = states[s_ind_max]

        return [γ_min, γ_max, ψ_min, ψ_max]
    end

    # Create collection of parameters
    arg_list = collect(N_j_list)
end

results_vec = @showprogress pmap(arg_list) do x
    eigenvalues_var(x...)
end;


"Separate results"

begin
    gam_min = zeros(length(arg_list));
    gam_max = zeros(length(arg_list));
    psi_min = []
    psi_max = []
    for j in eachindex(arg_list)
        gam_min[j] = results_vec[j][1]
        gam_max[j] = results_vec[j][2]
        append!(psi_min, [results_vec[j][3]])
        append!(psi_max, [results_vec[j][4]])
    end
    "separating finished"
end


"Plotting"

function gammas_plot()
    GLMakie.activate!()
    f = Figure(resolution=(800, 400))

    N_a = Nz*N_j_list[end]^2

    # LsqFit
    m(t, p) = log10(γ_e[1]) .- p[1] * log10.(t)
    p0 = [0.5]
    fit = curve_fit(m, N_j_list.^2 .* Nz, log10.(gam_min), p0)
    # fit parameters and confidence interval
    p = fit.param
    print(p)
    confidence_interval(fit, 0.1)

    Axis(f[1, 1],
         xlabel=L"Number of atoms$$",
         ylabel=L"$\gamma_D$",
         xlabelsize = 28, ylabelsize = 28,
         titlesize = 30,
         xticklabelsize = 22,
         yticklabelsize = 22,
         # xscale = log10,
         yscale = log10,
         )
    scatter!(f[1, 1], N_j_list.^2 .* Nz, gam_min,
             color=:red, markersize=10)
    lines!(f[1, 1], 1:0.1:N_a, 10.0.^m(1:0.1:N_a, fit.param),
             color=:black, linewidth=4)
    # scatter!(f[1, 1], N_j_list.^2 .* Nz, gam_max,
    #          color=:blue, markersize=10)
    # save((PATH_FIGS * "gammaD_" * LAT_TYPE *
    #                 "_N_opt10.pdf"), f) # here, you save your figure.
    return f
end

gammas_plot()
