begin
    if pwd()[end-14:end] == "AtomicArrays.jl"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
end
using Pkg
Pkg.activate(PATH_ENV)

using QuantumOptics
using FFTW
using BenchmarkTools, Interpolations
using LaTeXStrings
using CairoMakie, GLMakie
using LinearAlgebra, EllipsisNotation
using HDF5, FileIO
using Markdown

using Revise
using AtomicArrays
const EMField = AtomicArrays.field.EMField
const effective_constants = AtomicArrays.effective_interaction.effective_constants

import EllipsisNotation: Ellipsis
const .. = Ellipsis()


"Load data"

"""
* `args`: `N_1`, `N_2`, `eq_type`, `dim_vars`, `path_data`
* `output`: `filename`
"""
function filename_create(first_part_name, args)
    N_1, N_2, eq_type, dim_vars, path_data = args
    part_name = (string(dim_vars) * "D" * "_" * string(N_1)
                    * "x" * string(N_2) * "_" * eq_type * ".h5")
    filename = (first_part_name * part_name )
    @assert isfile(path_data * filename) "There is no file $filename in the directory $path_data"
    return filename
end


"""
* `args`: `path_data::String, file::String
* `output`: `dict`
"""
function load_dict(path_data::String, file_name::String)
    return load(path_data * file_name)
end


"Parameters"

begin

    """Parameters"""

    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0
    const γ_0 = 1e-2
    const μ_0 = [1, 0, 0]

    const Nx = 2
    const Ny = 2
    const Nz = 1
    const N = Nx*Ny*Nz

    const EQ_TYPE = "Q"
    const DIM_VARS = 1

    const NMAX = 1000

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc.path()

end


"Load dicts and interpolate"

begin
    # File names
    args_files = [Nx, Ny, EQ_TYPE, DIM_VARS, PATH_DATA]
    # const FILE_Q = filename_create("quantDict_Efreq", args_files)
    const FILE_Q = filename_create("quantDict", args_files)

    # Loading data
    dict_q = load_dict(PATH_DATA, FILE_Q)
    print(keys(dict_q), "\n")
    dict_q["order"]
end

begin
    # Interpolation
    var_str = "E"

    fs_tot = LinearInterpolation((dict_q[var_str], [0, 1]), dict_q["sigma_tot_un"])
    fs_tot_1a = LinearInterpolation((dict_q[var_str], [0, 1]), dict_q["sigma_tot_1a"])
    entropy = LinearInterpolation((dict_q[var_str], [0, 1]), dict_q["entropy"])
    c_D = LinearInterpolation((dict_q[var_str], [0, 1]), dict_q["population_D"])
    purity = LinearInterpolation((dict_q[var_str], [0, 1]), dict_q["purity"])

    # Variables
    var_str = range(dict_q[var_str][1], dict_q[var_str][end], NMAX)
    "Interpolation's done"
end


"""Plotting"""


function quant_results()

#     CairoMakie.activate!()
    GLMakie.activate!()

    f = Figure(resolution=(600,800))
    
    E = var_str
    # Calculating functions to plot
    obj = AtomicArrays.field.objective(fs_tot(E, 0) ./ fs_tot_1a(E, 0),
                                              fs_tot(E, 1) ./ fs_tot_1a(E, 1))
    Om_R = norm(μ_0) .* E ./ γ_0

    # Plot
    ax1 = Axis(f[1, 1],# title=L"Direction: $0$",
            # xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"\sigma_\mathrm{tot}^{f,b} / \sigma_\mathrm{tot}^0",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            # xminorticksvisible = true, xminorgridvisible = true,
            xscale = log10)
    lines!(f[1, 1], Om_R, fs_tot(E, 0) ./ fs_tot_1a(E, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[1, 1], Om_R, fs_tot(E, 1) ./ fs_tot_1a(E, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")
    lines!(f[1, 1], Om_R, obj,
            color = :black, linewidth=2, linestyle = :dash, label=L"$M/\sigma_\mathrm{tot}^0$")

    axislegend(ax1, position=:rc)

    ax2 = Axis(f[2, 1], #title=L"Direction: $\pi$",
            # xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"c_D",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            xscale = log10)
    lines!(f[2, 1], Om_R, c_D(E, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[2, 1], Om_R, c_D(E, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")

    axislegend(ax2, position=:rc)

    ax3 = Axis(f[3, 1], #title=L"Direction: $\pi$",
            xlabel=L"|\Omega_R| / \gamma_0",
            ylabel=L"Entropy$$",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            xscale = log10)
    lines!(f[3, 1], Om_R, entropy(E, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[3, 1], Om_R, entropy(E, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")

    axislegend(ax3, position=:rc)

    # Adding letters
    ga = f[1, 1] = GridLayout()
    gb = f[2, 1] = GridLayout()
    gc = f[3, 1] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)", "(c)"],
                               [ga, gb, gc])
        Label(layout[1, 1, TopLeft()], label,
              textsize = 24,
              font = "TeX Gyre Heros Bold",
              padding = (0, 20, 0, 0),
              halign = :right)
    end

    colsize!(f.layout, 1, Auto(1.))

#     save((PATH_FIGS * "quant_" * string(Nx) * "x" * string(Ny) * "_" *
#         EQ_TYPE * ".pdf"), f) # here, you save your figure.

    return f
end


function quant_results_freq()

    CairoMakie.activate!()
#     GLMakie.activate!()

    f = Figure(resolution=(600,800))
    
    w_L = var_str
    # Calculating functions to plot
    obj = AtomicArrays.field.objective(fs_tot(w_L, 0) ./ fs_tot_1a(w_L, 0),
                                              fs_tot(w_L, 1) ./ fs_tot_1a(w_L, 1))

    w_L0 = w_L / om_0

    # Plot
    ax1 = Axis(f[1, 1],# title=L"Direction: $0$",
            # xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"\sigma_\mathrm{tot}^{f,b} / \sigma_\mathrm{tot}^0",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            # xminorticksvisible = true, xminorgridvisible = true,
            )
    lines!(f[1, 1], w_L0, fs_tot(w_L, 0) ./ fs_tot_1a(w_L, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[1, 1], w_L0, fs_tot(w_L, 1) ./ fs_tot_1a(w_L, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")
    lines!(f[1, 1], w_L0, obj,
            color = :black, linewidth=2, linestyle = :dash, label=L"$M/\sigma_\mathrm{tot}^0$")

    axislegend(ax1, position=:rc)

    ax2 = Axis(f[2, 1], #title=L"Direction: $\pi$",
            # xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"c_D",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            )
    lines!(f[2, 1], w_L0, c_D(w_L, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[2, 1], w_L0, c_D(w_L, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")

    axislegend(ax2, position=:rc)

    ax3 = Axis(f[3, 1], #title=L"Direction: $\pi$",
            xlabel=L"\omega_0 / \omega_{1,2}",
            ylabel=L"Entropy$$",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            )
    lines!(f[3, 1], w_L0, entropy(w_L, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[3, 1], w_L0, entropy(w_L, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")

    axislegend(ax3, position=:rc)

    # Adding letters
    ga = f[1, 1] = GridLayout()
    gb = f[2, 1] = GridLayout()
    gc = f[3, 1] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)", "(c)"],
                               [ga, gb, gc])
        Label(layout[1, 1, TopLeft()], label,
              textsize = 24,
              font = "TeX Gyre Heros Bold",
              padding = (0, 20, 0, 0),
              halign = :right)
    end

    colsize!(f.layout, 1, Auto(1.))

    save((PATH_FIGS * "quant_freq_" * string(Nx) * "x" * string(Ny) * "_" *
        EQ_TYPE * ".pdf"), f) # here, you save your figure.

    return f
end



function misc_results()

    CairoMakie.activate!()
    # GLMakie.activate!()

    f = Figure(resolution=(600,540))

    E = var_str

    # Calculating functions to plot
    obj = AtomicArrays.field.objective(fs_tot(E, 0), fs_tot(E, 1))
    Om_R = norm(μ_0) .* E ./ γ_0

    # Plot
    ax1 = Axis(f[1, 1],# title=L"Direction: $0$",
            # xlabel=L"\Re \sigma / |\sigma|",
            ylabel=L"\sigma_\mathrm{tot}^{f, b}",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            # xminorticksvisible = true, xminorgridvisible = true,
            xscale = log10)
    lines!(f[1, 1], Om_R, fs_tot(E, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[1, 1], Om_R, fs_tot(E, 1),
            color = :blue, linewidth=2, label=L"Direction: $b$")
    lines!(f[1, 1], Om_R, obj,
            color = :black, linewidth=2, linestyle = :dash, label=L"$M$")

    axislegend(ax1, position=:rc)

    ax2 = Axis(f[2, 1], #title=L"Direction: $\pi$",
            xlabel=L"|\Omega_R| / \gamma_0",
            ylabel=L"Tr$(\rho^2)$",
            xlabelsize=28, ylabelsize=28,
            titlesize=30,
            xticklabelsize=22,
            yticklabelsize=22,
            xscale = log10)
    lines!(f[2, 1], Om_R, purity(E, 0),
            color = :red, linewidth=2, label=L"Direction: $f$")
    lines!(f[2, 1], Om_R, purity(E, 1),
            color = :blue, linewidth=2, label=L"Direction: $p$")

    axislegend(ax2, position=:rc)

    # Adding letters
    ga = f[1, 1] = GridLayout()
    gb = f[2, 1] = GridLayout()
    for (label, layout) in zip(["(a)", "(b)"],
                               [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
              textsize = 24,
              font = "TeX Gyre Heros Bold",
              padding = (0, 20, 0, 0),
              halign = :right)
    end

    colsize!(f.layout, 1, Auto(1.))

    save((PATH_FIGS * "quantMisc_" * string(Nx) * "x" * string(Ny) * "_" *
        EQ_TYPE * ".pdf"), f) # here, you save your figure.

    return f
end


quant_results()

quant_results_freq()

misc_results()
