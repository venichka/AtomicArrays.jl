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
const EMField = AtomicArrays.field_module.EMField
const effective_constants = AtomicArrays.effective_interaction_module.effective_constants

import EllipsisNotation: Ellipsis
const .. = Ellipsis()


"Load data"

"""
* `args`: `N_x`, `N_2`, `lat_type`, `eq_type`, `dim_vars`, `path_data`
* `output`: `filename`
"""
function filename_create(args)
    N_x, N_y, lat_type, eq_type, dim_vars, path_data = args
    part_name = (string(dim_vars) * "D" * "_" * lat_type * "_" * string(N_x)
                    * "x" * string(N_y) * "_" * eq_type * ".h5")
    filename_fs = ("fs" * part_name )
    filename_fs_1a = ("fs_1a_" * part_name )
    @assert isfile(path_data * filename_fs) "There is no file $filename_fs in the directory $path_data"
    @assert isfile(path_data * filename_fs_1a) "There is no file $filename_fs_1a in the directory $path_data"
    return filename_fs, filename_fs_1a
end


"""
* `args`: `path_data::String, file::String
* `output`: `dict`, `dict_1a`
"""
function load_dict(path_data::String, file_name::String, file_name_1a::String)
    dict = load(path_data * file_name)
    dict_1a = load(path_data * file_name_1a)
    return dict, dict_1a
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

    const Nx = 10
    const Ny = 10
    const Nz = 2
    const N = Nx*Ny*Nz

    const LAT_TYPE = "freq"
    const EQ_TYPE = "mf"
    const DIM_VARS = 1

    const NMAX = 1000

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

end

"Load dicts and interpolate"

begin
    # File names
    args_files = [Nx, Ny, LAT_TYPE, EQ_TYPE, DIM_VARS, PATH_DATA]
    const FILE_FS, FILE_1A = filename_create(args_files)

    # Loading data
    dict_fs, dict_1a = load_dict(PATH_DATA, FILE_FS, FILE_1A)
    print(keys(dict_fs), "\n")
    dict_fs["order"]
end

begin
    # Interpolation
    # Linear
    fs_tot = LinearInterpolation((dict_fs["E"], [0, 1]), dict_fs["sigma_tot_un"])
    fs_tot_1a = LinearInterpolation((dict_1a["E"], [0, 1]), dict_1a["sigma_tot_1a"])

    # Variables
    E = range(dict_fs["E"][1], dict_fs["E"][end], NMAX)
    "Interpolation's done"
end


"""Plotting"""

function lin_E_var()

    CairoMakie.activate!()
    # GLMakie.activate!()

    f = Figure(resolution=(600,300))

    # Calculating functions to plot
    obj = AtomicArrays.field_module.objective(fs_tot(E, 0) ./ fs_tot_1a(E, 0),
                                              fs_tot(E, 1) ./ fs_tot_1a(E, 1))
    Om_R = norm(μ_0) .* E ./ γ_0

    # Plot
    ax1 = Axis(f[1, 1],# title=L"Direction: $0$",
            xlabel=L"|\Omega_R| / \gamma_0",
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
            color = :black, linewidth=2, linestyle = :dash, label=L"$M / \sigma_\mathrm{tot}^0$")

    axislegend(ax1, position=:lt)

    save((PATH_FIGS * "fsE_"*string(Nx)*"x"*string(Ny)*"_RL_"*
        LAT_TYPE*"_"*EQ_TYPE*".pdf"), f) # here, you save your figure.

    return f
end


function misc_lin_E_var()

    CairoMakie.activate!()
    # GLMakie.activate!()

    f = Figure(resolution=(600,300))

    # Calculating functions to plot
    obj = AtomicArrays.field_module.objective(fs_tot(E, 0),
                                              fs_tot(E, 1))
    Om_R = norm(μ_0) .* E ./ γ_0

    # Plot
    ax1 = Axis(f[1, 1],# title=L"Direction: $0$",
            xlabel=L"|\Omega_R| / \gamma_0",
            ylabel=L"\sigma_\mathrm{tot}^{f,b}",
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

    axislegend(ax1, position=:rt)

    save((PATH_FIGS * "fsE_"*string(Nx)*"x"*string(Ny)*"_RL_"*
        LAT_TYPE*"_"*EQ_TYPE*"_un.pdf"), f) # here, you save your figure.

    return f
end

lin_E_var()

misc_lin_E_var()
