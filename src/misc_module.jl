module misc_module


"""
    misc_module.path()

# Output: 
* PATH_FIGS
* PATH_DATA
"""
function path()
    home = homedir()
    if home == "C:\\Users\\nnefedkin"
        PATH_FIGS = "D:/nnefedkin/Google_Drive/Work/In process/Projects/Collective_effects_QMS/Figures/two_arrays/forward_scattering/"
        PATH_DATA = "D:/nnefedkin/Google_Drive/Work/In process/Projects/Collective_effects_QMS/Data/data_2arrays_mpc_mf/"
    elseif home == "/home/nikita"
        PATH_FIGS = "/home/nikita/Documents/Work/Projects/two_arrays/Figs/"
        PATH_DATA = "/home/nikita/Documents/Work/Projects/two_arrays/Data/"
    elseif home == "/Users/jimi"
        PATH_FIGS = "/Users/jimi/Google Drive/Work/In process/Projects/Collective_effects_QMS/Figures/two_arrays/forward_scattering/"
        PATH_DATA = "/Users/jimi/Google Drive/Work/In process/Projects/Collective_effects_QMS/Data/data_2arrays_mpc_mf/" 
    end
    return PATH_FIGS, PATH_DATA
end

end #module