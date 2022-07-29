module geometry_module


"""
    geometry_module.rectangle(a, b; Nx=2, Ny=2, position_0=[0,0,0])
Positions of spins on a rectangular lattice in the xy-plane.
The lattice starts at the origin and continues into positive x and y direction.
# Arguments
* `a`: Spin-spin distance in x-direction.
* `b`: Spin-spin distance in y-direction.
* `Nx=2`: Number of spins into x direction.
* `Ny=2`: Number of spins into y direction.
* `position_0=[0,0,0]`: Position of the left bottom corner.
"""
rectangle(a::S, b::T; Nx::Int=2, Ny::Int=2, position_0::Vector{U}=[0.0,0.0,0.0]) where {S<:Real,T<:Real,U<:Real} = vec([[i*a+position_0[1],j*b+position_0[2],position_0[3]] for i=0:Nx-1, j=0:Ny-1])


"""
    geometry_module.dimer_square(a, b; position_0=[0.0,0.0,0.0])
Positions of spins on a dimer lattice in the xy-plane.
# Arguments
* `a`: Lattice period 
* `b`: Sub-lattice period
* `Nx=2`: Number of spins into x direction.
* `Ny=2`: Number of spins into y direction.
* `position_0=[0,0,0]`: Position of the left bottom corner.

         o
     ^   O 
     |
   a |   o  | 
     |   O  | b
     v  ...
"""
function dimer_square(a::S, b::T; Nx::Int=4, Ny::Int=4, position_0::Vector{U}=[0.0,0.0,0.0]) where {S<:Real,T<:Real,U<:Real}
    positions = Vector{float(T)}[]
    for ix=0:Nx-1
        for iy=0:Ny-1
            push!(positions, 
                  [((iseven(ix)) ? (ix÷2)*a : ((ix-1)÷2)*a+b) + position_0[1], 
                   ((iseven(iy)) ? (iy÷2)*a : ((iy-1)÷2)*a+b) + position_0[2], 
                   0 + position_0[3]])
        end
    end
    return positions
end


"""
    geometry_module.dimer_square_1(a_1, a_2; position_0=[0.0,0.0,0.0])
Positions of spins on a dimer lattice in the xy-plane.
a_1 + a_2 = a -- the whole lattice period
# Arguments
* `a_1`: Sub-lattice 1 period 
* `a_2`: Sub-lattice 2 period
* `Nx=2`: Number of spins into x direction.
* `Ny=2`: Number of spins into y direction.
* `position_0=[0,0,0]`: Position of the left bottom corner

        o
        O  | 
           | a_2
  a_1 | o  | 
      | O   
       ...
"""
function dimer_square_1(a_1::S, a_2::T; Nx::Int=4, Ny::Int=4, position_0::Vector{U}=[0.0,0.0,0.0]) where {S<:Real,T<:Real,U<:Real}
    positions = Vector{float(T)}[]
    for ix=1:Nx
        for iy=1:Ny
            push!(positions, 
                 [ix ÷ 2 * a_1 + (ix -1) ÷ 2 * a_2 + position_0[1], 
                  iy ÷ 2 * a_1 + (iy -1) ÷ 2 * a_2 + position_0[2], 
                  position_0[3]])
        end
    end
    return positions
end


end #module