import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import kwave.data
#from kwave import kWaveGrid, SimulationOptions, kWaveMedium
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationOptions
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.plot import voxel_plot
from kwave.utils.signals import tone_burst

debug_plot = 0

# DEFINE LITERALS, CK: here we define siome useful parameters to describe the simulation setup, the signals and some discretization parameters
c0 = 1500                   # this is the speed of sound in the medium, this parameter is important, since it will be used to calculate the discretization
rho0 = 1000                 # this is the density of the medium to be considered
source_f0 = 1e6             # this is the central frequency of the TX signal
source_amp = 1e6            # this describes the amplitude of the TX signal as sound pressure
source_cycles = 3           # number of cycles in the tone burst of the TX signals
source_focus = 20e-3        # this is typically used to define the orientation of transducer, here we don't use it
# the following 4 parameters are typically used to describe a linear array transducer, we don't used them in this script
element_num = 15
element_width = 1e-3
element_length = 10e-3
element_pitch = 2e-3

translation = kwave.data.Vector([5e-3, 0, 8e-3])    # this describes a 3D translation of the TX transducer after initial description
rotation = kwave.data.Vector([0, 0, 0])             # this describes 3D rotation of the transducer, we don't use it here, kwave.data.Vector([0, 20, 0])

# the following 3 parameter describe the physical dimension of the simulation domain in SI unit meter
grid_size_x = 40e-3
grid_size_y = 40e-3
grid_size_z = 40e-3
ppw = 5                 # this parameter describes the discretization resolution in points per wavelength, initially was set to 3; but 3 results in numerical artefacts
t_end = 35e-6           # this defines the length of the simulation in time
cfl = 0.3               # this the Courant-Friedrichs-Lewy number --> important for discretization, should be smaller than 1!!!! typically values between 0.15 and 0.5, has to be adapted if simulation is not converging

# GRID
dx = c0 / (ppw * source_f0)             # calculation of spatial discretization!
Nx = round(grid_size_x / dx)            # generate the numerical grid according to dx and the physical dimensions
Ny = round(grid_size_y / dx)
Nz = round(grid_size_z / dx)
kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])       # put it into an Object from kwave
kgrid.makeTime(c0, cfl, t_end)                      # discretization in time, generate time vector

# SOURCE
# define transducer as karray in the physical domain, see http://www.k-wave.org/documentation/example_at_array_as_source.php
# this description is general/generic and holds no information on discretization --> can be reused in different simulations!
karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)                         # generate a k-wave array as TX transducer (generic)
karray.add_disc_element(position = [-15e-3,0,0], diameter = 10e-3, focus_pos = [20e-3,0,0]) # add something to the array, here: only one disc/circular element

source_sig = source_amp * tone_burst(1 / kgrid.dt, source_f0, source_cycles)        # define the TX signal
if debug_plot:
    plt.plot(source_sig)

source = kSource()                                                 # just an object which holds different information on the source
source.p_mask = karray.get_array_binary_mask(kgrid)                # we need to adapt the array (physically described) onto the numerical grid
source.p = karray.get_distributed_source_signal(kgrid, source_sig)      # then we need to assign signal to every point, here we could also add delays
# for the source we could in theory also define a particle velocity instead of pressure


if debug_plot:
    voxel_plot(np.single(source.p_mask))

# MEDIUM
medium = kWaveMedium(sound_speed=c0, density=rho0)

sound_speed = np.ones((Nx, Ny, Nz))*c0              # define grid for the speed of sound of medium
sound_speed[Nx//2:,:,:] = c0                        # make half of the domain with different speed of sound

density = np.ones((Nx, Ny, Nz))*rho0                # define grid for density
density[3*Nx//5:4*Nx//5,:,:] = rho0                 # part of the grid can have higher density, try 3*rho0

medium.sound_speed=sound_speed                      # adding speed of sound and density to the medium
medium.density = density


# SENSOR
sensor_mask = np.zeros((Nx, Ny, Nz))                # sensor as the same grid as the simulation domain
sensor_mask[:, Ny // 2, :] = 1                      # record als signal in a plane, here x-z-plane at y = Ny/2
# sensor_mask = source.p_mask                         # Alternative: record the signal for the same mask as sensor --> adding up all signals gives the pulse-echo sensor signal
sensor = kSensor(sensor_mask, record=['p_max', 'p'])    # bring it into a ksensor object, define what will be recorded (here only pressure)
if debug_plot:
    voxel_plot(np.single(sensor.mask))
# SIMULATION
simulation_options = SimulationOptions(
    pml_auto=True,
    pml_inside=False,
    save_to_disk=True,
    data_cast='single',
)                                                   # setup important simulation options

execution_options = SimulationExecutionOptions(is_gpu_simulation=False)         # here you can choose whether to use GPU or something else

sensor_data = kspaceFirstOrder3DC(kgrid=kgrid, medium=medium, source=source, sensor=sensor,
                                  simulation_options=simulation_options, execution_options=execution_options)       # this is the simulation calculation it self


# PLOTTING, VISUALISATION, RESULTS
p_max = np.reshape(sensor_data['p_max'], (Nx, Nz), order='F')           # here we get the results as p_max (maximum pressure at all sensor points)

# VISUALISATION
plt.figure()
plt.imshow(1e-6 * p_max, extent=[1e3 * kgrid.x_vec[0][0], 1e3 * kgrid.x_vec[-1][0], 1e3 * kgrid.z_vec[0][0],
                                 1e3 * kgrid.z_vec[-1][0]], aspect='auto')
plt.xlabel('z-position [mm]')
plt.ylabel('x-position [mm]')
plt.title('Pressure Field')
plt.colorbar(label='[MPa]')
plt.show()


if 0:
    import os

    plt.figure()
    path = r"C:\Users\chris\Desktop\test"

    p_max = np.reshape(sensor_data['p'][0,:], (Nx, Nz), order='F')
    plt.imshow(1e-6 * p_max, extent=[1e3 * kgrid.x_vec[0][0], 1e3 * kgrid.x_vec[-1][0], 1e3 * kgrid.z_vec[0][0],
                                         1e3 * kgrid.z_vec[-1][0]], aspect='auto')
    plt.xlabel('z-position [mm]')
    plt.ylabel('x-position [mm]')
    plt.title('Pressure Field')
    plt.colorbar(label='[MPa]')

    for i in range(1,210):
        p_max = np.reshape(sensor_data['p'][i,:], (Nx, Nz), order='F')
        plt.imshow(1e-6 * p_max, extent=[1e3 * kgrid.x_vec[0][0], 1e3 * kgrid.x_vec[-1][0], 1e3 * kgrid.z_vec[0][0],
                                         1e3 * kgrid.z_vec[-1][0]], aspect='auto')
        plt.xlabel('z-position [mm]')
        plt.ylabel('x-position [mm]')
        plt.title('Pressure Field')
        plt.colorbar(label='[MPa]')
        plt.savefig(os.path.join(path, str(i)))