import matplotlib
matplotlib.use('Qt5Agg')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import kwave.data
from scipy.io import savemat
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

# Script options
debug_plot = 0
sim_data_base_path = r"C:\Users\chris\Desktop\Simulations_daten"

# make new folder with current date
info = "test_to_be_deleted"
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
sim_path = os.path.join(sim_data_base_path, timestamp + "_" + info)
if not os.path.exists(sim_path):
    os.makedirs(sim_path)

# materials
# ToDo: write dict with all parameters to also save it and save script also to the folder
# water
c_water = 1500
rho_water = 1000    # unit kg/m^3 (SI base units)

# pmma
c_pmma = 2780
rho_pmma = 1185     # unit kg/m^3 (SI base units)

# DEFINE LITERALS
f_0 = 1e6

# source signal
source_f0 = f_0
source_amp = 1e6
source_cycles = 5
source_focus = 20e-3

# transducer position
transducer_translation = kwave.data.Vector([-10e-3, 0, 0])
transducer_diameter = 10e-3
transducer_focus_pos = [20e-3,0,0]          # only used to define the angle of the transducer
transducer_rotation = kwave.data.Vector([0, 0, 0])

# physical domain for simulation and simulation parameters
grid_size_x = 40e-3
grid_size_y = 40e-3
grid_size_z = 40e-3
ppw = 3
t_end = 2*np.max([grid_size_x, grid_size_y, grid_size_z])/c_water
cfl = 0.3

# GRID
dx = c_water / (ppw * source_f0)             # attention: for dx the lowest speed of sound in the domain has to considered
Nx = round(grid_size_x / dx)
Ny = round(grid_size_y / dx)
Nz = round(grid_size_z / dx)
kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
kgrid.makeTime(c_water, cfl, t_end)

# SOURCE
karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
karray.add_disc_element(position=[0, 0, 0], diameter=transducer_diameter, focus_pos=transducer_focus_pos)
karray.set_array_position(transducer_translation, transducer_rotation)
source_sig = source_amp * tone_burst(1 / kgrid.dt, source_f0, source_cycles)
if debug_plot:
    plt.plot(source_sig)

source = kSource()
source.p_mask = karray.get_array_binary_mask(kgrid)
source.p = karray.get_distributed_source_signal(kgrid, source_sig)
if debug_plot:
    voxel_plot(np.single(source.p_mask))

# MEDIUM
medium = kWaveMedium(sound_speed=c_water, density=rho_water)

plate_thickness = 2*c_pmma/f_0
down_side_plate_idx = round(plate_thickness/dx)

sound_speed = np.ones((Nx, Ny, Nz))*c_water
sound_speed[Nx//2:Nx//2+down_side_plate_idx,:,:] = c_pmma

density = np.ones((Nx, Ny, Nz))*rho_water
density[Nx//2:Nx//2+down_side_plate_idx,:,:] = rho_pmma

medium.sound_speed = sound_speed
medium.density = density

# SENSOR: 2 masks for recording the signals
sensor_x_z_plane_mask = np.zeros((Nx, Ny, Nz))
sensor_x_z_plane_mask[:, Ny // 2, :] = 1
sensor_x_z_plane_mask = kSensor(sensor_x_z_plane_mask, record=['p_max', 'p'])

sensor_transducer_mask = np.zeros((Nx, Ny, Nz))
sensor_transducer_mask = source.p_mask
sensor_transducer_mask = kSensor(sensor_transducer_mask, record=['p_max', 'p'])

if debug_plot:
    voxel_plot(np.single(sensor.mask))

# Simulation
simulation_options = SimulationOptions(
    pml_auto=True,
    pml_inside=False,
    save_to_disk=True,
    data_cast='single',
)
execution_options = SimulationExecutionOptions(is_gpu_simulation=False)
sensor_data = kspaceFirstOrder3DC(kgrid=kgrid, medium=medium, source=source, sensor=sensor_transducer_mask,
                                  simulation_options=simulation_options, execution_options=execution_options)
rx_signal = np.mean(sensor_data["p"], axis=-1)

# save received data as .mat file
rx_signal_dict = {"rx_signal_mat": rx_signal, "label": "received signal"}
savemat(os.path.join(sim_path, "rx_signal"), rx_signal_dict)

# save as pickle file for future use in python
file = open(os.path.join(sim_path, 'rx_signal_pickle'), 'wb')
pickle.dump(rx_signal, file)
file.close()

# save plot of signal
fig = plt.figure()
plt.plot(rx_signal)
plt.xlabel('samples')
plt.ylabel('amplitude in a.u.')
plt.title('received signal')
fig.savefig(os.path.join(sim_path, "rx_signal_plot"))

# Save plot of setup
fig, axs = plt.subplots(3, 3)
fig.suptitle('Central Planes')
fig.set_figwidth(15)
fig.set_figheight(15)

##### SoS #####
vmin = 1000
vmax = 3000
row = 0
col = 0
pos = axs[row,col].imshow(medium.sound_speed[:,Ny//2,:], vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
axs[row,col].set_xlabel('x/mm')
axs[row,col].set_ylabel('z/mm')
axs[row,col].set_title("x-z-plane")
cbar = fig.colorbar(pos, ax=axs[row,col])
cbar.set_label("SoS / m/s")

row = 1
col = 0
pos = axs[row,col].imshow(medium.sound_speed[:,:,Nz//2], vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_y/2*1000,grid_size_y/2*1000])
axs[row,col].set_xlabel('x/mm')
axs[row,col].set_ylabel('y/mm')
axs[row,col].set_title("x-y-plane")
cbar = fig.colorbar(pos, ax=axs[row,col])
cbar.set_label("SoS / m/s")

row = 2
col = 0
pos = axs[row,col].imshow(medium.sound_speed[Nx//2,:,:], vmin = vmin, vmax=vmax, extent= [-grid_size_y/2*1000,grid_size_y/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
axs[row,col].set_xlabel('y/mm')
axs[row,col].set_ylabel('z/mm')
axs[row,col].set_title("y-z-plane")
cbar = fig.colorbar(pos, ax=axs[row,col])
cbar.set_label("SoS / m/s")

##### Density #####
vmin = 500
vmax = 1500
row = 0
col = 1
pos = axs[row,col].imshow(medium.density[:,Ny//2,:], vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
axs[row,col].set_xlabel('x/mm')
axs[row,col].set_ylabel('z/mm')
axs[row,col].set_title("x-z-plane")
cbar = fig.colorbar(pos, ax=axs[row,col])
cbar.set_label(r"$\rho$ / kg/m^3")

row = 1
col = 1
pos = axs[row,col].imshow(medium.density[:,:,Nz//2], vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_y/2*1000,grid_size_y/2*1000])
axs[row,col].set_xlabel('x/mm')
axs[row,col].set_ylabel('y/mm')
axs[row,col].set_title("x-y-plane")
cbar = fig.colorbar(pos, ax=axs[row,col])
cbar.set_label(r"$\rho$ / kg/m^3")

row = 2
col = 1
pos = axs[row,col].imshow(medium.density[Nx//2,:,:], vmin = vmin, vmax=vmax, extent= [-grid_size_y/2*1000,grid_size_y/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
axs[row,col].set_xlabel('y/mm')
axs[row,col].set_ylabel('z/mm')
axs[row,col].set_title("y-z-plane")
cbar = fig.colorbar(pos, ax=axs[row,col])
cbar.set_label(r"$\rho$ / kg/m^3")

#### transducer ####
vmin = 0
vmax = 1
row = 0
col = 2
pos = axs[row,col].imshow(source.p_mask[:,Ny//2,:], vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
axs[row,col].set_xlabel('x/mm')
axs[row,col].set_ylabel('z/mm')
axs[row,col].set_title("Transducer,x-z-plane")

row = 1
col = 2
pos = axs[row,col].imshow(source.p_mask[:,:,Nz//2], vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_y/2*1000,grid_size_y/2*1000])
axs[row,col].set_xlabel('x/mm')
axs[row,col].set_ylabel('y/mm')
axs[row,col].set_title("Transducer,x-y-plane")

row = 2
col = 2
pos = axs[row,col].imshow(source.p_mask[Nx//2,:,:], vmin = vmin, vmax=vmax, extent= [-grid_size_y/2*1000,grid_size_y/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
axs[row,col].set_xlabel('y/mm')
axs[row,col].set_ylabel('z/mm')
axs[row,col].set_title("Transducer,y-z-plane")

fig.tight_layout()
fig.savefig(os.path.join(sim_path, "simulation_setup"))

# save script with current parameters



