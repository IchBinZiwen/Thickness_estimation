import matplotlib
matplotlib.use('Qt5Agg')
import shutil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import kwave.data
from scipy.io import savemat
from scipy import signal
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


if __name__ == '__main__':

    plate_thickness_list_in_lambda = np.arange(1,10,1)

    for plate_thickness_in_lambda in plate_thickness_list_in_lambda:
        # Script options
        debug_plot = 0
        sim_data_base_path = r"C:\Users\chris\Desktop\Simulations_daten"

        # make new folder with current date
        info = "pmma_focused_thickness_" + str(plate_thickness_in_lambda) + "_lambda"
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        sim_path = os.path.join(sim_data_base_path, timestamp + "_" + info)
        if not os.path.exists(sim_path):
            os.makedirs(sim_path)

        # materials
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
        source_cycles = 7
        source_focus = 20e-3

        # transducer position
        transducer_translation = kwave.data.Vector([-20e-3, 0, 0])
        transducer_diameter = 15e-3
        transducer_focus_pos = [20e-3,0,0]          # only used to define the angle of the transducer
        transducer_rotation = kwave.data.Vector([0, 0, 0])

        # experimental setup
        plate_thickness = plate_thickness_in_lambda*c_pmma/f_0

        # physical domain for simulation and simulation parameters
        grid_size_x = 100e-3
        grid_size_y = 20e-3
        grid_size_z = 20e-3
        ppw = 5
        t_end = 2*np.max([grid_size_x, grid_size_y, grid_size_z])/c_water
        cfl = 0.3

        # GRID
        dx = c_water / (ppw * source_f0)             # attention: for dx the lowest speed of sound in the domain has to considered
        Nx = round(grid_size_x / dx)
        Ny = round(grid_size_y / dx)
        Nz = round(grid_size_z / dx)
        kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
        kgrid.makeTime(c_water, cfl, t_end)

        # make dict of all parameters
        params = {
            "source_f0": f_0,
            "source_cycles": source_cycles,
            "transducer_translation": transducer_translation,
            "transducer_diameter": transducer_diameter,
            "transducer_focus_pos": transducer_focus_pos,
            "transducer_rotation": transducer_rotation,
            "plate_thickness": plate_thickness
        }

        # SOURCE
        karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
        karray.add_bowl_element(position=tuple(np.array(transducer_translation)), radius=transducer_focus_pos[0], diameter=transducer_diameter, focus_pos=transducer_focus_pos)
        # karray.set_array_position(transducer_translation, transducer_rotation)
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
        down_side_plate_idx = round(plate_thickness/dx)     # plate surface always at x=0

        sound_speed = np.ones((Nx, Ny, Nz))*c_water
        sound_speed[Nx//2:Nx//2+down_side_plate_idx,:,:] = c_pmma

        density = np.ones((Nx, Ny, Nz))*rho_water
        density[Nx//2:Nx//2+down_side_plate_idx,:,:] = rho_pmma

        medium.sound_speed = sound_speed
        medium.density = density

        # masks for recording the signals
        sensor_transducer_mask = np.zeros((Nx, Ny, Nz))
        sensor_transducer_mask = source.p_mask
        sensor_transducer = kSensor(sensor_transducer_mask, record=['p_max', 'p'])

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
        sensor_data = kspaceFirstOrder3DC(kgrid=kgrid, medium=medium, source=source, sensor=sensor_transducer,
                                          simulation_options=simulation_options, execution_options=execution_options)       # ATTENTION: this somehow modifies existing objects like the sensor or source object under certain circumstances and changes the discretization
        rx_signal = np.mean(sensor_data["p"], axis=-1)

        #########################################################################################################
        #########################################################################################################
        ############################ Plotting and Storing results ###############################################
        #########################################################################################################
        #########################################################################################################

        # save received data as .mat file
        rx_signal_dict = {"rx_signal_mat": rx_signal, "label": "received signal"}
        savemat(os.path.join(sim_path, "rx_signal_mat"), rx_signal_dict)

        # save received data as .mat file
        params_dict = {"params": params, "label": "Parameters"}
        savemat(os.path.join(sim_path, "params_mat"), params_dict)

        # save as pickle file for future use in python
        file = open(os.path.join(sim_path, 'rx_signal_pickle'), 'wb')
        pickle.dump(rx_signal, file)
        file.close()

        # save as pickle file for future use in python
        file = open(os.path.join(sim_path, 'params_pickle'), 'wb')
        pickle.dump(params, file)
        file.close()

        # save current script to folder
        src = __file__
        dst = os.path.join(sim_path, "used_script.py")
        shutil.copyfile(src, dst)


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
        pos = axs[row,col].imshow(medium.sound_speed[:,Ny//2,:].T, vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
        axs[row,col].set_xlabel('x/mm')
        axs[row,col].set_ylabel('z/mm')
        axs[row,col].set_title("x-z-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])
        cbar.set_label("SoS / m/s")

        row = 1
        col = 0
        pos = axs[row,col].imshow(medium.sound_speed[:,:,Nz//2].T, vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_y/2*1000,grid_size_y/2*1000])
        axs[row,col].set_xlabel('x/mm')
        axs[row,col].set_ylabel('y/mm')
        axs[row,col].set_title("x-y-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])
        cbar.set_label("SoS / m/s")

        row = 2
        col = 0
        pos = axs[row,col].imshow(medium.sound_speed[Nx//2,:,:].T, vmin = vmin, vmax=vmax, extent= [-grid_size_y/2*1000,grid_size_y/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
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
        pos = axs[row,col].imshow(medium.density[:,Ny//2,:].T, vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
        axs[row,col].set_xlabel('x/mm')
        axs[row,col].set_ylabel('z/mm')
        axs[row,col].set_title("x-z-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])
        cbar.set_label(r"$\rho$ / kg/m^3")

        row = 1
        col = 1
        pos = axs[row,col].imshow(medium.density[:,:,Nz//2].T, vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_y/2*1000,grid_size_y/2*1000])
        axs[row,col].set_xlabel('x/mm')
        axs[row,col].set_ylabel('y/mm')
        axs[row,col].set_title("x-y-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])
        cbar.set_label(r"$\rho$ / kg/m^3")

        row = 2
        col = 1
        pos = axs[row,col].imshow(medium.density[Nx//2,:,:].T, vmin = vmin, vmax=vmax, extent= [-grid_size_y/2*1000,grid_size_y/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
        axs[row,col].set_xlabel('y/mm')
        axs[row,col].set_ylabel('z/mm')
        axs[row,col].set_title("y-z-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])
        cbar.set_label(r"$\rho$ / kg/m^3")

        #### transducer ####
        vmin = -1
        vmax = 2
        row = 0
        col = 2
        pos = axs[row,col].imshow(source.p_mask[:,np.shape(source.p_mask)[1]//2,:].T, vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000], interpolation='nearest')
        axs[row,col].set_xlabel('x/mm')
        axs[row,col].set_ylabel('z/mm')
        axs[row,col].set_title("Transducer,x-z-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])

        row = 1
        col = 2
        pos = axs[row,col].imshow(source.p_mask[:,:,np.shape(source.p_mask)[2]//2].T, vmin = vmin, vmax=vmax, extent= [-grid_size_x/2*1000,grid_size_x/2*1000,-grid_size_y/2*1000,grid_size_y/2*1000])
        axs[row,col].set_xlabel('x/mm')
        axs[row,col].set_ylabel('y/mm')
        axs[row,col].set_title("Transducer,x-y-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])

        row = 2
        col = 2
        pos = axs[row,col].imshow(source.p_mask[np.shape(source.p_mask)[0]//2,:,:].T, vmin = vmin, vmax=vmax, extent= [-grid_size_y/2*1000,grid_size_y/2*1000,-grid_size_z/2*1000,grid_size_z/2*1000])
        axs[row,col].set_xlabel('y/mm')
        axs[row,col].set_ylabel('z/mm')
        axs[row,col].set_title("Transducer,y-z-plane")
        cbar = fig.colorbar(pos, ax=axs[row,col])

        fig.tight_layout()
        fig.savefig(os.path.join(sim_path, "simulation_setup"), dpi=fig.dpi)




