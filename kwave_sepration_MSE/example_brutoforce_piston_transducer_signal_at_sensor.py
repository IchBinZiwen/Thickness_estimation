import matplotlib
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from scipy.optimize import minimize


def do_the_simulation(d1,d2):
    debug_plot = 0

    # DEFINE LITERALS, CK: here we define siome useful parameters to describe the simulation setup, the signals and some discretization parameters
    c0 = 1500  # this is the speed of sound in the medium, this parameter is important, since it will be used to calculate the discretization
    rho0 = 1000  # this is the density of the medium to be considered
    source_f0 = 1e6  # this is the central frequency of the TX signal
    source_amp = 1e6  # this describes the amplitude of the TX signal as sound pressure
    source_cycles = 3  # number of cycles in the tone burst of the TX signals
    source_focus = 20e-3  # this is typically used to define the orientation of transducer, here we don't use it
    # the following 4 parameters are typically used to describe a linear array transducer, we don't used them in this script
    element_num = 15
    element_width = 1e-3
    element_length = 10e-3
    element_pitch = 2e-3

    translation = kwave.data.Vector(
        [5e-3, 0, 8e-3])  # this describes a 3D translation of the TX transducer after initial description
    rotation = kwave.data.Vector(
        [0, 0, 0])  # this describes 3D rotation of the transducer, we don't use it here, kwave.data.Vector([0, 20, 0])

    # the following 3 parameter describe the physical dimension of the simulation domain in SI unit meter
    grid_size_x = 30e-3
    grid_size_y = 5e-3
    grid_size_z = 5e-3
    ppw = 10  # this parameter describes the discretization resolution in points per wavelength, initially was set to 3; but 3 results in numerical artefacts
    t_end = 35e-6  # this defines the length of the simulation in time
    cfl = 0.3  # this the Courant-Friedrichs-Lewy number --> important for discretization, should be smaller than 1!!!! typically values between 0.15 and 0.5, has to be adapted if simulation is not converging

    # GRID
    dx = c0 / (ppw * source_f0)  # calculation of spatial discretization!
    print('dx:',dx)
    Nx = round(grid_size_x / dx)  # generate the numerical grid according to dx and the physical dimensions
    Ny = round(grid_size_y / dx)
    Nz = round(grid_size_z / dx)
    print(Nx, Ny, Nz)
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])  # put it into an Object from kwave
    kgrid.makeTime(c0, cfl, t_end)  # discretization in time, generate time vector

    # SOURCE
    # define transducer as karray in the physical domain, see http://www.k-wave.org/documentation/example_at_array_as_source.php
    # this description is general/generic and holds no information on discretization --> can be reused in different simulations!
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)  # generate a k-wave array as TX transducer (generic)
    karray.add_disc_element(position=[-15e-3, 0, 0], diameter=10e-3,
                            focus_pos=[20e-3, 0, 0])  # add something to the array, here: only one disc/circular element

    source_sig = source_amp * tone_burst(1 / kgrid.dt, source_f0, source_cycles)  # define the TX signal
    if debug_plot:
        plt.plot(source_sig[0, :])

    source = kSource()  # just an object which holds different information on the source
    source.p_mask = karray.get_array_binary_mask(
        kgrid)  # we need to adapt the array (physically described) onto the numerical grid
    source.p = karray.get_distributed_source_signal(kgrid,
                                                    source_sig)  # then we need to assign signal to every point, here we could also add delays
    # for the source we could in theory also define a particle velocity instead of pressure

    if debug_plot:
        voxel_plot(np.single(source.p_mask))

    # MEDIUM
    medium = kWaveMedium(sound_speed=c0, density=rho0)

    sound_speed = np.ones((Nx, Ny, Nz)) * c0  # define grid for the speed of sound of medium
    sound_speed[Nx // 2:, :, :] = c0  # make half of the domain with different speed of sound

    density = np.ones((Nx, Ny, Nz)) * rho0  # define grid for density
    density[ Nx * d1 // 100: Nx * (d1+d2)// 100, :, :] = 3 * rho0  # part of the grid can have higher density

    medium.sound_speed = sound_speed  # adding speed of sound and density to the medium
    medium.density = density

    # SENSOR
    sensor_mask = np.zeros((Nx, Ny, Nz))  # sensor as the same grid as the simulation domain
    # sensor_mask[:, Ny // 2, :] = 1                      # record als signal in a plane, here x-z-plane at y = Ny/2
    sensor_mask = source.p_mask  # Alternative: record the signal for the same mask as sensor --> adding up all signals gives the pulse-echo sensor signal
    sensor = kSensor(sensor_mask, record=['p_max',
                                          'p'])  # bring it into a ksensor object, define what will be recorded (here only pressure)
    if debug_plot:
        voxel_plot(np.single(sensor.mask))
    # SIMULATION
    simulation_options = SimulationOptions(
        pml_auto=True,
        pml_inside=False,
        save_to_disk=True,
        data_cast='single',
    )  # setup important simulation options

    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=True)  # here you can choose whether to use GPU or something else

    sensor_data = kspaceFirstOrder3DC(kgrid=kgrid, medium=medium, source=source, sensor=sensor,
                                      simulation_options=simulation_options,
                                      execution_options=execution_options)  # this is the simulation calculation it self

    # PLOTTING, VISUALISATION, RESULTS
    p_sensor = np.sum(sensor_data['p'], -1)  # here we get the results as p_max (maximum pressure at all sensor points)

    # VISUALISATION
    t = np.squeeze(kgrid.t_array)
    p = p_sensor
    # plt.figure()
    # plt.plot(t, p)
    # plt.xlabel('time [s]')
    # plt.ylabel('Amplitude [a.u.]')
    # plt.title('signal at sensor')
    # plt.show()
    return t,p,d1,d2
def creat_the_datafile(new_data=0):
    # new_data = 0
    signal_dict = {}
    if new_data == 1:
        d_list = [2,3,4,5,6,8,10,13,16]
        for d2 in d_list:
            d1, d2 = 30, d2
            t, p, d1, d2 = do_the_simulation(d1, d2)
            signal_dict[f"data_{d2}"] = [t, p, d1, d2]
        print(signal_dict.keys())
        np.save('data.npy', signal_dict)
def seprate_the_waves(loaded_dict):
    dt = 3e-8  # time interval of samples. unit [s]
    wave_dict = {}
    for i in list(loaded_dict.keys())[:]:
        plt.plot(loaded_dict[i][0], loaded_dict[i][1])
        m_x = 1.34e-5
        b = 0.18e-5
        wave = [0, 0, 0]
        for k in [0, 1, 2]:
            thickness = 30e-3 * float(i[5:]) / 100
            rectangle = plt.Rectangle((m_x - b + k * thickness * 2 / 1500, -1e7), 2 * b, 2e7, color='red', fill=False)
            idx = int((m_x - b + k * thickness * 2 / 1500) // dt)
            idx2 = idx + int((2 * b // dt))
            # plt.plot([loaded_dict[i][0][idx],loaded_dict[i][0][idx2]],[1e8,1e8])
            # plt.gca().add_patch(rectangle)#

            wave_one = loaded_dict[i][1][idx:idx2]
            wave[k] = wave_one
            # plt.plot(loaded_dict[i][0], loaded_dict[i][1])
            # plt.plot(loaded_dict[i][0][idx:idx2],wave_one)
            # plt.show()
        # print(np.array(wave).shape)
        wave_dict[i] = np.array(wave)
    return wave_dict

def creat_matrix_A(waves,d,r):
    """y=Ax,use wave_dict to creat A"""
    A=np.zeros((len(r),3))
    l_wave= waves.shape[1]
    idx1=386
    dt = 3e-8
    c0 = 1500
    d_mm=d*30e-3/100
    idx2=int(idx1+1*2*d_mm/c0/dt)
    idx3=int(idx1+2*2*d_mm/c0/dt)
    A[idx1:idx1 + l_wave, 0] = waves[0,:]
    A[idx2:idx2 + l_wave, 1] = waves[1,:]
    A[idx3:idx3 + l_wave, 2] = waves[2,:]
    # plt.plot(A)
    # plt.plot(r)
    # plt.show()
    return A,d
def optimize_x(A, r):
    # 初始猜测值 x
    x_initial_guess = np.zeros(A.shape[1])
    def cost_function(x):
        y = np.dot(A, x)
        return np.dot((y - r).T, (y - r))
    result = minimize(cost_function, x_initial_guess, method='BFGS')
    x_optimal = result.x
    J_optimal = result.fun

    return x_optimal, J_optimal
def bruto_force(r,waves):
    x_ls=[]
    J_ls=[]
    d_ls=np.arange(2,15,1)
    for d in d_ls:
        A,d=creat_matrix_A(waves,d,r)
        x, J = optimize_x(A, r)
        x_ls.append(x)
        J_ls.append(J)
    min_idx = np.argmin(J_ls)
    optimal_d= d_ls[min_idx]


    return x_ls,J_ls,optimal_d


if __name__ == '__main__':
    # do the simulation and save in hard disk
    creat_the_datafile(new_data=0 )
    # load simulated signal
    loaded_dict = np.load('data.npy', allow_pickle=True).item()
    print('keys of dict：', loaded_dict.keys())
    waves_dict=seprate_the_waves(loaded_dict)
    print('keys of wave：', waves_dict.keys())
    for i in list(loaded_dict.keys())[0:1]:
        r=loaded_dict[i][1]
        waves=waves_dict[list(loaded_dict.keys())[-1]]
        x_ls, J_ls, optimal_d = bruto_force(r, waves)
        print(i)
        print(optimal_d)





