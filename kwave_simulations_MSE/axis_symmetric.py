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
    Nx = 128;
    Ny = 64;
    dx = 0.1e-3;
    dy = 0.1e-3;
    kgrid = kWaveGrid(Nx, dx, Ny, dy);

    medium.sound_speed = 1500 * ones(Nx, Ny);
    medium.sound_speed(Nx / 2: end,:) = 1800;
    medium.density = 1000 * ones(Nx, Ny);
    medium.density(Nx / 2: end,:) = 1200;

    source.p0 = 10 * makeDisc(Nx, 2 * Ny, Nx / 4 + 8, Ny + 1, 5);
    source.p0 = source.p0(:, Ny + 1: end);