% by CK, based on kwave example example_at_circular_piston_3D.m
% 
% Simulation of a plane piston transducer in water with a pmma plate
%

clearvars;

% =========================================================================
% DEFINE LITERALS
% =========================================================================
    
% select which k-Wave code to run
%   1: MATLAB CPU code
%   2: MATLAB GPU code
%   3: C++ code
%   4: CUDA code
model           = 1;
simulation_data_path = "C:\Users\chris\Desktop\Simulations_daten";
info = "test_sim_mat";

% medium parameters
c0              = 1500;     % sound speed [m/s]
rho0            = 1000;     % density [kg/m^3]
c_pmma          = 2750;
rho_pmma        = 1180;

% source parameters
source_f0       = 1e6;      % source frequency [Hz]
source_diam     = 15e-3;    % piston diameter [m]
source_amp      = 1e6;      % source pressure [Pa]

% experimental setup
distance_transducer_plate_surface = 37.5e-3;    % This is the focal length of a 15 mm diameter transducer
plate_thickness_in_lambda = 2;
plate_thickness_in_meter = plate_thickness_in_lambda/source_f0*c_pmma;

% grid parameters
axial_size      = 50e-3;    % total grid size in the axial dimension [m]
lateral_size    = 20e-3;    % total grid size in the lateral dimension [m]

% computational parameters
ppw             = 5;        % number of points per wavelength
t_end           = 40e-6;    % total compute time [s] (this must be long enough to reach steady state)
record_periods  = 1;        % number of periods to record
cfl             = 0.5;      % CFL number
bli_tolerance   = 0.03;     % tolerance for truncation of the off-grid source points
upsampling_rate = 10;       % density of integration points relative to grid

% =========================================================================
% RUN SIMULATION
% =========================================================================

% --------------------
% GRID
% --------------------

% calculate the grid spacing based on the PPW and F0
dx = c0 / (ppw * source_f0);   % [m]

% compute the size of the grid
Nx = roundEven(axial_size / dx);
Ny = roundEven(lateral_size / dx);
Nz = Ny;

% create the computational grid
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

% compute points per temporal period
PPP = round(ppw / cfl);

% compute corresponding time spacing
dt = 1 / (PPP * source_f0);

% create the time array using an integer number of points per period
Nt = round(t_end / dt);
kgrid.setTime(Nt, dt);

% calculate the actual CFL and PPW
disp(['PPW = ' num2str(c0 / (dx * source_f0))]);
disp(['CFL = ' num2str(c0 * dt / dx)]);

% --------------------
% SOURCE
% --------------------

% create time varying source
source_sig = source_amp * toneBurst(1/kgrid.dt, source_f0, 3);

% create empty kWaveArray
karray = kWaveArray('BLITolerance', bli_tolerance, 'UpsamplingRate', upsampling_rate);

% add disc shaped element at one end of the grid --> @ Ziwen: we use this
% as the reference point
karray.addDiscElement([kgrid.x_vec(1), 0, 0], source_diam, [0, 0, 0]);
    
% assign binary mask
source.p_mask = karray.getArrayBinaryMask(kgrid);

% assign source signals
source.p = karray.getDistributedSourceSignal(kgrid, source_sig);
    
% --------------------
% MEDIUM
% --------------------

% assign medium properties
[diff, top_surface_plate_idx_x_direction] = min(abs(kgrid.x_vec - (kgrid.x_vec(1) + distance_transducer_plate_surface)));
[diff, bottom_surface_plate_idx_x_direction] = min(abs(kgrid.x_vec - (kgrid.x_vec(1) + (distance_transducer_plate_surface + plate_thickness_in_meter))));

medium.sound_speed = c0 * ones(Nx, Ny, Nz);         % [m/s]
medium.sound_speed(top_surface_plate_idx_x_direction:bottom_surface_plate_idx_x_direction, :, :) = c_pmma;       % speed of sound of PMMA [m/s]
medium.density = rho0 * ones(Nx, Ny, Nz);           % [kg/m^3]
medium.density(top_surface_plate_idx_x_direction:bottom_surface_plate_idx_x_direction, :, :) = rho_pmma;         % density of PMMA

% --------------------
% SENSOR
% --------------------

% set sensor mask to record central plane, not including the source point
sensor.mask = karray.getArrayBinaryMask(kgrid);

% record the pressure
sensor.record = {'p'};

% --------------------
% SIMULATION
% --------------------

% set input options
input_args = {...
    'PMLSize', 'auto', ...
    'PMLInside', false, ...
    'PlotPML', false, ...
    'DisplayMask', 'off'};

% run code
switch model
    case 1
        
        % MATLAB CPU
        sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, ...
            input_args{:}, ...
            'DataCast', 'single', ...
            'PlotScale', [-1, 1] * source_amp);
        
    case 2
        
        % MATLAB GPU
        sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, ...
            input_args{:}, ...
            'DataCast', 'gpuArray-single', ...
            'PlotScale', [-1, 1] * source_amp);
        
    case 3
        
        % C++
        sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, input_args{:});
        
    case 4
        
        % C++/CUDA GPU
        sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
        
end

file = append(simulation_data_path,"\", datestr(datetime("now"), 'yyyymmddTHHMMSS'), "_", info);
save(file)


