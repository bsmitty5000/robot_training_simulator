# -- Constants --
PIXELS_PER_METER = 500  # Scale factor for converting meters to pixels
WIDTH = 1280
HEIGHT = 720
FRAME_RATE = 20
COVERAGE_ABORT_S = 5
SIM_DT = 1.0 / FRAME_RATE  # Simulation time step in seconds
HEADLESS_MODE = True  # If True, run without a GUI for faster simulations
LOG_FILE_TO_SEED = r"C:\Users\Batman\projects\remote_robot\simulation\ga_20250617_081545.log"
DEMO_RUN = True  # If True, run a demo with a fixed genotype