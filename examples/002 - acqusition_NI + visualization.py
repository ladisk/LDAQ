"""
NI Acquisition with live visualization (manual start).

This example shows how to:
1. Create an AITask with channels
2. Wrap it in NIAcquisition
3. Add live visualization (time domain + FFT)
4. Start measurement manually using the Start button in the GUI (no trigger)
"""
import LDAQ

# Create a NI AITask:
task_in = LDAQ.national_instruments.AITask("input_task_vis", sample_rate=25000)
task_in.add_channel(channel_name='vol1', device_ind=2, channel_ind=0, units='V')
task_in.add_channel(channel_name='vol2', device_ind=2, channel_ind=1, units='V')
task_in.add_channel(channel_name='vol3', device_ind=2, channel_ind=2, units='V')

# Create acquisition source:
acq = LDAQ.national_instruments.NIAcquisition(task_in, acquisition_name="NI")

# Configure live visualization:
vis = LDAQ.Visualization(refresh_rate=100)
vis.add_lines((0, 0), "NI", [0, 1, 2])  # Time domain: all 3 channels
vis.add_lines((1, 0), "NI", [0], function="fft", refresh_rate=3000)  # FFT of vol1

vis.config_subplot((0, 0), t_span=0.1, ylim=(-5, 5))
vis.config_subplot((1, 0), t_span=5.0, ylim=(0, 1.2), xlim=(0, 5000))

# Create core and run (no trigger, press Start button in GUI to begin recording):
ldaq = LDAQ.Core(acquisitions=[acq], visualization=vis)
ldaq.run(measurement_duration=5.0)

# Retrieve measurement data:
measurement = ldaq.get_measurement_dict()
print(measurement)
