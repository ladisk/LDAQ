import nidaqmx
from nidaqmx import constants
import numpy as np
import pandas as pd


class NITask:
    def __init__(self, task_name, sample_rate, settings_file=None):
        self.task_name = task_name

        self.system = nidaqmx.system.System.local()
        self.device_list = [_.name for _ in list(self.system.devices)]

        self.unit_conv = {
            'mV/g': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
            'mV/m/s**2': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G, # TODO: check this
            'g': constants.AccelUnits.G,
            'm/s**2': constants.AccelUnits.METERS_PER_SECOND_SQUARED,
            'mV/N': constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
            'N': constants.ForceUnits.NEWTONS
        }

        self.sample_rate = sample_rate
        self.samples_per_channel = sample_rate # doesn't matter for LDAQ
        self.sample_mode = constants.AcquisitionType.CONTINUOUS
        self.channels = {}
        
        if settings_file is not None:
            self._read_settings_file(settings_file)

    def _create_task(self):
        try:
            self.task = nidaqmx.task.Task(new_task_name=self.task_name)
        except nidaqmx.DaqError:
            raise Exception(f"Task name {self.task_name} already exists.")

    def _read_settings_file(self, file_name):
        if isinstance(file_name, str):
            if file_name.endswith('.xlsx'):
                self.settings = pd.read_excel(file_name)
            elif file_name.endswith('.csv'):
                self.settings = pd.read_csv(file_name)
            else:
                raise Exception('Settings filename must be a .xlsx or .csv file.')
        else:
            raise Exception('Settings filename must be a string.')

    def initiate(self, start_task=True):
        if self.task_name in self.system.tasks.task_names:
            self._delete_task()

        self._create_task()

        self._add_channels()
        
        self._setup_task()

        if start_task:
            self.task.start()

    def add_channel(self, channel_name, device_ind, channel_ind, sensitivity=None, sensitivity_units=None, units=None, serial_nr=None):
        if channel_name in self.channels:
            raise Exception(f"Channel name {channel_name} already exists.")

        if device_ind not in range(len(self.device_list)):
            raise Exception(f"Device index {device_ind} not in range. Available devices: {self.device_list}")
        
        if (device_ind, channel_ind) in [(self.channels[_]['device_ind'], self.channels[_]['channel_ind']) for _ in self.channels]:
            raise Exception(f"Channel {channel_ind} already in use on device {device_ind}.")
        
        if serial_nr is not None:
            # Read data from excel file
            if isinstance(serial_nr, str):
                row = self.settings[self.settings['serial_nr'] == serial_nr]
                if 'sensitivity' not in row.columns:
                    raise Exception('No column "sensitivity" in settings file.')
                if 'sensitivity_units' not in row.columns:
                    raise Exception('No column "sensitivity_units" in settings file.')
                if 'units' not in row.columns:
                    raise Exception('No column "units" in settings file.')
                
                if len(row):
                    sensitivity = row['sensitivity'].iloc[0]
                    sensitivity_units = row['sensitivity_units'].iloc[0]
                    units = row['units'].iloc[0]
                else:
                    raise Exception(f"Serial number {serial_nr} not found in settings file.")
            else:
                raise Exception('Serial number must be a string.')

        if sensitivity is None:
            raise Exception('Sensitivity must be specified.')
        if sensitivity_units is None:
            raise Exception('Sensitivity units must be specified.')
        if units is None:
            raise Exception('Units must be specified.')
        
        self.channels[channel_name] = {
            'device_ind': device_ind,
            'channel_ind': channel_ind,
            'sensitivity': sensitivity,
            'sensitivity_units': sensitivity_units,
            'units': units,
            'serial_nr': serial_nr,
        }

        # list of channel names
        self.channel_list = [self.channels[_]['channel_ind'] for _ in self.channels]
        # number of channels
        self.number_of_ch = len(self.channel_list)

    def _add_channels(self):
        self.channel_objects = []

        for channel_name in self.channels:
            self._add_channel(channel_name)
        
    def _add_channel(self, channel_name):
        mode = self.unit_conv[self.channels[channel_name]['units']].__objclass__.__name__
        channel_ind = self.channels[channel_name]['channel_ind']
        device_ind = self.channels[channel_name]['device_ind']
        physical_channel = f"{self.device_list[device_ind]}/ai{channel_ind}"

        options = {
            'physical_channel': physical_channel,
            'name_to_assign_to_channel': channel_name,
            'terminal_config': constants.TerminalConfiguration.PSEUDO_DIFF,
            'sensitivity': self.channels[channel_name]['sensitivity'],
            'sensitivity_units': self.unit_conv[self.channels[channel_name]['sensitivity_units']],
            'units': self.unit_conv[self.channels[channel_name]['units']],
            'current_excit_val': 0.004,
            'current_excit_source': constants.ExcitationSource.INTERNAL,
        }

        if mode == 'ForceUnits':
            self.channel_objects.append(self.task.ai_channels.add_ai_force_iepe_chan(**options))
        elif mode == 'AccelUnits':
            self.channel_objects.append(self.task.ai_channels.add_ai_accel_chan(**options))

    def _setup_task(self):
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate, 
            sample_mode=self.sample_mode,
            samps_per_chan=self.samples_per_channel)  # set sampling for the task
        
        # set task handle
        self.taskHandle = self.task._handle

    def clear_task(self, wait_until_done=False):
        if hasattr(self, 'task'):
            self.task.close()
        else:
            print('No task to clear.')

    def acquire_base(self):
        return np.array(self.task.read(number_of_samples_per_channel=constants.READ_ALL_AVAILABLE))

    def acquire(self, wait_4_all_samples=False):
        """Acquires the data from the task.

        Attributes:
        time_out: maximal waiting time for the measured data to be available
        wait_4_all_samples: return when all samples are acquired
        acquire_sleep: sleep time in seconds between acquisitions in continuous mode

        Returns:
            Nothing.

        Raises:
            Nothing.
        """
        self.data = None

        data = self.acquire_base()
        
        if self.data is None:
            self.data = data
        else:
            self.data = np.concatenate((self.data, data), axis=1)

    def _delete_task(self):
        tasks = [_ for _ in self.system.tasks]
        task_ind = self.system.tasks.task_names.index(self.task_name)
        tasks[task_ind].delete()

    def save(self, clear_task=True):
        if not hasattr(self, 'Task'):
            self.initiate(start_task=False)

        self.task.save(self.task_name, overwrite_existing_task=True)
        
        if clear_task:
            self.clear_task()