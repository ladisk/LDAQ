import nidaqmx
from nidaqmx import constants
from nidaqmx import Scale
import numpy as np
import pandas as pd

UNITS = {
    'mV/g': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
    'mV/m/s**2': constants.AccelSensitivityUnits.MILLIVOLTS_PER_G, # TODO: check this
    'g': constants.AccelUnits.G,
    'm/s**2': constants.AccelUnits.METERS_PER_SECOND_SQUARED,
    'mV/N': constants.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
    'N': constants.ForceUnits.NEWTONS,
    'V': constants.VoltageUnits.VOLTS,
}

class NITaskOutput:
    def __init__(self, task_name, sample_rate):
        """Create a new NI task for analog output.
        
        :param task_name: name of the task
        :param sample_rate: sample rate in Hz
        """
        self.task_name = task_name
        self.sample_rate = sample_rate
        self.channels = {}
        
    def add_channel(self, channel_name, device_ind, channel_ind, min_val=None, max_val=None):
        """Add a channel to the task.
        
        :param channel_name: name of the channel.
        :param device_ind: index of the device. To see all devices, see ``self.device_list`` attribute.
        :param channel_ind: index of the channel on the device.
        :param min_val: minimum value of the channel.
        :param max_val: maximum value of the channel.
        """
        self.channels[channel_name] = {
            'device_ind': device_ind,
            'channel_ind': channel_ind,
            'min_val': min_val,
            'max_val': max_val,
        }

    def _create_task(self):
        try:
            self.task = nidaqmx.task.Task(new_task_name=self.task_name)
        except nidaqmx.DaqError:
            raise Exception(f"Task name {self.task_name} already exists.")
        
    def _add_channels(self):
        self.channel_objects = []

        for channel_name in self.channels:
            self.channel_objects.append(self._add_channel(channel_name))
            
    def _add_channel(self, channel_name):
        channel_ind = self.channels[channel_name]['channel_ind']
        device_ind = self.channels[channel_name]['device_ind']
        physical_channel = f"{self.device_list[device_ind]}/ai{channel_ind}"

        self.task.ao_channels.add_ao_voltage_chan(
            physical_channel=physical_channel,
            name_to_assign_to_channel=channel_name,
            min_val=self.channels[channel_name]['min_val'],
            max_val=self.channels[channel_name]['max_val'],
        )

    def _setup_task(self):
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate, 
            sample_mode=self.sample_mode,
            samps_per_chan=self.samples_per_channel)  # set sampling for the task
        
        # set task handle
        self.taskHandle = self.task._handle

    def initiate(self, start_task=True):
        """Initiate the task.

        :param start_task: start the task after initiating it.
        """
        if self.task_name in self.system.tasks.task_names:
            self._delete_task()

        self._create_task()

        self._add_channels()
        
        self._setup_task()

        # if start_task:
        #     self.task.start()

    def generate(self, signal, clear_task=False):
        self.task.write(signal, auto_start=True)

    def clear_task(self, wait_until_done=False):
        if hasattr(self, 'task'):
            self.task.close()
        else:
            print('No task to clear.')



class NITask:
    def __init__(self, task_name, sample_rate, settings_file=None):
        """Create a new NI task.
        
        :param task_name: name of the task
        :param sample_rate: sample rate in Hz
        :param settings_file: path to xlsx settings file.

        The settings file must contain the following columns:
        - serial_nr: serial number of the sensor
        - sensitivity: sensitivity of the sensor
        - sensitivity_units: units of the sensitivity
        - units: units of the sensor
        Other columns are ignored.
        """
        self.task_name = task_name

        self.system = nidaqmx.system.System.local()
        self.device_list = [_.name for _ in list(self.system.devices)]

        self.sample_rate = sample_rate
        self.samples_per_channel = sample_rate # doesn't matter for LDAQ
        self.sample_mode = constants.AcquisitionType.CONTINUOUS
        self.channels = {}
        
        self.settings = None
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
        """Initiate the task.

        :param start_task: start the task after initiating it.
        """
        if self.task_name in self.system.tasks.task_names:
            self._delete_task()

        self._create_task()

        self._add_channels()
        
        self._setup_task()

        if start_task:
            self.task.start()

    def add_channel(self, channel_name, device_ind, channel_ind, sensitivity=None, sensitivity_units=None, units=None, serial_nr=None, scale=None):
        """Add a channel to the task. The channel is not actually added to the task until the task is initiated.

        :param channel_name: name of the channel.
        :param device_ind: index of the device. To see all devices, see ``self.device_list`` attribute.
        :param channel_ind: index of the channel on the device.
        :param sensitivity: sensitivity of the sensor.
        :param sensitivity_units: units of the sensitivity.
        :param units: output units.
        :param serial_nr: serial number of the sensor. If specified, the sensitivity, sensitivity_units and units are read from the settings file.
        :param scale: scale the signal. If specified, the sensitivity, sensitivity_units are ignored. The prescaled units are assumed to be Volts, the
            scaled units are assumed to be ``units``. The scale can be float or a tuple. If float, this is the slope of the linear scale and y-interception is
            at 0. If tuple, the first element is the slope and the second element is the y-interception.
        """
        if scale is None and sensitivity_units not in UNITS:
            raise Exception(f"Sensitivity units {sensitivity_units} not in {UNITS.keys()}.")
        
        if scale is None and units not in UNITS:
            raise Exception(f"Units {units} not in {UNITS.keys()}.")

        if channel_name in self.channels:
            raise Exception(f"Channel name {channel_name} already exists.")

        if device_ind not in range(len(self.device_list)):
            raise Exception(f"Device index {device_ind} not in range. Available devices: {self.device_list}")
        
        if (device_ind, channel_ind) in [(self.channels[_]['device_ind'], self.channels[_]['channel_ind']) for _ in self.channels]:
            raise Exception(f"Channel {channel_ind} already in use on device {device_ind}.")
        
        if serial_nr is not None and self.settings is not None:
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
            
        if units is None:
            raise Exception('Units must be specified.')
        
        self.channels[channel_name] = {
            'device_ind': device_ind,
            'channel_ind': channel_ind,
            'sensitivity': sensitivity,
            'sensitivity_units': sensitivity_units,
            'units': units,
            'custom_scale_name': "",
            'serial_nr': serial_nr,
        }

        if scale is not None:
            if isinstance(scale, float):
                scale_channel = Scale.create_lin_scale(f'{channel_name}_scale', slope=scale, y_intercept=0, pre_scaled_units=constants.VoltageUnits.VOLTS, scaled_units=self.channels[channel_name]['units'])
            elif isinstance(scale, tuple):
                scale_channel = Scale.create_lin_scale(f'{channel_name}_scale', slope=scale[0], y_intercept=scale[1], pre_scaled_units=constants.VoltageUnits.VOLTS, scaled_units=self.channels[channel_name]['units'])
            else:
                raise Exception('Scale must be a float or a tuple.')

            self.channels[channel_name]['custom_scale_name'] = scale_channel.name

        else:
            if sensitivity is None:
                raise Exception('Sensitivity must be specified.')
            if sensitivity_units is None:
                raise Exception('Sensitivity units must be specified.')
            
        # list of channel names
        self.channel_list = list(self.channels.keys())

        # number of channels
        self.number_of_ch = len(self.channel_list)

    def _add_channels(self):
        self.channel_objects = []

        for channel_name in self.channels:
            self.channel_objects.append(self._add_channel(channel_name))
        
    def _add_channel(self, channel_name):
        if self.channels[channel_name]['units'] in UNITS:
            mode = UNITS[self.channels[channel_name]['units']].__objclass__.__name__
        else:
            mode = 'VoltageUnits'

        channel_ind = self.channels[channel_name]['channel_ind']
        device_ind = self.channels[channel_name]['device_ind']
        physical_channel = f"{self.device_list[device_ind]}/ai{channel_ind}"

        options = {
            'physical_channel': physical_channel,
            'name_to_assign_to_channel': channel_name,
            'terminal_config': constants.TerminalConfiguration.DEFAULT,
            'sensitivity': self.channels[channel_name]['sensitivity'],
            'custom_scale_name': self.channels[channel_name]['custom_scale_name'],
        }

        if mode == 'ForceUnits':
            options['sensitivity_units'] = UNITS[self.channels[channel_name]['sensitivity_units']]
            options['units'] = UNITS[self.channels[channel_name]['units']]
            options = dict([(k, v) for k, v in options.items() if k in ['physical_channel', 'name_to_assign_to_channel', 'terminal_config', 'sensitivity', 'sensitivity_units', 'units']])
            self.channel_objects.append(self.task.ai_channels.add_ai_force_iepe_chan(**options))

        elif mode == 'AccelUnits':
            options['sensitivity_units'] = UNITS[self.channels[channel_name]['sensitivity_units']]
            options['units'] = UNITS[self.channels[channel_name]['units']]
            options = dict([(k, v) for k, v in options.items() if k in ['physical_channel', 'name_to_assign_to_channel', 'terminal_config', 'sensitivity', 'sensitivity_units', 'units']])
            self.channel_objects.append(self.task.ai_channels.add_ai_accel_chan(**options))

        elif mode == 'VoltageUnits':
            options = dict([(k, v) for k, v in options.items() if k in ['physical_channel', 'name_to_assign_to_channel', 'terminal_config', 'custom_scale_name']])
            if options['custom_scale_name'] != "":
                options['units'] = constants.VoltageUnits.FROM_CUSTOM_SCALE
            self.channel_objects.append(self.task.ai_channels.add_ai_voltage_chan(**options))

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
        """Save the task to the system (NI MAX).

        If the task is not initiated yet, it will be initiated.
        
        :param clear_task: clear the task after saving.
        """
        if not hasattr(self, 'Task'):
            self.initiate(start_task=False)

        self.task.save(self.task_name, overwrite_existing_task=True)
        
        if clear_task:
            self.clear_task()