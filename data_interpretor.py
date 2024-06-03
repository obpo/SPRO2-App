"""Data interpreter for stress related data"""

__version__ = '0.3.3'
__author__ = 'Oliver Bebe Poulin'
__all__ = ['Data', 'shift_range', 'interpolate', 'ema']

import numpy as np

from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline


def shift_range(val, in_min: float = 0, in_max: float = None,
                out_min: float = 0, out_max: float = None) -> np.ndarray:
    """Shifts the range of values: [in_min:in_max] -> [out_min:out_max]"""
    return (val - in_min) * out_max/in_max + out_min


def interpolate(x, y, num: int = None, k: int = 2, return_x: bool = False
                , top: int = None, bottom: int = 0) -> (np.ndarray, np.ndarray):
    """Fills in the gaps of a dataset, so all sets can be the same length

    :arg x -- array_like, shape(n)
    :arg y -- array_like, shape(n)
    :arg num -- Length of new dataset
    :arg k -- B-spline degree. Default is linear ''2''
    :arg return_x -- if the function should return the new x-axis along with the y-axis
    :arg top -- Upper limit of interp
    :arg bottom -- Lower limit of interp
    """
    if num is None:
        num = len(x)

    if top is None:
        top = max(x)

    x_new = np.linspace(bottom, top, num=num, endpoint=True)

    if k >= len(x):
        if return_x:
            return x_new, np.zeros(num)
        return np.zeros(num)

    interp = make_interp_spline(x, y, k=k)

    if return_x:
        return x_new, interp(x_new)
    return interp(x_new)


def ema(data, weight: float = 0) -> np.ndarray:
    """Exponential Moving Average: Smooths a dataset, so spikes are less noticeable

    :arg data -- array_like, shape(n)
    :arg weight -- smoothing factor in range [0, 1] 0 is minimum smoothing and 1 is maximum smoothing
    :source -- https://stackoverflow.com/a/488700
    """
    if 0 >= weight >= 1:
        raise ValueError('Weight must be in range [0, 1]')

    out = np.empty(len(data))
    out[0] = data[0]

    for i in range(1, len(data)):
        out[i] = out[i-1]*weight + data[i]*(1-weight)

    return out


# Custom Exception for if datafile is of the wrong format
class FormatError(Exception):
    pass


class Data:
    """Stress interpreter

    :parameter filepath: location of datafile in storage.

    :var self.raw_data: Raw input data of format - time[s], heart_rate[V], temperature_delta[V], list(FFT)
    :var self.heart_rate: Heartrate data of format - time[s], beats[V], heart rate [bpm]
    :var self.temperature: Temperature data of format - time[s], temperature_delta[C~]
    :var self.frequencies: Frequency data of format - time[s], Average Frequency[Hz]
    :var self.stress: Calculated stress of format - time[s], stress[unknown]

    :exception FormatError: If datafile is of the wrong format
    """
    def __init__(self, filepath: str):

        self.raw: tuple[np.ndarray, np.ndarray, np.ndarray]
        self.heart_rate: tuple[np.ndarray, np.ndarray, np.ndarray] = ...
        self.temperature: tuple[np.ndarray, np.ndarray] = ...
        self.frequencies: tuple[np.ndarray, np.ndarray] = ...
        self.stress: tuple[np.ndarray, np.ndarray] = ...

        # scaling * e^(factor * power)
        self.param: dict[str: dict[str:int, str:int, str:int, str:int, str:bool]] = {
            'hrt': {'power': 2.0, 'scaling': 4.4, 'lower_limit': 85, 'upper_limit': 120, 'auto_scale': True},
            'tmp': {'power': 2.0, 'scaling': 4.4, 'lower_limit': 2.5, 'upper_limit': 5.0, 'auto_scale': True},
            'frq': {'power': 2.0, 'scaling': 4.4, 'lower_limit': 350, 'upper_limit': 500, 'auto_scale': True}}

        self.settings: dict[str: dict[str: int]] = {
            'hrt': {'smoothing': 0.05, 'hrt_min': 20, 'hrt_max': 200},
            'tmp': {'smoothing': 0.25},
            'frq': {'smoothing': 0.95, 'sampling_rate': 1000}}

        self.raw = self.__read_file(filepath)

        self.recalculate()

    def __read_file(self, filepath: str):

        raw_time, raw_ppm, raw_ntc, frq_bins = [], [], [], []

        with open(filepath, 'r') as file:
            for line in file:
                line = line.rstrip("\n")
                if not line:
                    continue

                if line[-1] == ',':
                    line = line[:-1]

                data = [x for x in line.split(';')]
                if len(data) != 4:
                    raise FormatError('Datafile not of expected format[time; hrt_val; tmp_val; frq_bin0, ... frq_bin8]')

                freq_bin = [float(x) for x in data[3].split(',')]
                if len(freq_bin) != 9:
                    raise ResourceWarning('There may not be enough frequency bins')

                raw_time.append(int(data[0]))
                raw_ppm.append(int(data[1]))
                raw_ntc.append(int(data[2]))
                frq_bins.append(np.asarray(freq_bin[1:]))

        raw_time = np.asarray(raw_time)/1000
        raw_time, index = np.unique(raw_time, return_index=True)

        raw_ppm = shift_range(np.asarray(raw_ppm), 0, 1023, 0, 5)[index]
        raw_ntc = shift_range(np.asarray(raw_ntc), 0, 1023, 0, 5)[index]
        frq_bins = np.asarray(frq_bins)

        self.raw = raw_ppm, raw_ntc
        self.time = raw_time

        return raw_time, raw_ppm, raw_ntc, frq_bins

    @staticmethod
    def __ppm(time, amplitude, amp_min=1.0, hrt_min=20, hrt_max=200, smoothing=0):

        # Remove Bad data
        amplitude_mask = amplitude > amp_min
        time = time[amplitude_mask]
        amplitude = amplitude[amplitude_mask]

        smoothed = ema(amplitude, smoothing)

        # Calculate Heart Rate
        center = (max(amplitude) + min(amplitude)) / 2.1
        center_mask = smoothed > center
        time = time[center_mask]
        amplitude = smoothed[center_mask]

        maxima = time[np.r_[True, amplitude[1:] > amplitude[:-1]] & np.r_[amplitude[:-1] >= amplitude[1:], True]]
        beats = amplitude[np.r_[True, amplitude[1:] > amplitude[:-1]] & np.r_[amplitude[:-1] >= amplitude[1:], True]]
        heart_rate = 60 / (maxima[1:] - maxima[:-1])
        heart_rate = np.insert(heart_rate, 0, np.nan)

        # Remove erroneous data caused by earlier removal of bad data
        heart_rate_mask = (hrt_min < heart_rate) * (heart_rate < hrt_max)
        maxima = maxima[heart_rate_mask]
        beats = beats[heart_rate_mask]
        heart_rate = heart_rate[heart_rate_mask]

        return maxima, beats, heart_rate

    @staticmethod
    def __ntc(time, amplitude, smoothing):
        amplitude = ema(amplitude, smoothing)
        return interpolate(time, amplitude, return_x=True)

    @staticmethod
    def __mic(time, frq_bins, smoothing, sampling_rate=1000):
        bin_width = len(frq_bins[0])
        bin_size = sampling_rate / bin_width

        bin_sums = np.asarray([sum(b) for b in frq_bins])
        bin_mask = bin_sums != 0
        time, frq_bins, bin_sums = time[bin_mask], frq_bins[bin_mask], bin_sums[bin_mask]

        bins = frq_bins * np.linspace(1, bin_width, bin_width)

        frequencies = np.asarray([sum(b) for b in bins])
        frequencies = (frequencies * bin_size) / bin_sums
        frequencies = ema(frequencies, smoothing)

        return time, frequencies

    @staticmethod
    def __stress_factor(factor, power, scaling, *, lower_limit=0.0, upper_limit=None, auto_scale=False):

        if auto_scale:
            upper_limit = max(upper_limit, max(factor))

        factor = np.where(factor > lower_limit, factor-lower_limit, 0)
        factor = factor / (upper_limit - lower_limit)

        return scaling * np.exp(factor*power) - scaling

    def recalculate(self):
        raw_time, raw_ppm, raw_ntc, frq_bins = self.raw

        self.heart_rate = self.__ppm(raw_time, raw_ppm, **self.settings['hrt'])
        self.temperature = self.__ntc(raw_time, raw_ntc, **self.settings['tmp'])
        self.frequencies = self.__mic(raw_time, frq_bins, **self.settings['frq'])
        self.calc_stress()

    def calc_stress(self):
        num = len(self.time)
        span = max(self.time)

        time, _, bpm = self.heart_rate
        time, bpm = time[1:], bpm[1:]
        bpm_factor = interpolate(time, bpm, num=num)
        temp_factor = interpolate(*self.temperature, num=num)
        freq_factor = interpolate(*self.frequencies, num=num)

        bpm_factor = self.__stress_factor(bpm_factor, **self.param['hrt'])
        temp_factor = self.__stress_factor(temp_factor, **self.param['tmp'])
        freq_factor = self.__stress_factor(freq_factor, **self.param['frq'])

        self.stress = np.linspace(0, span, num=num, endpoint=True), bpm_factor + temp_factor + freq_factor

    def configure(self, dataset: str, **kwargs):
        if dataset not in self.settings.keys():
            raise ValueError(f"Dataset: ({dataset}) doesn't exist. Expected values are: {self.settings.keys()}")

        for arg, item in kwargs.items():
            if arg in self.param[dataset].keys():
                self.param[dataset][arg] = int(item)
            elif arg in self.settings[dataset].keys():
                self.settings[dataset][arg] = int(item)
            else:
                raise TypeError(f"Keyword argument ({arg}) not expected. Expected keyword arguments are {list(self.settings[dataset].keys()) + list(self.param[dataset].keys())}")


def main():
    data = Data('f_data0.txt')

    rt, hrt, ntc, _ = data.raw
    time, beats, heart_rate = data.heart_rate

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
    ax2 = ax1.twinx()
    plt.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.07)

    plt.title(f"Average BPM = {heart_rate[1:].mean()}")
    ax1.set_ylabel("Stress [%]")
    ax1.set_xlabel("Times [s]")
    ax2.set_ylabel("BPM")

    plt.title("Stress")

    # Temperature
    ax1.plot(*data.temperature, color="g", linewidth=0.8)

    # Heartrate
    ax1.plot(rt, hrt, linewidth=0.8)
    ax1.scatter(time, beats, color="r", )
    ax2.plot(time, heart_rate, color="r")

    # Frequencies
    ax2.plot(*data.frequencies, color='purple')

    # Stress
    ax2.plot(*data.stress, color='orange')

    ax2.grid()
    plt.show()


if __name__ == '__main__':
    main()
