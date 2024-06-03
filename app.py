"""CustomTkinter based app, for displaying data related to stress by use of matplotlib"""

__version__ = '0.3.2'
__author__ = 'Oliver Bebe Poulin'
__all__ = ['App']

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from tkinter.colorchooser import askcolor

import customtkinter as ctk  # pip install custom tkinter
import matplotlib            # pip install custom matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.axes import Axes

from data_interpretor import Data  # Requires version 0.3 or greater

matplotlib.use('TkAgg')


class App(ctk.CTk):
    stress_data: Data | None = None

    def __init__(self, title: str, size: tuple[int, int]):
        super().__init__()
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(*size)

        self.output = OutputFrame(self)
        self.input = InputFrame(self)

        center = 0.4
        self.input.place(x=5, y=5, relwidth=center-0.005, relheight=0.99)
        self.output.place(relx=center+0.005, y=5, relwidth=0.99-center, relheight=0.99)

        self.output.stress_plot.generate_axes(title='Stress Levels', xlabel='Time [s]', ylabel='Stress Amplitude')

        self.mainloop()

    def import_data(self, filename):
        self.stress_data = Data(filename)
        self.update_stress()
        self.output.stress_plot.plot_axis(0, self.stress_data.stress, color='orange')

        self.input.datapicker.update_graph()

    def clear_data(self):
        self.output.stress_plot.reset()
        self.output.data_plot.reset()
        self.stress_data = None

    def update_stress(self):
        self.output.stress_plot.reset()
        self.output.stress_plot.generate_axes(title='Stress Levels', xlabel='Time [s]', ylabel='Stress Amplitude')
        self.output.stress_plot.plot_axis(0, self.stress_data.stress, color='orange')

    def __temporary_data_plotter(self):
        self.output.data_plot.generate_axes(title='Data Plot', xlabel='Time [s]', ylabel=('Voltage [V]',
                                                                                          'HeartRate [bpm]'))

        rt, ppm = self.stress_data.time, self.stress_data.raw[0]
        self.output.data_plot.plot_axis(0, (rt, ppm), color='blue', linewidth=0.8)

        temperature = self.stress_data.temperature
        self.output.data_plot.plot_axis(0, temperature, color='green', linewidth=0.8)

        time, beats, heart_rate = self.stress_data.heart_rate
        self.output.data_plot.scatter_axis(0, (time, beats), color='red', s=8)
        self.output.data_plot.plot_axis(1, (time, heart_rate), color='red')
        self.output.data_plot.update_canvas()


# ==================================================================================
# Left side of App
class InputFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        FileGetter(self, parent)
        UserData(self, parent)
        self.datapicker = DataPicker(self, parent)


class FileGetter(ctk.CTkFrame):
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root

        self.rowconfigure((0, 1), weight=1, uniform='a')
        self.columnconfigure((0, 1, 2, 3), weight=1, uniform='a')

        # Generate Widgets
        self.entry = ctk.CTkEntry(self, placeholder_text='Data File')

        browse_button = ctk.CTkButton(self, text=' Browse ... ', command=self.browse)
        import_button = ctk.CTkButton(self, text=' Import Data', command=self.import_data)
        clear_button = ctk.CTkButton(self, text='Clear Data', command=self.root.clear_data)

        # Widget Layout
        self.entry.grid(row=0, column=0, columnspan=3, sticky='we', padx=10, pady=4)
        browse_button.grid(row=0, column=3, padx=10, pady=4)
        import_button.grid(row=1, column=2, columnspan=2, padx=10, pady=4)
        clear_button.grid(row=1, column=1, padx=10, pady=4)

        self.pack(side='top', padx=5, pady=10, expand=False, fill='x')

    def browse(self):
        filename = askopenfilename(title='Select data file', filetypes=[('Text File', 'txt'),
                                                                        ('Data File', 'dat')])
        self.entry.delete(0, 'end')
        self.entry.insert(0, filename)

    def import_data(self):
        try:
            self.root.import_data(self.entry.get())
        except FileNotFoundError:
            messagebox.showerror('Missing File', 'File not found!')
        except IndexError:
            messagebox.showerror('Format Error', 'Data format\nnot as expected!')
        except Exception as e:
            messagebox.showerror('Unknown Error', f'Unknown exception of type {type(e).__name__} occurred!\n<{str(e)}>')


class UserData(ctk.CTkFrame):
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root

        title = ctk.CTkFont(family='Default', size=14, weight='bold')
        ctk.CTkLabel(self, text='User Data', justify='left', font=title).pack(expand=False, fill='x')

        self.gender = NamedComboBox(self, text='Gender', default='', values=['Male', 'Female'])
        self.resting_hrt = NamedEntry(self, text='Resting Heartrate', default=85)
        self.voice = NamedComboBox(self, text='Voice Frequency', default='Unknown', values=['Unknown', 'Low',
                                                                                            'Middle', 'High'])

        self.button = ctk.CTkButton(self, text='Update Data', command=self.update_config)
        self.button.pack(side='right')

        self.pack(side='top', padx=5, pady=0, expand=False, fill='both', ipady=5)

    def update_config(self):
        print('updating')
        if self.root.stress_data is None:
            return

        ll, ul = 350, 550
        match self.gender.get() + self.voice.get():
            case 'Unknown':
                pass
            case 'Low':
                ll, ul = 300, 500
            case 'Middle':
                pass
            case 'High':
                ll, ul = 400, 600
            case 'MaleUnknown':
                pass
            case 'MaleLow':
                ll, ul = 300, 450
            case 'MaleMiddle':
                ll, ul = 350, 500
            case 'MaleHigh':
                ll, ul = 400, 550
            case 'FemaleUnknown':
                ll, ul = 400, 550
            case 'FemaleLow':
                ll, ul = 350, 500
            case 'FemaleMiddle':
                ll, ul = 400, 550
            case 'FemaleHigh':
                ll, ul = 450, 600
        self.root.stress_data.configure('frq', lower_limit=ll, upper_limit=ul)

        resting_hrt = self.resting_hrt.get()
        ll = resting_hrt*1.05
        ul = resting_hrt*1.40
        abs_min = resting_hrt/2.0
        abs_max = resting_hrt*2.5
        self.root.stress_data.configure('hrt', lower_limit=ll, upper_limit=ul, hrt_min=abs_min, hrt_max=abs_max)

        self.root.stress_data.recalculate()
        self.root.update_stress()


class NamedEntry(ctk.CTkFrame):
    def __init__(self, parent, text='Entry', default=None):
        super().__init__(parent)

        if default is None:
            return
        elif isinstance(default, str):
            self.var = ctk.StringVar(value=default)
        elif isinstance(default, int):
            self.var = ctk.IntVar(value=default)
        else:
            raise NotImplementedError(f'Variable of type {type(default)} not expected. Try (str, int)')

        self.rowconfigure(0, weight=1, uniform='a')
        self.columnconfigure((0, 1), weight=1, uniform='a')

        ctk.CTkLabel(self, text=text).grid(row=0, column=0, sticky='w')
        ctk.CTkEntry(self, textvariable=self.var).grid(row=0, column=1, sticky='e')

        self.pack(side='top', padx=5, pady=0, expand=False, fill='x')

    def get(self):
        return self.var.get()


class NamedComboBox(ctk.CTkFrame):
    def __init__(self, parent, text='Combo Box', default=None, values: list = None, state='readonly'):
        super().__init__(parent)

        if default is None:
            return
        elif isinstance(default, str):
            self.var = ctk.StringVar(value=default)
        elif isinstance(default, int):
            self.var = ctk.IntVar(value=default)
        else:
            raise NotImplementedError(f'Variable of type {type(default)} not expected. Try (str, int)')

        if values is None:
            return

        self.rowconfigure(0, weight=1, uniform='a')
        self.columnconfigure((0, 1), weight=1, uniform='a')

        ctk.CTkLabel(self, text=text).grid(row=0, column=0, sticky='w')
        ctk.CTkComboBox(self, values=values, variable=self.var, state=state).grid(row=0, column=1, sticky='e')

        self.pack(side='top', padx=5, pady=0, expand=False, fill='x')

    def get(self):
        return self.var.get()


class DataPicker(ctk.CTkFrame):
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self.graph = root.output.data_plot

        self.tabview = ctk.CTkTabview(self, command=self.update_graph)
        self.tabview.pack(padx=5, pady=10, expand=True, fill='both')

        hrt = self.tabview.add('Heart Rate')
        tmp = self.tabview.add('Temperature')
        frq = self.tabview.add('Frequency')
        self.tabview.set('Heart Rate')

        # Heartrate tab
        self.show_raw_hrt = DataToggle(hrt, self, text='Raw Heartrate Data [V]', color='blue', linewidth=0.8)
        self.show_beats = DataToggle(hrt, self, text='Beat Detection [V]', color='red')
        self.show_hrt = DataToggle(hrt, self, text='Heartrate', color='red', linewidth=1.0)

        # Temperature tab
        self.show_tmp = DataToggle(tmp, self, text='Temperature Data', color='green', linewidth=0.8)

        # Frequency tab
        self.show_frq = DataToggle(frq, self, text='Average Frequency', color='purple', linewidth=1.0)

        self.pack(side='bottom', padx=5, pady=10, expand=True, fill='both')

    def update_graph(self):
        self.graph.reset()

        if self.root.stress_data is None:
            return

        match self.tabview.get():
            case 'Heart Rate':
                self.graph.generate_axes(title='Heartrate Data', xlabel='Time [s]', ylabel=('Heartrate [bpm]',
                                                                                            'Raw data [V]'))
                rt, ppm, _, _ = self.root.stress_data.raw
                time, beats, heartrate = self.root.stress_data.heart_rate
                self.plot(self.show_raw_hrt, 1, (rt, ppm))
                self.scatter(self.show_beats, 1, (time, beats))
                self.plot(self.show_hrt, 0, (time, heartrate))

            case 'Temperature':
                self.graph.generate_axes(title='Temperature Data', xlabel='Time [s]', ylabel='Temperature Difference [C]')

                time, tmp = self.root.stress_data.temperature
                self.plot(self.show_tmp, 0, (time, tmp))

            case 'Frequency':
                self.graph.generate_axes(title='Frequency Data', xlabel='Time [s]', ylabel='Average Frequency [Hz]')

                time, frq = self.root.stress_data.frequencies
                self.plot(self.show_frq, 0, (time, frq))
        self.graph.update_canvas()

    def plot(self, toggle, axis, data):
        if not toggle.get():
            return

        self.graph.plot_axis(axis, data, **toggle.line)

    def scatter(self, toggle, axis, data):
        if not toggle.get():
            return

        self.graph.scatter_axis(axis, data, color=toggle.line['color'], s=8)


class DataToggle(ctk.CTkFrame):
    def __init__(self, parent, master, text='', color='black', linewidth=1.0):
        super().__init__(parent)
        self.master = master

        self.line = {'color': color, 'linewidth': linewidth}
        self.rowconfigure(0, weight=1, uniform='a')
        self.columnconfigure((0, 1), weight=1, uniform='a')

        self.c_var = ctk.BooleanVar(value=False)
        self.check_box = ctk.CTkSwitch(self, text=text, variable=self.c_var, onvalue=True, offvalue=False, command=self.event)
        self.check_box.grid(row=0, column=0, sticky='w')

        size = 30
        self.color_picker = ctk.CTkCanvas(master=self, width=size, height=size, bg=color, highlightthickness=0)
        self.color_picker.bind("<Button-1>", self.new_color)

        self.color_picker.create_rectangle((1, 1), (size-1, size-1), width=2, outline='#888')
        self.color_picker.create_polygon((size - 3, size - 3),
                                         (size - 3, size * 0.5),
                                         (size * 0.5, size - 3), fill='#444')
        self.color_picker.create_rectangle((3, 3), (size-3, size-3), width=2, outline='#eee')
        self.color_picker.grid(row=0, column=1, sticky='e')

        self.pack(side='top', expand=False, fill='x')

    def new_color(self, _):
        new_color: str | None = askcolor()[1]

        if new_color is not None:
            self.line['color'] = new_color
            self.color_picker.configure(bg=new_color)
            self.master.update()

    def event(self):
        self.master.update_graph()

    def get(self):
        return self.check_box.get()


# ==================================================================================
# Right side of App
class OutputFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.stress_plot = Plot(self)
        self.data_plot = Plot(self, axes=2)


class Plot(ctk.CTkFrame):
    def __init__(self, parent, axes=1):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100)

        if axes not in [1, 2]:
            raise ValueError('Max two axes')

        self.axes: list[Axes] = []
        self.axes.append(self.figure.add_subplot())
        self.figure.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.07)
        if axes == 2:
            self.axes.append(self.axes[0].twinx())

        figure_canvas = FigureCanvasTkAgg(self.figure, self)
        NavigationToolbar2Tk(figure_canvas, self)

        figure_canvas.get_tk_widget().pack(padx=10, pady=10, expand=True, fill='x')
        self.update_canvas()
        self.pack(side='top', padx=10, pady=10, expand=True, fill='both')

    def generate_axes(self, title: str, xlabel: str, ylabel: str | tuple[str, str]):

        if isinstance(ylabel, tuple):
            self.axes[0].set_ylabel(ylabel[0])
            self.axes[1].set_ylabel(ylabel[1], labelpad=-self.figure.get_figwidth()*67)

        else:
            self.axes[0].set_ylabel(ylabel)

        self.axes[0].set_title(title)
        self.axes[0].set_xlabel(xlabel)

    def scatter_axis(self, axis: int, data: tuple[np.ndarray, np.ndarray], *args, **kwargs):
        self.axes[axis].scatter(*data, *args, **kwargs)
        self.update_canvas()

    def plot_axis(self, axis: int, data: tuple[np.ndarray, np.ndarray], *args, **kwargs):
        self.axes[axis].plot(*data, *args, **kwargs)
        self.update_canvas()

    def reset(self):
        print(f'Clearing Plot: {self}')
        for axis in self.axes:
            axis.cla()

        self.update_canvas()

    def update_canvas(self):
        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()

        if ctk.get_appearance_mode() == 'Dark':
            self.set_darkmode()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def set_darkmode(self):
        self.figure.patch.set_facecolor('#303030')
        self.axes[0].patch.set_facecolor('#212121')


def main():
    ctk.set_appearance_mode('Light')
    App('Stress Calculator', (1200, 900))


if __name__ == '__main__':
    main()
