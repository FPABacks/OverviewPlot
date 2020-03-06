import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import LogNorm, Normalize

from scipy.interpolate import interp2d

print "Yay! I am being imported!"

class OverviewPlot:

    def set_unique_parameters(self):
        self.uni1 = np.unique(self.parameters[:,0])
        self.uni2 = np.unique(self.parameters[:,1])
        self.uni3 = np.unique(self.parameters[:,2])
        self.uni4 = np.unique(self.parameters[:,3])

    def __init__(self, parameters=None, data=None, plot_function=None, plot_args=None, plot_kwargs={},
                 x1_ax = 3, y1_ax = 0, x2_ax = 2, y2_ax = 1, y1_label = r"$R_{\rm in} = $", x1_label = r"$i= $",
                 y1_unit = r"AU", x1_unit = r"$^\circ$", y2_label = r"Reference Scaleheight [AU]",
                 x2_label = r"Inner Disk Mass [$M_\odot$]", color_label = r"$\chi^2$", savename = "grid_overview_chi",
                 color_scale = "log", save = False, vmin = None, vmax = None, contour = True, levels = []):

        self.parameters = np.array(parameters)
        self.data = data
        self.plot_function = plot_function
        self.plot_args = plot_args
        self.plot_kwargs = plot_kwargs
        self.x1_ax = x1_ax
        self.y1_ax = y1_ax
        self.x2_ax = x2_ax
        self.y2_ax = y2_ax
        self.y1_label = y1_label
        self.x1_label = x1_label
        self.y1_unit = y1_unit
        self.x1_unit = x1_unit
        self.y2_label = y2_label
        self.x2_label = x2_label
        self.color_label = color_label
        self.color_scale = color_scale
        self.vmin = vmin
        self.vmax = vmax
        self.contour = contour
        self.levels = levels
        self.cmap = "viridis"
        self.norm = Normalize()
        self.fontsize = 15
        self.labelsize = 15
        self.figsize = (8.27, 11.69)
        self.contour_cmap = "Greys"
        self.contour_norm = Normalize()
        self.fig = None
        self.axarr = None

        # The dimension of the plot within the figure
        self.top = 0.97
        self.bottom = 0.12
        self.left = 0.08
        self.right = 0.85

        # The unique values of the parameters
        self.uni1 = None
        self.uni2 = None
        self.uni3 = None
        self.uni4 = None

        if self.parameters != []:
            self.set_unique_parameters()

    def set(self, key, value):
        setattr(self, key, value)

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.set_unique_parameters()

    def make_figure(self):

        if len(self.parameters) != len(self.data):
            print "The number of models and data points do not agree!"
            print "Shape parameters: ", self.parameters.shape, " shape: data", self.data.shape
            sys.exit()

        # Get the row and column values
        x1_vals = np.sort(np.unique(self.parameters[:, self.x1_ax]))
        y1_vals = np.sort(np.unique(self.parameters[:, self.y1_ax]))[::-1]

        self.fig, self.axarr = plt.subplots(len(y1_vals), len(x1_vals), figsize=self.figsize, sharex=True, sharey=True)

        # Initialize contour levels and normalization
        if self.contour:
            if self.levels == []:
                min_val = np.nanmin(self.data)
                self.levels = [min_val + 10, min_val + 100, min_val + 1000]

            if isinstance(self.levels, (list, type(np.array([])))):
                self.contour_norm = LogNorm(np.min(self.levels), np.max(self.levels))
            elif isinstance(self.levels, int):
                self.contour_norm = LogNorm(np.nanmin(self.data), np.nanmax(self.data))

        if self.color_scale == "log":
            self.norm = LogNorm(vmin=self.vmin, vmax=self.vmax)
        elif self.color_scale == "lin" or self.color_scale == "linear":
            self.norm = Normalize(vmin=self.vmin, vmax=self.vmax)

        # Begin making subplots by looping over the unique values
        for y1i, y1 in enumerate(y1_vals):
            for x1i, x1 in enumerate(x1_vals):
                # find models with the right parameter values
                points_to_use = np.where((self.parameters[:, self.y1_ax] == y1) & (self.parameters[:, self.x1_ax] == x1))

                x = self.parameters[points_to_use, self.x2_ax][0]
                y = self.parameters[points_to_use, self.y2_ax][0]
                z = self.data[points_to_use]

                x_uni = np.sort(np.unique(self.parameters[:, self.x2_ax]))
                y_uni = np.sort(np.unique(self.parameters[:, self.y2_ax]))

                # put the values in an organized manner and fill up missing models.
                ordered_z = np.zeros((len(y_uni), len(x_uni)))
                for xi, xx in enumerate(x_uni):
                    x_points = x == xx
                    sort_order = np.argsort(y[x_points])

                    # in case of missing models, add a nan in the right place
                    temp_z = z[x_points][sort_order]
                    y_points = y[x_points][sort_order]
                    for i, y_point in enumerate(y_uni):
                        if y_point not in y_points:
                            temp_z = np.insert(temp_z, i, np.nan)

                    ordered_z[:, xi] = temp_z

                # adjust the grid such that all data points actually show up
                # center of the "bins" is the parameter value
                fixed_x, xscale = self.adjust_grid(x_uni, return_scale=True)
                fixed_y, yscale = self.adjust_grid(y_uni, return_scale=True)

                x_grid, y_grid = np.meshgrid(fixed_x, fixed_y)

                cm = self.axarr[y1i][x1i].pcolormesh(x_grid, y_grid, ordered_z, cmap=self.cmap, norm=self.norm)

                if self.contour:
                    self.add_contour(self.axarr[y1i][x1i], x_uni, y_uni, ordered_z, fixed_x, fixed_y)

                # fix for saving as pdf
                cm.set_edgecolor("face")
                # This label is to find back the right model parameters in the interactive parts
                self.axarr[y1i][x1i].set_label("%s,%s" % (x1, y1))
                self.axarr[y1i][x1i].set_xscale(xscale)
                self.axarr[y1i][x1i].set_yscale(yscale)
                self.axarr[y1i][x1i].set_xlim(np.min(fixed_x), np.max(fixed_x))
                self.axarr[y1i][x1i].set_ylim(np.min(fixed_y), np.max(fixed_y))
                self.axarr[y1i][x1i].tick_params(direction="inout", bottom=True, left=True, right=True, top=True,
                                            which="both", labelsize=self.labelsize)

                if y1i == 0:
                    self.axarr[y1i][x1i].set_title(r"%s %.3g%s" % (self.x1_label, x1, self.x1_unit),
                                                   fontsize=self.fontsize)

        # The dimension of the plot within the figure, relative to figure size
        # This does not include labels or ticks. If anything needs moving change these values.
        top = self.top        # highest point on the top
        bottom = self.bottom  # lowest point on the bottom
        left = self.left      # most left point
        right = self.right    # most right point

        # add the model values on the "y1 axis" parameters.
        for i, y1 in enumerate(y1_vals[::-1]):
            height = bottom + i * (top - bottom) / len(y1_vals) + (top - bottom) / len(y1_vals) / 2.
            self.fig.text(right + 0.005, height, r"%s %.3g %s" % (self.y1_label, y1, self.y1_unit), rotation="horizontal",
                     va="center", fontsize=self.fontsize)

        # The "x2" and "y2" labels
        self.fig.text((left + right) / 2., bottom - 0.045, self.x2_label, ha="center", fontsize=self.fontsize)
        self.fig.text(0.01, (0.5 * (top + bottom)), self.y2_label, va="center", rotation="vertical", fontsize=self.fontsize)

        # Add the colorbar
        self.fig.subplots_adjust(hspace=0, wspace=0, bottom=bottom, right=right, left=left, top=top)
        cbar_ax = self.fig.add_axes([left, bottom - 0.075, right - left, 0.02])
        cb = self.fig.colorbar(cm, cax=cbar_ax, orientation="horizontal")
        cb.set_label(self.color_label, labelpad=-1, fontsize=self.fontsize)
        cb.ax.tick_params(labelsize=self.labelsize)

        # The interactive stuff.
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect('key_press_event', self.ontype)

    def onclick(self, event):
        # put a dot where the user clicks.
        toolbar = plt.get_current_fig_manager().toolbar
        # fig = plt.gcf()
        if event.button == 1 and toolbar.mode == '':
            if event.inaxes is not None:
                ax = event.inaxes
                x = event.xdata
                y = event.ydata
                point = ax.plot(x, y, "ro", ms=3, picker=5, label="show_point")
                # only draw the new point
                ax.draw_artist(point[0])
                self.fig.canvas.blit(ax.bbox)

    def onpick(self, event):
        # when the user clicks right on a point, remove it
        # fig = plt.gcf()
        if event.mouseevent.button == 3:
            if hasattr(event.artist, 'get_label') and event.artist.get_label() == 'show_point':
                event.artist.remove()
                self.fig.canvas.draw_idle()

    def ontype(self, event):
        # Find which gridpoints have been clicked and call the plotting function
        if event.key == "enter":

            model_properties = {0: [], 1: [], 2: [], 3: []}
            # axes = plt.gcf().axes
            for axar in self.axarr:
                for ax in axar:
                    label = ax.properties()["label"].split(",")
                    if len(label) > 1:
                        x1, y1 = ax.properties()["label"].split(",")

                        x1 = float(x1)
                        y1 = float(y1)

                        for artist in ax.get_children():
                            if hasattr(artist, 'get_label') and artist.get_label() == 'show_point':
                                x2, y2 = artist.get_data()

                                model_properties[self.x1_ax].append(x1)
                                model_properties[self.y1_ax].append(y1)
                                model_properties[self.x2_ax].append(x2)
                                model_properties[self.y2_ax].append(y2)

            model_indices = self.get_model_indices(model_properties)
            self.plot_function(model_indices, *self.plot_args, **self.plot_kwargs)

        # Clear all points
        if event.key == "c":
            # axes = plt.gcf().axes
            for axar in self.axarr:
                for ax in axar:
                    for artist in ax.get_children():
                        if hasattr(artist, 'get_label') and artist.get_label() == 'show_point':
                            artist.remove()
            plt.draw()

    def closest(self, arr, val):
        return arr[np.abs(arr - val).argmin()]

    def get_model_indices(self, parameter_vals):
        """
        Gets the indices of the models closest to parameter_vals.
        :param parameter_vals: the clicked parameter values.
        :return: list of the indices.
        """

        model_indices = []
        for i in range(len(parameter_vals[0])):

            val1 = self.closest(self.uni1, parameter_vals[0][i])
            val2 = self.closest(self.uni2, parameter_vals[1][i])
            val3 = self.closest(self.uni3, parameter_vals[2][i])
            val4 = self.closest(self.uni4, parameter_vals[3][i])

            model_index = np.where((val1 == self.parameters[:, 0]) &
                                   (val2 == self.parameters[:, 1]) &
                                   (val3 == self.parameters[:, 2]) &
                                   (val4 == self.parameters[:, 3]))
            model_indices.append(model_index[0])

        return np.array(model_indices).reshape(-1)

    def add_contour(self, ax, x, y, z, plot_range_x, plot_range_y):
        """
        Adds contours to the matplotlib axes given by ax. Also interpolates in the plot_range for extra smoothness.
        :return:
        """
        f = interp2d(x, y, z, kind="linear")
        new_x = np.geomspace(plot_range_x[0], plot_range_x[-1], 50)
        new_y = np.geomspace(plot_range_y[0], plot_range_y[-1], 50)
        new_z = f(new_x, new_y)

        ax.contour(new_x, new_y, new_z, self.levels, cmap=self.contour_cmap, norm=self.contour_norm)

    def adjust_grid(self, x, log=None, return_scale=False):
        """
        Adjusts the grid such that the gridpoints fall in the middle of the "bins". Compatible with log spaced and
        linear spaced data, has to be uniform in either of accurate effect. Can specify the space, but can also figure
        it out by itself. If return_scale=True the the "new" grid and a string indicating a "linear" or "log" scale.
        :param x:
        :param log:
        :param return_scale:
        :return:
        """

        if log is None:
            if np.any(np.round(np.diff(x), 2) != np.round(np.diff(x)[0], 2)):
                log = True
            else:
                log = False

        # assumes all values are evenly spaced in log space
        if log:  # log == True
            diff = np.log10(x[1]) - np.log10(x[0])
            new_x = np.logspace(np.log10(x[0]) - diff / 2, np.log10(x[-1]) + diff / 2, len(x) + 1)
        # Assume uniformly spaced values
        else:  # log != True so anything but
            diff = x[1] - x[0]
            new_x = np.linspace(x[0] - diff / 2, x[-1] + diff / 2, len(x) + 1)

        if return_scale:
            if log:
                return new_x, "log"
            else:
                return new_x, "linear"
        else:
            return new_x


