from grid_overview import *


def make_super_cool_plot(indices, important_argument1, important_argiment2, **kwargs):
    """
    A function that makes amazing plots.
    :param indices:                 array of integers
    :param important_argument1:     argument for test purposes
    :param important_argiment2:     another argument for test purposes
    :param kwargs:                  Key word arguments can be added if desired
    :return:                        None, just makes a plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, **kwargs)
    ax1.scatter([0] * len(indices), indices)
    ax2.plot(important_argument1, important_argiment2)
    plt.show()


# Generate a random grid of parameters of arbitrary size
pars = []
for i in np.linspace(1, 100, 5):
    for j in np.linspace(100, 1000, 5):
        for k in np.linspace(234, 1244, 5):
            for l in np.linspace(1, 5, 5):
                pars.append([i,j,k,l])

# For the example randomize the order of the parameters and leave some out (simulate faulty models)
pars = np.array(pars)
pars = pars[np.random.choice(range(len(pars)), size=len(pars) - 10, replace=False)]

# Generate data of same size
data = np.sum(pars, axis=1)

print pars.shape
print data.shape

# Generate some data for the plot of individual models
x = np.arange(10)
y = x**2

remove = np.logical_not((pars[:,0] == pars[125,0]) * (pars[:,2] == pars[139,2]))
pars = pars[remove]
data = data[remove]

# initialize the figure
a = OverviewPlot(parameters=pars, data=data, plot_kwargs={"figsize": (4, 8)}, plot_function=make_super_cool_plot,
                 plot_args=[x, y], color_scale="lin", contour=False)

# Can change any of the values (and some more) after initialization 
a.set("x1_ax", 0)
a.set("x2_ax", 1)
a.set("y1_ax", 2)
a.set("y2_ax", 3)
a.set("x1_label", "x1=")
a.set("x2_label", "x2=")
a.set("y1_label", "y1=")
a.set("y2_label", "y2=")
a.x1_unit = ""
a.y1_unit = ""
a.fontsize=None
a.labelsize=None

# make and show the figure
a.make_figure()
plt.show()
