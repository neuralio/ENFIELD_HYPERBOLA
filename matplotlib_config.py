import matplotlib.pyplot as plt
import logging

# Suppress Matplotlib's font search debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Define global rcParams
def configure_matplotlib():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Ubuntu',
        'font.monospace': 'Ubuntu Mono',
        'font.size': 10,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.linewidth': 0.5,  
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,  
        'ytick.major.size': 6,  
        'axes.titlesize': 'large',  
        'axes.labelsize': 'large',  
        'xtick.labelsize': 'medium',  
        'ytick.labelsize': 'medium' ,
        'axes.grid': False,#True,
        'grid.color': 'black',
        'axes.grid.axis': 'both',#'y', #show only horizontal grid
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        'axes.facecolor': '#f9f6f0',
        'figure.facecolor': '#f9f6f0',
        'figure.edgecolor': '#f9f6f0',
        'figure.figsize': '15, 8',
        'legend.markerscale': 1.5,
        'legend.edgecolor': '#757575',
        'legend.facecolor': '#EEEEEE',
        'legend.framealpha': 0.9,
        'legend.labelcolor': '#555555',
        'legend.title_fontsize': 11,

    })

COLORS = ["#EF233C", "#233cef", "#006a6f", "#ef7023", "#efd623", "#8D99AE", "#2B2F42"]