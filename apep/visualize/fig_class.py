


import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import PIL.Image as Image
import io



class FigClass(object):
    def __init__(self, num_rows, num_columns, dpi=100):
        fig, axes = plt.subplots(num_rows, num_columns, dpi=dpi)
        if num_rows == 1 and num_columns == 1:
            axes = np.array([axes])
        self.fig = fig
        self.axes = axes
        self.num_rows = num_rows
        self.num_columns = num_columns

    def resize(self, scale=0.05):
        # self.fig.set_tight_layout(True)
        # space = min(self.fig.subplotpars.wspace, self.fig.subplotpars.hspace)
        space = 0.0
        w = (self.num_columns + (self.num_columns+1)*space) * max([np.diff(np.array(ax.get_xlim()))[0] for ax in self.axes.flatten()])
        h = (self.num_rows + (self.num_rows+1)*space) * max([np.diff(np.array(ax.get_ylim()))[0] for ax in self.axes.flatten()])

        self.fig.set_size_inches(w  *scale, h  *scale)
        # self.fig.subplots_adjust(
        #     left=0.1,  # Left margin
        #     right=0.9,  # Right margin
        #     bottom=0.1,  # Bottom margin
        #     top=0.9,  # Top margin
        #     wspace=0.1,  # Width space between subplots
        #     hspace=0.1  # Height space between subplots
        # )

    def align(self):
        for ax in self.axes.flatten():
            ax.set_aspect('equal')

    def set_fig_title(self, title, fontsize=None, y=None):
        self.fig.suptitle(title, fontsize=fontsize, y=y)



    def array_old(self):
        """
            fig = plt.figure()
            image = array(fig)
        """
        # draw the renderer
        self.fig.canvas.draw()
    
        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)[:,:,:3]
        return image


    def array(self):
        # Ensure the figure is drawn
        self.fig.canvas.draw()

        # Create a BytesIO buffer to save the figure
        buffer = io.BytesIO()

        # Save the figure to the buffer using the print_figure method, which allows
        # for bounding box adjustments (bbox_inches='tight' crops the figure as desired)
        self.fig.canvas.print_figure(buffer, format='png', bbox_inches='tight', pad_inches=0.1)

        # Retrieve the PNG buffer and open it with PIL
        buffer.seek(0)
        image = Image.open(buffer)
        
        # Convert the PIL Image to a NumPy array
        image_array = np.asarray(image)

        # Close the buffer stream
        buffer.close()

        # Optionally, remove the alpha channel if you are only interested in RGB
        image_array = image_array[:,:,:3]

        return image_array

    def tight_layout(self):
        self.fig.tight_layout()

    def clear(self):
        [ax.clear() for ax in self.axes.flatten()]

    def show(self):
        self.resize()
        plt.show()

    def save(self, path):
        self.resize()
        self.fig.savefig(path, bbox_inches='tight')

    def close(self):
        plt.close()

