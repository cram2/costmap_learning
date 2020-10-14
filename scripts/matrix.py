import matplotlib.pyplot as plt
import numpy as np

from geometry_msgs.msg import Point

class OutputMatrix(object):
    """This class mainly saves boundaries in a two dimensional space.
    Moreover, it allows to visualizes discrete values saved in the matrix."""

    def __init__(self, x, y, width, height):
        self.resolution = 0.01
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.matrix = []

    def copy(self):
        return OutputMatrix(self.x, self.y, self.width, self.height)

    def insert(self, m):
        self.matrix = m

    def empty(self):
        return self.matrix is None

    def get_boundries(self, other):
        x0 = min(self.x, other.x)
        x1 = max(self.x + self.width, other.x + other.width)
        y0 = min(self.y, other.y)
        y1 = max(self.y + self.height, other.y + other.height)
        return OutputMatrix(x0, y0, abs(x1-x0), abs(y1-y0))

    def set_ros_costmap_response(self, response):
        if response:
            response.bottem_lefts.append(Point(float(self.x), float(self.y), float(0.0)))
            response.widths.append(float(self.width))
            response.heights.append(float(self.height))
            return response

    def plot(self, text="", name=None, plot_in_other=False):
        plt.title(text)
        plt.imshow(self.matrix, extent=[self.x, self.x + self.width, self.y, self.y + self.height], alpha=0.5)
        if not plot_in_other:
            plt.colorbar()
        # plots the table borders
        plt.plot([-1.32, -0.515], [0.565, 0.565], 'k-', lw=2) # bottom right to top right
        plt.plot([-0.515, -0.515], [0.565, 3.02], 'k-', lw=2) # top right to top left
        plt.plot([-0.515, -1.32], [3.02, 3.02], 'k-', lw=2) # top left to bottom left
        plt.plot([-1.32, -1.32], [3.02, 0.565], 'k-', lw=2) # bottom left to bottom right
        if name:
            plt.savefig(name, pad_inches=0.2, bbox_inches='tight')
            plt.close()
        else:
            if not plot_in_other:
                plt.show()

    #deprecated
    def get_intersected_boundary(self, other):
        leftX = max(self.x, other.x)
        rightX = min(self.x + self.width, other.x + other.width)
        bottomY = max(self.y, other.y)
        topY = min(self.y + self.height, other.y + other.height)
        if (leftX < rightX and bottomY < topY):
            return OutputMatrix(leftX, bottomY, abs(rightX - leftX), abs(topY - bottomY))
        else:
            return None
        r_x0, y_x0, r_to_x1, r_to_y1 = [-1, -1, -1, -1]
        if self.x < other.x and self.y < other.y:
            r_x0 = other.x
            r_y0 = other.y
        else:
            r_x0 = self.x
            r_y0 = self.y
        if x1 <= other.x + other.width and y1 <= other.y + other.height:
            r_to_x1 = abs((other.x + other.width) - x1)
            r_to_y1 = abs((other.y + other.height) - y1)
        else:
            r_to_x1 = abs(x1 - (other.x + other.width))
            r_to_y1 = abs(y1 - (other.y + other.height))
        width = (x1 - r_to_x1) - r_x0
        height = (y1 + r_to_y1) - r_y0
        if width > 0 and height > 0:
            return OutputMatrix(r_x0, r_y0, width, height)

    #deprecated
    def get_value(self, other, x_i, y_i):
        if other and x_i >= 0 and y_i >= 0:
            return self.matrix[int((other.x - self.x) / self.resolution) + x_i][
                int((other.y - self.y) / self.resolution) + y_i]

    #deprecated
    @staticmethod
    def summarize(output_matrices):
        x0 = min(list(map(lambda o: o.x, output_matrices)))
        y0 = min(list(map(lambda o: o.y, output_matrices)))
        x1 = max(list(map(lambda o: o.x + o.width, output_matrices)))
        y1 = max(list(map(lambda o: o.y + o.height, output_matrices)))
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        sum = OutputMatrix(x0, y0, w, h)
        s = int(w / sum.resolution)
        z = int(h / sum.resolution)
        m = np.zeros((z, s))
        for output_matrix in output_matrices:
            x_i0 = int(abs((output_matrix.x - x0) / sum.resolution))
            y_i0 = int(abs((output_matrix.y - y0) / sum.resolution))
            x_steps = output_matrix.matrix.shape[1]
            y_steps = output_matrix.matrix.shape[0]
            i_start = (z - y_i0) - y_steps
            i_end = z - y_i0
            j_start = x_i0
            j_end = x_i0 + x_steps
            m[i_start:i_end, j_start:j_end][:] += output_matrix.matrix[:]
        # m /= m.max()
        sum.insert(m)
        return sum