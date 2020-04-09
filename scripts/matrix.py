import matplotlib.pyplot as plt
import numpy as np

from costmap_learning.srv import GetCostmapResponse
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

class OutputMatrix(object):

    def __init__(self, x, y, width, height):
        self.resolution = 0.01
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.matrix = []
        self.angles = []

    def copy(self):
        return OutputMatrix(self.x, self.y, self.width, self.height)

    def insert(self, m):
        self.matrix = m

    def empty(self):
        return self.matrix is None

    def get_value(self, other, x_i, y_i):
        if other and x_i >= 0 and y_i >= 0:
            return self.matrix[int((other.x - self.x) / self.resolution) + x_i][
                int((other.y - self.y) / self.resolution) + y_i]

    def merge(self, other, normalize_factor=2.0):
        if other:
            intersected_area = self.get_boundries(other)
            #print(intersected_area.x, intersected_area.y, intersected_area.width, intersected_area.height
            #      )
            #print(self.resolution)
            if intersected_area:
                #print("to intersect")
                #print(self.x, self.y, self.width, self.height)
                #print(other.x, other.y, other.width, other.height)
                cols = abs(int(intersected_area.width / self.resolution))
                rows = abs(int(intersected_area.height / self.resolution))
                #print(rows)
                #print(cols)
                #print("intersected")
                #print(intersected_area.x, intersected_area.y, intersected_area.width, intersected_area.height)
                m = np.zeros((rows, cols))
                for row in range(0, rows):
                    for col in range(0, cols):
                        #print("col %d, row %d" % (col, row))
                        m[col][row] = (self.get_value(intersected_area, col, row)
                                       * other.get_value(intersected_area, col, row)) \
                                      / normalize_factor
                result = intersected_area.copy()
                result.insert(m)
                return result
            else:
                return None
                #raise ValueError("there was no intersected area merging the output matrix")

    @staticmethod
    def get_ros_costmap_response(output_matrices=None):
        # matrix as a list of vectors which are in the matrix the
        # "rows"
        response = GetCostmapResponse()
        for output_matrix in output_matrices:
            # print(output_matrix)
            response.bottem_lefts.append(Point(float(output_matrix.x), float(output_matrix.y), float(0.0)))
            response.widths.append(float(output_matrix.width))
            response.heights.append(float(output_matrix.height))
        response.resolution = float(output_matrices[0].resolution)
        output_matrix = OutputMatrix.summarize(output_matrices)
        tmp = np.array(output_matrix.matrix).astype(np.float)
        response.x_y_vecs = list(tmp.flatten())
        response.global_width = float(output_matrix.width)
        response.global_height = float(output_matrix.height)
        # print(response.x_y_vecs)
        # response.angles = [] # self.angles)
        return response

    @staticmethod
    def merge_matrices(output_matrices):
        if len(output_matrices) == 1:
            return output_matrices[0]
        else:
            merged = None
            for i in range(0, len(output_matrices)):
                if i == len(output_matrices) - 1:
                    return merged
                merged = output_matrices[i].merge(output_matrices[i + 1])
            return merged

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


    def colliding(self, other):
        return True

    def get_boundries(self, other):
        x0 = min(self.x, other.x)
        x1 = max(self.x + self.width, other.x + other.width)
        y0 = min(self.y, other.y)
        y1 = max(self.y + self.height, other.y + other.height)
        return OutputMatrix(x0, y0, abs(x1-x0), abs(y1-y0))

    def get_intersected_boundary(self, other):
        if self.colliding(other):
            #x1 = self.x + self.width
            #y1 = self.y + self.height
            #left1 = self.x
            #right1 = self.x + self.width
            #top1 = self.y
            #bottom1 = self.y + self.height
            #left2 = other.x
            #right2 = other.x + other.width
            #top2 = other.y
            #bottom2 = other.y + other.height
            #x_overlap = max(0, abs(min(right1, right2) - max(left1, left2)))
            #y_overlap = max(0, abs(min(bottom1, bottom2) - max(top1, top2)))
            leftX = max(self.x, other.x)
            rightX = min(self.x + self.width, other.x + other.width)
            bottomY = max(self.y, other.y)
            topY = min(self.y + self.height, other.y + other.height)
            if (leftX < rightX and bottomY < topY):
                return OutputMatrix(leftX, bottomY, abs(rightX - leftX), abs(topY - bottomY))
            else:
                return None
                # raise ValueError("there was no intersected area between the output matrices")
            r_x0, y_x0, r_to_x1, r_to_y1 = [-1, -1, -1, -1]
            if self.x < other.x and self.y < other.y:
                r_x0 = other.x
                r_y0 = other.y
                # if x1 > other.x + other.width and y1 > other.y + other.height:
                #    return OutputMatrix(r_x0, r_y0, other.width, other.height)
            else:
                r_x0 = self.x
                r_y0 = self.y
                #if other.x + other.width > x1 and other.y + other.height:
                 #   return OutputMatrix(r_x0, r_y0, self.width, self.height)

            if x1 <= other.x + other.width and y1 <= other.y + other.height:
                r_to_x1 = abs((other.x + other.width) - x1)
                r_to_y1 = abs((other.y + other.height) - y1)
            else:
                r_to_x1 = abs(x1 - (other.x + other.width))
                r_to_y1 = abs(y1 - (other.y + other.height))
            return OutputMatrix(r_x0, r_y0, (x1 - r_to_x1) - r_x0, (y1 + r_to_y1) - r_y0)

    def plot(self, text, name=None):
        plt.title(text)
        plt.imshow(self.matrix, extent=[self.x, self.x + self.width, self.y, self.y + self.height], alpha=0.5)
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
            plt.show()
