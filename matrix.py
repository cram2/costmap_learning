import time

import matplotlib.pyplot as plt
import numpy as np


class OutputMatrix(object):

    def __init__(self, x, y, width, height):
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

    def get_value(self, other, x_i, y_i, resolution=0.02):
        if other and x_i >= 0 and y_i >= 0:
            return self.matrix[int((other.x - self.x) / resolution) + x_i][
                int((other.y - self.y) / resolution) + y_i]

    def merge(self, other, resolution=0.02, normalize_factor=2.0):
        if other:
            intersected_area = self.get_intersected_boundary(other)
            if intersected_area:
                #print("to intersect")
                #print(self.x, self.y, self.width, self.height)
                #print(other.x, other.y, other.width, other.height)
                x_steps = abs(int(intersected_area.width / resolution))
                y_steps = abs(int(intersected_area.height / resolution))
                #print("intersected")
                #print(intersected_area.x, intersected_area.y, intersected_area.width, intersected_area.height)
                m = np.zeros((x_steps, y_steps))
                for x_i in range(0, x_steps):
                    for y_i in range(0, y_steps):
                        m[x_i][y_i] = (self.get_value(intersected_area, x_i, y_i, resolution)
                                       * other.get_value(intersected_area, x_i, y_i, resolution)) \
                                      / normalize_factor
                result = intersected_area.copy()
                result.insert(m)
                return result
            else:
                raise ValueError("there was no intersected area merging the output matrix")

    @staticmethod
    def merge_matrices(output_matrices, resolution=0.02):
        if len(output_matrices) == 1:
            return output_matrices[0]
        else:
            merged = None
            for i in range(0, len(output_matrices)):
                if i == len(output_matrices) - 1:
                    return merged
                merged = output_matrices[i].merge(output_matrices[i + 1], resolution=resolution)
            return merged

    @staticmethod
    def summarize(output_matrices, resolution=0.02):
        x0 = min(list(map(lambda o: o.x, output_matrices)))
        y0 = min(list(map(lambda o: o.y, output_matrices)))
        x1 = max(list(map(lambda o: o.x + o.width, output_matrices)))
        y1 = max(list(map(lambda o: o.y + o.height, output_matrices)))
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        sum = OutputMatrix(x0, y0, w, h)
        s = w / resolution
        z = h / resolution
        s = int(s)
        z = int(z)
        m = np.zeros((z, s))
        n = len(output_matrices)
        for output_matrix_i in range(0, n):
            x_i0 = int(abs((output_matrices[output_matrix_i].x - x0) / resolution))
            y_i0 = int(abs((output_matrices[output_matrix_i].y - y0) / resolution))
            x_steps = output_matrices[output_matrix_i].matrix.shape[1]
            y_steps = output_matrices[output_matrix_i].matrix.shape[0]
            i_start = (z - y_i0) - y_steps
            i_end = z - y_i0
            j_start = x_i0
            j_end = x_i0 + x_steps
            m[i_start:i_end, j_start:j_end][:] = output_matrices[output_matrix_i].matrix[:]
        sum.insert(m)
        return sum


    def colliding(self, other):
        return True

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
                raise ValueError("there was no intersected area between the output matrices")
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

    def plot(self, text):
        plt.title(text)
        plt.imshow(self.matrix, extent=[self.x, self.x + self.width, self.y, self.y + self.height], alpha=0.5)
        plt.colorbar()
        plt.show()
