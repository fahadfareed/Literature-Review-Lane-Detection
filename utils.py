import numpy as np
import csv

y_ = [720, 667.82611084, 627.69232178, 595.86206055, 570, 548.57141113, 530.52630615, 515.12194824, 501.81817627, 490.21276855, 480]

def remove_outliers(coords):
    elements = np.array([c[0] for c in coords])
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    return  [c for c in coords if (c[0] > mean - 2 * sd and c[0] < mean + 2* sd)]

def apply_polynomial(coords, order=2):
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    z = np.polyfit(y, x, order)
    p = np.poly1d(z)

    return [(p(y), y) for y in y_]

def create_grid(coordinates, shape=(11, 32), size_x=1280, size_y=720, res=32.0):
    grid = np.zeros(shape=shape)

    for c in coordinates:
        x = int(c[0] / 40.0)
        y = int((720 - c[1]) / 22.5)
        if x>31: x=31
        if y>10: y=10
        grid[y][x] = 1

    return [g for gr in grid for g in gr]

def read_csv(file):
    data = []
    with open(file) as f:
        reader = csv.reader(f)

        for row in reader:
            data.append(row)

    return data
