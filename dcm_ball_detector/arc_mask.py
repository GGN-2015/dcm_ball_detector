import numpy as np
import functools

def arc_f(x):
    return (x - 256)**2/256 + 200

@functools.cache
def get_arch():
    arc_image = np.ones((512, 512))
    for x in range(512):
        for y in range(512):
            if y >= arc_f(x):
                arc_image[y, x] = 0
    return arc_image