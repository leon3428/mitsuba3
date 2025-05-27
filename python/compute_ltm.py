import mitsuba as mi
import drjit as dr
import numpy as np
import time
import scipy

mi.set_variant('cuda_mono')
scene = mi.load_file('scenes/test.xml')

start_t = time.time()

sensor_size = (1024, 1024)
projector_size = (1024, 1024)

dataset_generator = mi.DatasetGenerator(5, False, sensor_size, projector_size)
dataset_generator.render(scene)

