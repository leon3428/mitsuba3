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

start_t = time.time()

dataset_generator = mi.DatasetGenerator(5, False, 32, sensor_size, projector_size, 1, 1e-4)
dataset_generator.render("ltm.h5", scene)

end_t = time.time()
print(f"Time taken: {end_t - start_t:.2f} seconds")

# Before:
#   Time: 30.11 seconds
#   Size: 186M
# After:
#   Time: 25.83 seconds
#   Size: 111M