import mitsuba as mi
import drjit as dr
import numpy as np
import time
import scipy

mi.set_variant('cuda_mono')
scene = mi.load_file('scenes/test.xml')

start_t = time.time()

sensor_size = (128, 128)
projector_size = (128, 128)

renderer = mi.LightTransport(5, False)
values, rows, cols = renderer.render_light_transport(
    scene, sensor_size, projector_size)

lt = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(
    sensor_size[0]*sensor_size[1], projector_size[0]*projector_size[1]))
lt.sum_duplicates()
end_t = time.time()
print("Compute time: ", end_t - start_t)

start_t = time.time()
scipy.sparse.save_npz("ltm.npz", lt)
end_t = time.time()
print("Output time: ", end_t - start_t)
