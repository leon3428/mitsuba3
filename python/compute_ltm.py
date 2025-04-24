import mitsuba as mi
import drjit as dr
import numpy as np
import time
import scipy

mi.set_variant('llvm_mono')
scene = mi.load_file('scenes/test.xml')

start_t = time.time()

sensor_size = (32, 32)
projector_size = (32, 32)

sensor = scene.sensors()[0]
sampler = sensor.sampler()
medium = None
sample_count = sampler.sample_count()

integrator = mi.LTM(sensor_size[0], sensor_size[1],
                    projector_size[0], projector_size[1], 5, False)

active = True
sampler.set_samples_per_wavefront(sample_count)
wavefront_size = sensor_size[0] * sensor_size[1] * sample_count
sampler.seed(mi.UInt(0), wavefront_size)
diff_scale_factor = dr.rsqrt(float(sample_count))

idx = dr.arange(mi.UInt32, wavefront_size) // sample_count
pos = mi.Point2i()
pos.y = idx // sensor_size[0]
pos.x = idx - sensor_size[0] * pos.y

scale = 1.0 / mi.ScalarVector2f(sensor_size[0], sensor_size[1])
sample_pos = pos + sampler.next_2d(active)
adjusted_pos = sample_pos * scale

aperture_sample = mi.Point2f(0.5)
if sensor.needs_aperture_sample():
    aperture_sample = sampler.next_2d(active)

ray_time = sensor.shutter_open()
if sensor.shutter_open_time() > 0.0:
    ray_time += sampler.next_1d() * sensor.shutter_open_time()

ray, ray_weight = sensor.sample_ray_differential(
    ray_time, 0.0, adjusted_pos, aperture_sample)
if ray.has_differentials:
    ray.scale_differential(diff_scale_factor)

values, us, vs, mask = integrator.sample(scene, sampler, ray, pos, active)

end_t = time.time()
print(end_t - start_t)

values = values.numpy().reshape(wavefront_size * 4)
us = us.numpy().reshape(wavefront_size * 4)
vs = vs.numpy().reshape(wavefront_size * 4)
mask = (values != 0) & (us >= 0) & (us <= 1) & (vs >= 0) & (vs <= 1)

us = np.astype(us * (projector_size[0] - 1), np.uint32)
vs = np.astype((1.0 - vs) * (projector_size[1] - 1), np.uint32)

pos = pos.numpy()
row = pos[0] * sensor_size[0] + pos[1]
row = np.tile(row, 4)
col = us * projector_size[0] + vs

row = row[mask]
col = col[mask]
values = values[mask]

lt = scipy.sparse.coo_matrix((values, (row, col)), shape=(sensor_size[0]*sensor_size[1], projector_size[0]*projector_size[1]))
lt.sum_duplicates()
scipy.sparse.save_npz("ltm.npz", lt)