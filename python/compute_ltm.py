import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time

mi.set_variant('cuda_mono')
scene = mi.load_file('scenes/test.xml')

start_t = time.time()

sensor_size = (128, 128)
projector_size = (128, 128)

sensor = scene.sensors()[0]
sampler = sensor.sampler()
medium = None
sample_count = sampler.sample_count()

integrator = mi.LTM(sensor_size[0], sensor_size[1],
                    projector_size[0], projector_size[1], 6, 5, False)

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

spec, mask = integrator.sample(scene, sampler, ray, pos, active)

ltm = spec.numpy()
np.savez_compressed('python/tmp', ltm=ltm,
                    sensor_size=sensor_size, projector_size=projector_size)

end_t = time.time()
print(end_t - start_t)
