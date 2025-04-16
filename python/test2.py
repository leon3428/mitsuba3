import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

mi.set_variant('scalar_mono')

scene = mi.load_file('/home/leon/dev/deep-structure/scenes/test.xml')

camera_size = mi.Point2u(64, 64)
projector_size = mi.Point2u(64, 64)

sensor = scene.sensors()[0]
sampler = sensor.sampler()
medium = None
sample_count = sampler.sample_count()

# integrator = scene.integrator()
integrator = mi.LTM(projector_size.x, projector_size.y, 6, 5, False)

active = True
sampler.seed(0)
result = dr.zeros(dr.scalar.ArrayXf, camera_size.x * camera_size.y * projector_size.x * projector_size.y)
diff_scale_factor = dr.rsqrt(float(sample_count))

tmp = projector_size.x * projector_size.y

for i in tqdm(range(camera_size.x * camera_size.y)):
    pos = mi.Point2u(i % camera_size.x, mi.UInt(i / camera_size.x))
    frac_pos = mi.Point2f(pos.x / camera_size.x, pos.y / camera_size.y)
    
    for j in range(sample_count):
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0.0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        scale = 1.0 / camera_size
        adjusted_pos = frac_pos + sampler.next_2d(active) * scale

        aperture_sample = mi.Point2f(0.5)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d(active)

        ray, ray_weight = sensor.sample_ray_differential(time, 0.0, adjusted_pos, aperture_sample)
        if ray.has_differentials:
            ray.scale_differential(diff_scale_factor)

        # spec, mask, aov = integrator.sample(scene, sampler, ray, medium, active)
        # rgb = spec * ray_weight
        # result[i * 3] += rgb.x
        # result[i * 3 + 1] += rgb.y
        # result[i * 3 + 2] += rgb.z

        spec, mask = integrator.sample(scene, sampler, ray, active)
        index = dr.arange(dr.scalar.ArrayXu64, start = i * tmp, stop = (i + 1) * tmp)
        dr.scatter_add(result, spec, index)

ltm = result.numpy()
np.savez_compressed('python/tmp', ltm=ltm)