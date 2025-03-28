import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from tqdm import tqdm

mi.set_variant('scalar_rgb')

scene = mi.load_file('/home/leon/dev/deep-structure/scenes/test.xml')

camera_size = mi.Point2u(128, 128)

sensor = scene.sensors()[0]
sampler = sensor.sampler()
medium = None
sample_count = sampler.sample_count()
print(sample_count)
integrator = scene.integrator()
active = True
sampler.seed(0)
result = dr.zeros(dr.scalar.ArrayXf, camera_size.x * camera_size.y * 3)
diff_scale_factor = dr.rsqrt(float(sample_count))

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

        spec, mask, aov = integrator.sample(scene, sampler, ray, medium, active)
        rgb = spec * ray_weight
        result[i * 3] += rgb.x
        result[i * 3 + 1] += rgb.y
        result[i * 3 + 2] += rgb.z

image = mi.TensorXf(result, shape=(camera_size.x, camera_size.y, 3))

plt.imshow(image / dr.max(image)) 
plt.axis('off')
plt.show()