import multiprocessing
import numpy as np
from perlin_noise import generate_fractal_noise_3d
from renderer import Renderer
from tqdm import tqdm
import tensorflow as tf
from radon_tf import radon_parabeam

MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER


# def post_process(q):
#     counter = 0
#     arr = []
#     while type(render := q.get()) is not str:
#         arr.append(render)
#         if len(arr) == 256:
#             np.save(f"out1/sinograms/sinogram_{counter}.npy", np.array(arr))
#             arr = []
#             counter += 1
#
#
# def generate_noise(block_control, q):
#     while (i := block_control.get()) != "stop":
#         block = tf.clip_by_value(generate_fractal_noise_3d((256, 256, 256), res=(2, 2, 2), octaves=3), 0, 1).numpy()
#         np.save(f"out1/ground_truth/gt_{i}.npy", block)
#         q.put(block)


def generate_data_tf(batch_size, img_size, num_angles, num_detectors, num_photons, size):
    imgs = tf.clip_by_value(generate_fractal_noise_3d((batch_size, img_size, img_size), res=(2, 2, 2), octaves=3), 0, 1)
    imgs = imgs * MU_MAX
    sinogram = radon_parabeam(imgs, num_angles, num_detectors, size)

    # noise addition
    sinogram = tf.squeeze(tf.random.poisson((1,), tf.exp(-1 * sinogram) * num_photons) / num_photons, axis=0)
    sinogram = tf.math.log(tf.clip_by_value(sinogram, clip_value_min=0.1 / num_photons, clip_value_max=tf.float32.max))

    return imgs, sinogram


def main():
    import matplotlib.pyplot as plt
    imgs, sinograms = generate_data_tf(16, 256, 256, 256, 4096, 0.13)
    plt.imshow(imgs[0])
    plt.show()
    plt.imshow(sinograms[0])
    plt.show()
    # renderer = Renderer(num_angles=None, num_det=None)
    #
    # pool = multiprocessing.Pool(processes=12)
    # for i in tqdm(range(40)):
    #     block = tf.clip_by_value(generate_fractal_noise_3d((256, 256, 256), res=(2, 2, 2), octaves=3), 0, 1).numpy()
    #     block = block * MU_MAX
    #     res = pool.map_async(renderer.render, block)
    #     np.save(f"out/ground_truth/gt_{i}.npy", block)
    #     arr = res.get()
    #     np.save(f"out/sinograms/sinogram_{i}.npy", np.array(arr) / (-MU_MAX))
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    main()
