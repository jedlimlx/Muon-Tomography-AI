from multiprocessing import Queue, Process
import numpy as np
from perlin_noise import generate_fractal_noise_3d
from renderer import Renderer
from tqdm import tqdm


def post_process(q):
    counter = 0
    arr = []
    while type(render := q.get()) is not str:
        arr.append(render)
        if len(arr) == 256:
            np.save(f"renders_{counter}.npy", np.array(arr))
            arr = []
            counter += 1


def generate_noise(block_control, q):
    while (i := block_control.get()) != "stop":
        block = np.clip(generate_fractal_noise_3d((256, 256, 256), res=(2, 2, 2), octaves=3), a_min=0, a_max=1)
        np.save(f"ct_{i}.npy", block)
        q.put(block)


def main():
    renderer = Renderer(num_angles=256, num_det=256)
    block_control = Queue(2)
    block_q = Queue(1)
    in_q = Queue(10)
    out_q = Queue(10)
    render_thread = Process(target=renderer.render_q, args=(in_q, out_q))
    render_thread.start()
    post_thread = Process(target=post_process, args=(out_q,))
    post_thread.start()
    noise_thread = Process(target=generate_noise, args=(block_control, block_q))
    noise_thread.start()

    block_control.put(0)
    for i in tqdm(range(10)):
        block = block_q.get()
        if i != 9:
            block_control.put(i + 1)

        for img in block:
            in_q.put(img)

    in_q.put("stop")
    block_control.put("stop")
    render_thread.join()
    post_thread.join()


if __name__ == "__main__":
    main()
