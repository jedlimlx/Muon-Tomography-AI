import odl
import numpy as np
from multiprocessing import Queue


class Renderer:
    """
    Renders 2D slices
    """

    def __init__(self, res=256, size: float = 0.13, num_angles: int = None, num_det: int = None, impl="astra_cpu",
                 seed=None, n_photons=4096):
        """
        Args:
            res: Resolution of 2D slices
            size: Physical size of simulation geometry
            num_angles: Number of angles to simulate (decided by ODL if None)
            num_det: Number of detectors (decided by ODL if None)
            impl: Rendering method ("astra_cuda" or "astra_cpu")
            seed: seed for random number generator
            n_photons: number of photons per pixel
        """
        space = odl.uniform_discr(min_pt=(-size, -size), max_pt=(size, size), shape=(res, res))
        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=num_angles, det_shape=num_det)
        self.ray_tracer = odl.tomo.RayTransform(space, geometry, impl=impl)
        self.rng = np.random.default_rng(seed)
        self.n_photons = n_photons
        self.impl = impl
        self.size = size
        self.res = res

    def render(self, img):
        """
        Renders a single slice
        Args:
            img: 2D slice

        Returns: Rendered image

        """
        out = self.ray_tracer(img).asarray().astype(np.float64)

        out = self.rng.poisson(np.exp(-1 * out) * self.n_photons) / self.n_photons
        out = np.log(np.clip(out, a_min=0.1 / self.n_photons, a_max=None))
        return out

    def render_q(self, in_q: Queue, out_q: Queue):
        img = in_q.get()
        while type(img) is not str:
            # print(np.mean(img * self.MU_MAX), np.std(img * self.MU_MAX))
            # if self.impl == "astra_cuda":
            #     img = img * self.size * 2 / self.res
            out_q.put(self.render(img))
            img = in_q.get()

        out_q.put("stop")
