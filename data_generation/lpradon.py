import tensorflow as tf
from keras.layers import *  # todo
import numpy as np


class LPRadon(Layer):
    def __init__(self, n_angles, n_det=None, n_span=3, cor=None, interp_type='cubic', *args, **kwargs):
        super(LPRadon, self).__init__(*args, **kwargs)

        self.cor = cor  # I have no idea what this variable does. It should be n // 2
        self.interp_type = interp_type
        self.n_span = n_span
        self.n_angles = n_angles
        self.n_det = n_det

        self.n = None
        self.batch_size = None
        self.beta = None
        self.a_R = None
        self.a_m = None
        self.g = None

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert(h == w, "Image must be square")
        self.n = h.numpy()
        self.batch_size = b.numpy()

        self.cor = self.cor or self.n // 2

    def pre_compute(self):
        self.beta = np.pi / self.n_span

        # expand the image so that we have enough room to rotate
        self.n = int(np.ceil(self.n + abs(self.n / 2 - self.cor) / 8.) * 16.)
        self.n_det = self.n_det or self.n

        # oversample, this will change the shape of the image
        os_angles = int(max(round(3. * self.n / 2 / self.n_angles), 1))
        self.n_angles = os_angles * self.n_angles

        # polar grid
        angles = np.arange(0, self.n_angles) * np.pi / self.n_angles - self.beta / 2
        s = np.linspace(-1, 1, self.n_det)  # idk why one is np.arange and the other is np.linspace

    def get_lp_params(self):
        self.a_R = np.sin(self.beta / 2) / (1 + np.sin(self.beta / 2))
        self.a_m = (np.cos(self.beta / 2) - np.sin(self.beta / 2)) / (1 + np.sin(self.beta / 2))\

        t = np.linspace(-np.pi / 2, np.pi / 2, 1000)
        w = self.a_R * np.cos(t) + (1 - self.a_R) * 1j * self.a_R * np.sin(t)
        self.g = max(np.log(abs(w)) + np.log(np.cos(self.beta / 2 - np.arctan2(w.imag, w.real))))

        n_th = self.n_det
        n_rho = 2 * self.n_det