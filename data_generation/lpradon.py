import tensorflow as tf
from keras.layers import *  # todo
import numpy as np
import tensorflow_addons as tfa
from scipy.signal.windows import cosine


class LPRadonBase(Layer):
    def __init__(self, n_angles, n_det=None, n_span=3, cor=None, interp_type='linear', *args, **kwargs):
        super(LPRadonBase, self).__init__(*args, **kwargs)

        self.cor = cor  # I have no idea what this variable does. It should be n // 2
        self.interp_type = interp_type
        self.n_span = n_span
        self.n_angles = n_angles
        self.n_det = n_det

        self.root_2 = tf.convert_to_tensor(np.sqrt(2.), dtype=self.dtype)

        self.complex_dtype = tf.complex(self.root_2, self.root_2).dtype  # hack

        self.n = None
        self.os_angles = None
        self.batch_size = None
        self.beta = None
        self.a_R = None
        self.a_m = None
        self.g = None
        self.d_rho = None
        self.n_th = None
        self.n_rho = None
        self.d_th = None
        self.angles = None
        self.s = None
        self.th_lp = None
        self.rho_lp = None
        self.b3_com = None
        self.zeta_coeffs = None

    def get_config(self):
        cfg = super(LPRadonBase, self).get_config()
        cfg.update({
            'n_angles': self.n_angles,
            'n_det': self.n_det,
            'n_span': self.n_span,
            'cor': self.cor
        })
        return cfg

    def pre_compute(self):
        self.beta = np.pi / self.n_span

        # expand the image so that we have enough room to rotate
        self.n = int(np.ceil((self.n + abs(self.n / 2 - self.cor) * 2.) / 16.) * 16.)
        self.n_det = self.n_det or self.n

        # oversample, this will change the shape of the image
        self.os_angles = int(max(round(3. * self.n / 2 / self.n_angles), 1))
        self.n_angles = self.os_angles * self.n_angles

        # polar grid
        self.angles = np.arange(0, self.n_angles) * np.pi / self.n_angles - self.beta / 2
        self.s = np.linspace(-1, 1, self.n_det)  # idk why one is np.arange and the other is np.linspace

        self.get_lp_params()

        # log-polar grid
        self.th_lp = np.arange(-self.n_th / 2, self.n_th / 2) * self.d_th
        self.rho_lp = np.arange(-self.n_rho, 0) * self.d_rho

        # compensate for cubic interpolation
        b3_th = splineB3(self.th_lp, 1)
        b3_th = np.fft.fft(np.fft.ifftshift(b3_th))
        b3_rho = splineB3(self.rho_lp, 1)
        b3_rho = np.fft.fft(np.fft.ifftshift(b3_rho))
        self.b3_com = np.outer(b3_rho, b3_th)

    def get_lp_params(self):
        self.a_R = np.sin(self.beta / 2) / (1 + np.sin(self.beta / 2))
        self.a_m = (np.cos(self.beta / 2) - np.sin(self.beta / 2)) / (1 + np.sin(self.beta / 2))

        t = np.linspace(-np.pi / 2, np.pi / 2, 1000)
        w = self.a_R * np.cos(t) + (1 - self.a_R) + 1j * self.a_R * np.sin(t)
        self.g = np.nanmax(np.log(abs(w)) + np.log(np.cos(self.beta / 2 - np.arctan2(w.imag, w.real))))

        self.n_th = self.n_det
        self.n_rho = 2 * self.n_det

        self.d_th = 2 * self.beta / self.n_th
        self.d_rho = (self.g - np.log(self.a_m)) / self.n_rho

    def get_zeta_coeffs(self, a=0, osthlarge=4, adj=False):
        k_rho = np.arange(-self.n_rho / 2, self.n_rho / 2, dtype='float32')
        n_th_large = osthlarge * self.n_th
        th_sp_large = np.arange(-n_th_large / 2, n_th_large / 2) / n_th_large * self.beta * 2
        fZ = np.zeros([self.n_rho, n_th_large], dtype='complex64')
        h = np.ones(n_th_large, dtype='float32')
        # correcting = 1+[-3 4 -1]/24correcting(1) = 2*(correcting(1)-0.5)
        # correcting = 1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0correcting[0]
        # = 2*(correcting[0]-0.5)
        correcting = 1 + np.array([-216254335, 679543284, -1412947389, 2415881496, -3103579086,
                                   2939942400, -2023224114, 984515304, -321455811, 63253516, -5675265]) / 958003200.0
        correcting[0] = 2 * (correcting[0] - 0.5)
        h[0] = h[0] * (correcting[0])

        adj = 1. if adj else -1.

        for j in range(1, len(correcting)):
            h[j] = h[j] * correcting[j]
            h[-1 - j + 1] = h[-1 - j + 1] * (correcting[j])

        for j in range(len(k_rho)):
            fcosa = np.power(np.cos(th_sp_large),
                             (adj * 2 * np.pi * 1j * k_rho[j] / (self.g - np.log(self.a_m)) - 1 - a))
            fZ[j, :] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(h * fcosa)))

        fZ = fZ[:, n_th_large // 2 - self.n_th // 2:n_th_large // 2 + self.n_th // 2]
        fZ = fZ * (th_sp_large[1] - th_sp_large[0])

        # put imag to 0 for the border
        fZ[0] = 0
        fZ[:, 0] = 0
        return fZ


class LPRadonForward(LPRadonBase):
    def __init__(self, n_angles, n_det=None, n_span=3, cor=None, interp_type='cubic', *args, **kwargs):
        super(LPRadonForward, self).__init__(n_angles, n_det, n_span, cor, interp_type, *args, **kwargs)
        self.e_rho = None
        self.lp2c = None
        self.p2lp = None
        # self.lp2c1 = []
        # self.lp2c2 = []  # literally stands for log-polar to cartesian
        self.pids = []  # indices by span
        # self.p2lp1 = []
        # self.p2lp2 = []

        self.reshape_1 = None
        self.reshape_2 = None

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert h == w, "Image must be square"
        self.n = h
        self.batch_size = b

        self.cor = self.cor or self.n // 2

        self.pre_compute()

    def pre_compute(self):
        super().pre_compute()
        # forward projection params
        self.zeta_coeffs = np.fft.fftshift(self.get_zeta_coeffs())

        # log-polar to cartesian
        tmp1 = np.outer(np.exp(np.array(self.rho_lp)), np.cos(np.array(self.th_lp))).flatten()
        tmp2 = np.outer(np.exp(np.array(self.rho_lp)), np.sin(np.array(self.th_lp))).flatten()

        lp2c1 = []
        lp2c2 = []

        for k in range(self.n_span):
            lp2c1.append(((tmp1 - (1 - self.a_R)) * np.cos(k * self.beta + self.beta / 2) -
                          tmp2 * np.sin(k * self.beta + self.beta / 2)) / self.a_R)
            lp2c2.append(((tmp1 - (1 - self.a_R)) * np.sin(k * self.beta + self.beta / 2) +
                          tmp2 * np.cos(k * self.beta + self.beta / 2)) / self.a_R)
            lp2c2[k] *= (-1)
            # cids = np.where((lp2c1[k] ** 2 + lp2c2[k] ** 2) <= 1)[0]
            # lp2c1[k] = lp2c1[k][cids]
            # lp2c2[k] = lp2c2[k][cids]

        s0, th0 = np.meshgrid(self.s, self.angles)
        th0 = th0.flatten()
        s0 = s0.flatten()
        for k in range(self.n_span):
            self.pids.append((th0 >= k * self.beta - self.beta / 2) &
                             (th0 < k * self.beta + self.beta / 2))

        # (self.pids[0].shape)

        p2lp1 = []
        p2lp2 = []

        # polar to log-polar coordinates
        for k in range(self.n_span):
            th00 = th0 - k * self.beta
            s00 = s0
            p2lp1.append(th00)
            p2lp2.append(np.log(s00 * self.a_R + (1 - self.a_R) * np.cos(th00)))
            np.nan_to_num(p2lp2[k], copy=False)

        # transform to unnormalized coordinates for interp
        for k in range(self.n_span):
            lp2c1[k] = (lp2c1[k] + 1) / 2 * (self.n_det - 1)
            lp2c2[k] = (lp2c2[k] + 1) / 2 * (self.n_det - 1)
            p2lp1[k] = (p2lp1[k] - self.th_lp[0]) / (self.th_lp[-1] - self.th_lp[0]) * (self.n_th - 1)
            p2lp2[k] = (p2lp2[k] - self.rho_lp[0]) / (self.rho_lp[-1] - self.rho_lp[0]) * (self.n_rho - 1)

        self.lp2c = tf.stack([lp2c1, lp2c2], axis=-1)
        self.p2lp = tf.stack([p2lp2, p2lp1], axis=-1)

        const = np.sqrt(self.n_det * self.os_angles / self.n_angles) * np.pi / 4 / self.a_R / np.sqrt(2)  # black magic
        self.zeta_coeffs = self.zeta_coeffs[:, :self.n_th // 2 + 1] * const

        if self.interp_type == 'cubic':
            self.zeta_coeffs /= self.b3_com[:, :self.n_th // 2 + 1]

        self.reshape_1 = Reshape((self.n_rho, self.n_th, 1))
        self.reshape_2 = Reshape((self.n_angles, self.n_det, 1))

        # more precomputation stuff
        self.e_rho = tf.reshape(tf.exp(tf.convert_to_tensor(self.rho_lp, dtype=self.dtype)), (1, -1, 1, 1))
        self.zeta_coeffs = tf.convert_to_tensor(self.zeta_coeffs, dtype=self.complex_dtype)
        self.pids = [tf.reshape(tf.convert_to_tensor(self.pids[k], dtype=self.dtype),
                                (1, self.n_angles * self.n_det, 1)) for k in range(len(self.pids))]

        self.lp2c = tf.cast(tf.math.rint(self.lp2c), tf.int32)
        self.p2lp = tf.cast(tf.math.rint(self.p2lp), tf.int32)

    def call(self, inputs, *args, **kwargs):
        b, h, w, c = inputs.shape
        f = tf.image.pad_to_bounding_box(inputs, (self.n_det - h) // 2, (self.n_det - h) // 2, self.n_det, self.n_det)

        out = tf.zeros((1, self.n_angles * self.n_det, 1))
        for k in range(self.n_span):
            # interpolate to log-polar grid
            lp_img = self.reshape_1(interpolate_nearest(f, self.lp2c[k:k + 1]))

            # multiply by e^rho
            lp_img *= self.e_rho

            # fft
            fft_img = tf.signal.rfft2d(tf.squeeze(lp_img, axis=-1))
            fft_img *= self.zeta_coeffs

            # ifft
            lp_sinogram = tf.expand_dims(tf.signal.irfft2d(fft_img), -1)
            p_sinogram = interpolate_nearest(lp_sinogram, self.p2lp[k:k + 1])
            p_sinogram *= self.pids[k]
            out += p_sinogram

        return self.reshape_2(out)


class LPRadonFBP(LPRadonBase):
    def __init__(self, n_angles, n_det=None, n_span=3, cor=None, interp_type='cubic', *args, **kwargs):
        super(LPRadonFBP, self).__init__(n_angles, n_det, n_span, cor, interp_type, *args, **kwargs)

        self.filter = None
        self.cids = None
        self.lpids = None
        self.pids = None
        self.wids = None
        self.c2lp = []
        self.lp2p = []

    def build(self, input_shape):
        b, h, w, c = input_shape
        self.n = w
        self.batch_size = b

        self.cor = self.cor or self.n // 2

        self.pre_compute()

    def pre_compute(self):
        super(LPRadonFBP, self).pre_compute()
        self.zeta_coeffs = np.fft.fftshift(self.get_zeta_coeffs(adj=True))

        # cartesian to log-polar
        x2, x1 = np.meshgrid(np.linspace(-1, 1, self.n_det), np.linspace(-1, 1, self.n_det))
        x1 = x1.flatten()
        x2 = x2.flatten()
        x2 = x2 * (-1)

        self.cids = tf.convert_to_tensor(x1 ** 2 + x2 ** 2 <= 1, dtype=self.dtype)
        c2lp1 = []
        c2lp2 = []
        for k in range(0, self.n_span):
            z1 = self.a_R * (x1 * np.cos(k * self.beta + self.beta / 2) + x2
                             * np.sin(k * self.beta + self.beta / 2)) + (1 - self.a_R)
            z2 = self.a_R * (-x1 * np.sin(k * self.beta + self.beta / 2) +
                             x2 * np.cos(k * self.beta + self.beta / 2))
            c2lp1.append(np.arctan2(z2, z1))
            c2lp2.append(np.log(np.sqrt(z1 * z1 + z2 * z2)))

        # self.c2lp = np.stack([c2lp2, c2lp1])

        # log-polar to polar
        th, rho = np.meshgrid(self.th_lp, np.exp(self.rho_lp))
        th = th.flatten()
        rho = rho.flatten()

        rho_n = rho - (1 - self.a_R) * np.cos(th)
        rho_n = rho_n / self.a_R

        lpids = (th >= - self.beta / 2) & (th < self.beta / 2) & (np.abs(rho_n) <= 1)

        lp2p1 = []
        lp2p2 = []
        for k in range(self.n_span):
            lp2p1.append(th + k * self.beta)
            lp2p2.append(rho_n)

        # something something wrapping (????)
        # right side
        wids = np.log(rho) > self.g
        rho_n = np.exp(np.log(rho) + np.log(self.a_m) - self.g) - (1 - self.a_R) * np.cos(th)
        rho_n = rho_n / self.a_R
        lpidsw = (th >= -self.beta / 2) & (th < self.beta / 2) & (np.abs(rho_n) <= 1)

        # left side
        wids2 = np.log(rho) < np.log(self.a_m) - self.g + (self.rho_lp[1] - self.rho_lp[0])
        rho_n2 = np.exp(np.log(rho) - np.log(self.a_m) + self.g) - (1 - self.a_R) * np.cos(th)
        lpidsw2 = (th >= - self.beta / 2) & (th < self.beta / 2) & (np.abs(rho_n2) <= 1)

        self.wids = (wids2 & lpidsw2) | (wids & lpidsw)

        lp2p1w = []
        lp2p2w = []
        for k in range(self.n_span):
            lp2p1w.append(np.where(lpidsw | lpidsw2, th, 0))
            lp2p2w.append(np.where(lpidsw, rho_n, rho_n2))

        self.pids = []
        for k in range(self.n_span):
            self.pids.append((self.angles >= k * self.beta - self.beta / 2) &
                             (self.angles < k * self.beta + self.beta / 2))

        proj0 = []
        projl = []
        for k in range(self.n_span):
            proj0.append(self.angles[self.pids[k].nonzero()[0][0]])
            projl.append(self.angles[self.pids[k].nonzero()[0][-1]] - proj0[-1])

        # probably has nothing to do with Japan
        projp = (self.n_angles - 1) / (proj0[self.n_span - 1] + projl[self.n_span - 1] - proj0[0])

        for k in range(self.n_span):
            lp2p1[k] = (lp2p1[k] - proj0[k]) / projl[k] * (np.count_nonzero(self.pids[k]) - 1) + (
                    proj0[k] - proj0[0]) * projp
            lp2p2[k] = (lp2p2[k] + 1) / 2 * (self.n_det - 1)
            lp2p1w[k] = (lp2p1[k] - proj0[k]) / projl[k] * (np.count_nonzero(self.pids[k]) - 1) + (
                    proj0[k] - proj0[0]) * projp
            lp2p2w[k] = (lp2p2[k] + 1) / 2 * (self.n_det - 1)

            lp2p = np.stack([lp2p1[k], lp2p2[k]], axis=-1)
            lp2pw = np.stack([lp2p1w[k], lp2p2w[k]], axis=-1)

            c2lp1[k] = (c2lp1[k] - self.th_lp[0]) / (self.th_lp[-1] - self.th_lp[0]) * (self.n_th - 1)
            c2lp2[k] = (c2lp2[k] - self.rho_lp[0]) / (self.rho_lp[-1] - self.rho_lp[0]) * (self.n_rho - 1)

            self.lp2p.append(tf.convert_to_tensor(np.where(np.expand_dims(wids, -1), lp2pw, lp2p),
                                                  dtype=self.dtype)[np.newaxis, ...])
            self.c2lp.append(tf.convert_to_tensor(np.stack([c2lp2[k], c2lp1[k]], axis=-1), dtype=self.dtype))

        self.lpids = tf.convert_to_tensor(lpids | wids, self.dtype)

        const = (self.n_det + 1) * (self.n_det - 1) / self.n_det ** 2 / 2 * np.sqrt(
            self.os_angles * self.n_angles / self.n_det / 2)
        self.zeta_coeffs = tf.convert_to_tensor(self.zeta_coeffs[np.newaxis, :, :self.n_th // 2 + 1] * const,
                                                dtype=self.complex_dtype)

        self.filter = tf.convert_to_tensor(np.fft.fftshift(cosine(self.n_angles))[:self.n_angles // 2 + 1] * np.bartlett(self.n_angles)[:self.n_angles // 2 + 1],
                                           dtype=self.complex_dtype)
        self.filter *= tf.cast(tf.linspace(0., 1., self.n_angles // 2 + 1) < 0.2, dtype=self.complex_dtype)

    def call(self, inputs, *args, **kwargs):
        out = tf.zeros((1, self.n_det * self.n_det, 1))
        test = []
        # out = []

        inputs = tf.transpose(inputs, (0, 2, 1, 3))
        inputs = tf.signal.rfft(tf.squeeze(inputs, axis=-1))
        inputs *= self.filter
        inputs = tf.signal.irfft(inputs)
        inputs = tf.expand_dims(tf.transpose(inputs, (0, 2, 1)), axis=-1)

        for k in range(self.n_span):
            # interp from polar to log-polar
            lp_sinogram = tfa.image.interpolate_bilinear(inputs, self.lp2p[k]) * self.lpids[..., tf.newaxis]
            lp_sinogram = tf.reshape(lp_sinogram, (-1, self.n_rho, self.n_th, 1))
            # lp_sinogram = tf.transpose(lp_sinogram, (0, 2, 1, 3))

            # fft
            fft_img = tf.signal.rfft2d(tf.squeeze(lp_sinogram, axis=-1))
            fft_img *= self.zeta_coeffs

            # ifft
            lp_img = tf.expand_dims(tf.signal.irfft2d(fft_img), -1)

            test.append(lp_img)

            out += tfa.image.interpolate_bilinear(lp_img, self.c2lp[k][tf.newaxis, ...]) * self.cids[..., tf.newaxis]

        return test, tf.reshape(out, (-1, self.n_det, self.n_det, 1))


def splineB3(x2, r):
    sizex = len(x2)
    x2 = x2 - (x2[-1] + x2[0]) / 2
    stepx = x2[1] - x2[0]
    ri = int(np.ceil(2 * r))
    r = r * stepx
    x2c = x2[int(np.ceil((sizex + 1) / 2.0)) - 1]
    x = x2[int(np.ceil((sizex + 1) / 2.0) - ri - 1):int(np.ceil((sizex + 1) / 2.0) + ri)]
    d = np.abs(x - x2c) / r
    B3 = x * 0
    for ix in range(-ri, ri + 1):
        id_ = ix + ri
        if d[id_] < 1:  # use the first polynomial
            B3[id_] = (3 * d[id_] ** 3 - 6 * d[id_] ** 2 + 4) / 6
        else:
            if d[id_] < 2:
                B3[id_] = (-d[id_] ** 3 + 6 * d[id_] ** 2 - 12 * d[id_] + 8) / 6

    B3f = x2 * 0
    B3f[int(np.ceil((sizex + 1) / 2.0) - ri - 1):int(np.ceil((sizex + 1) / 2.0) + ri)] = B3
    return B3f


@tf.function
def interpolate_nearest(f, points):
    f = tf.transpose(f, (1, 2, 3, 0))
    return tf.transpose(tf.gather_nd(f, points), (3, 0, 1, 2))


def main():
    import matplotlib.pyplot as plt
    # from perlin_noise import generate_fractal_noise_2d

    # img = generate_fractal_noise_2d(64, [512, 512], [2, 2], octaves=2)
    # img = tf.expand_dims(img, axis=-1)
    import h5py
    import time
    with h5py.File("../data/ground_truth_test/ground_truth_test_000.hdf5") as f:
        img = f['data'][:64]
        img = img[:, :, :, tf.newaxis]

    forward = LPRadonForward(1024, 512, n_span=3)
    back = LPRadonFBP(1024, 512, n_span=3)

    start_time = time.time()
    sinogram = forward(img)
    steps, fbp = back(sinogram)
    print(time.time() - start_time)

    plt.hist(img.flatten(), bins=100)
    plt.show()
    fbp = tf.image.central_crop(fbp, 362 / 512)
    print(tf.reduce_mean(fbp), tf.math.reduce_std(fbp))
    plt.hist(fbp.numpy().flatten(), bins=100)
    plt.show()
    plt.imshow(fbp[0], cmap='gray')
    plt.show()

    # for i in steps:
    #     plt.imshow(i[0], cmap='gray')
    #     plt.show()

    plt.plot(np.abs(back.filter))
    plt.show()

    # plt.imshow(img[0], cmap='gray')
    # plt.show()
    #
    # plt.imshow(sinogram[0], cmap='gray')
    # plt.show()
    #
    #
    # plt.imshow(np.reshape(test.pids[2], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(np.reshape(test.pids[0], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(np.reshape(test.pids[1], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(np.reshape(test.pids[2], (test.n_angles, test.n_det)), cmap='gray')
    # plt.show()
    # plt.imshow(test(img)[0].numpy(), cmap='gray')
    # plt.show()
    # plt.imshow(test.zeta_coeffs.real, cmap='gray')
    # plt.show()
    #
    # plt.scatter(x=test.lp2c1[0], y=test.lp2c2[0], s=0.01)
    # plt.gca().set_aspect('equal')
    # plt.show()
    #
    # plt.scatter(x=test.lp2c1[1], y=test.lp2c2[1], s=0.01)
    # plt.gca().set_aspect('equal')
    # plt.show()
    #
    # plt.scatter(x=test.lp2c1[2], y=test.lp2c2[2], s=0.01)
    # plt.gca().set_aspect('equal')
    # plt.show()
    #
    # plt.imshow(test.pids, aspect='auto', interpolation='nearest')
    # plt.show()


if __name__ == '__main__':
    main()
