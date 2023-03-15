import random
from functools import reduce
from layers.vision_transformer import CTransformer


def create_model(
    phi,
    k_enc_layers=2,
    k_dec_layers=2,
    k_enc_dim=128,
    k_enc_mlp=256,
    k_dec_dim=128,
    k_dec_mlp=256,
    k_enc_heads=8,
    k_dec_heads=8,
):
    # main model
    model = CTransformer(
        enc_layers=int(2 * k_enc_layers ** phi),
        dec_layers=int(2 * k_dec_layers ** phi),
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=int(128 * k_enc_dim ** phi),
        enc_mlp_units=int(256 * k_enc_mlp ** phi),
        dec_dim=int(128 * k_dec_dim ** phi),
        dec_mlp_units=int(256 * k_dec_mlp ** phi),
        enc_heads=int(8 * k_enc_heads ** phi),
        dec_heads=int(8 * k_dec_heads ** phi),
        output_patch_width=16,
        output_patch_height=16,
        output_x_patches=32,
        output_y_patches=32,
        output_projection=True
    )

    model.compile(
        optimizer="adam", loss="mse"  # , metrics=(_psnr(), _ssim())
    )

    return model


if __name__ == "__main__":
    while True:
        parameters = [random.random() / 3 + 1 for x in range(8)]
        divisor = (reduce(lambda x, y: x * y, parameters, 1) / 2) ** (1/8)
        parameters = [x / divisor for x in parameters]

        for i in range(8):
            if parameters[i] < 1: break
        else:
            break

    print(parameters)

    model = create_model(
        1,
        k_enc_layers=1,
        k_dec_layers=1,
        k_enc_dim=1,
        k_enc_mlp=1,
        k_dec_dim=1,
        k_dec_mlp=1,
        k_enc_heads=1,
        k_dec_heads=1,
    )
