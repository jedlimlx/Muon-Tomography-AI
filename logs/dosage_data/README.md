- Material: Gold
- Max Dosage: 46000 - 45000 muons
- Beam: 1D
- Parameters:
```python
{
    "task": "sparse",
    "shape": x_train[0].shape,
    "blocks": (1, 2, 2, 3, 4),
    "filters": (64, 64, 64, 64, 64),
    "activation": "swish",
    "drop_connect_rate": 0.05,
    "dropout_rate": 0.05,
    "block_size": 10,
    "noise": 0.20,
    "dropblock_2d": True,
    "dropblock_3d": False,
    "block_type": "convnext",
    "attention": "se",
    "dimensions": 3,
    "initial_dimensions": 2,
    "final_activation": "sigmoid"
}
```