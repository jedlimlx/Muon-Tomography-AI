## Muons Multi-metal Dataset

| Blocks          | Filters              | Activation | Kernel Size | Droppath | Dropout | Block Size | Noise | Block    | Attention | Optimizer        | Weight Decay | LR    | Loss | MSE    | MAE    |
|-----------------|----------------------|------------|-------------|----------|---------|------------|-------|----------|-----------|------------------|--------------|-------|------|--------|--------|
| (1, 2, 2, 3, 4) | (64, 64, 64, 64, 64) | Swish      | 7           | 0.05     | 0.05    | 10         | 0.20  | ConvNeXt | None      | Lookahead, RAdam | 0            | 0.002 | MSE  |        |        |
