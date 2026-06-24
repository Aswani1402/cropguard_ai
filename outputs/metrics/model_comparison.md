# Model Comparison

| model               | role                                  |   accuracy |   macro_f1 |   weighted_f1 |   top_3_accuracy |   loss |   train_images |   valid_images |   num_classes |
|:--------------------|:--------------------------------------|-----------:|-----------:|--------------:|-----------------:|-------:|---------------:|---------------:|--------------:|
| EfficientNetV2B0    | Main transfer learning model          |     0.9796 |     0.9794 |        0.9795 |           0.9987 | 0.0675 |          70295 |          17572 |            38 |
| MobileNetV3Large    | Lightweight deployment-friendly model |     0.9792 |     0.979  |        0.9792 |           0.9986 | 0.0618 |          70295 |          17572 |            38 |
| Custom CNN Baseline | Baseline model trained from scratch   |     0.9614 |     0.9624 |        0.9619 |           0.9976 | 0.1153 |          70295 |          17572 |            38 |
