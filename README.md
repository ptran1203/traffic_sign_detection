# traffic_sign_detection

Solution for Zalo AI traffic signs detection


## Method
Tiling Image into parts with some overlaps and predict on each part + full size image. Then combine the result using non-max suppression.

### 1. Image tiling
![tiling](./images/tiling.png)

### 2. Predict on each part
![tiling](./images/tiling_pred_small.png)

### 3. Predict on the whole image
![tiling](./images/tiling_pred_big.png)

### 3. Combine the results
![tiling](./images/tiling_pred.png)


## Trainning

```bash
python3 train.py --input [path to training images]
```

## Inference

```bash
./predict.sh
or
python3 prediction.py --input [path to test images] --output [path to submission file]
```

## Inference result

![./images/pred_1.png](./images/pred_1.png)
![./images/pred_2.png](./images/pred_2.png)
![./images/pred_3.png](./images/pred_3.png)
