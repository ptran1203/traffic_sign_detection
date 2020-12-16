# traffic_sign_detection

Solution for Zalo AI traffic signs detection.

Start with [Colab](https://github.com/ptran1203/traffic_sign_detection/blob/main/traffic_signs_detection.ipynb)


## Method
After taking a quick look at the dataset, we can see that there are many small signs. Which makes the model hard to detect them. One of the possible methods in this scenario is [Image Tiling](https://openaccess.thecvf.com/content_CVPRW_2019/papers/UAVision/Unel_The_Power_of_Tiling_for_Small_Object_Detection_CVPRW_2019_paper.pdf)

In detail, I will crop the image into a few parts with some overlaps, predict each part and the original image. Then combine the result using non-max suppression.

### 1. Image cropping
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

---
**NOTE**

This project is running on Google Colaboratory environment, so it may contain some issues when running on local machine, please don't hesitate to create a new issue

---
