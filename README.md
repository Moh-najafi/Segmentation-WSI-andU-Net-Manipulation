# Segmentation-WSI-andU-Net-Manipulation
This is the private project assigned by DiaDeep
# Objectives

The main goal here is to prepare WSIs data for a segmentation task. We have at our disposal original WSIs together with annotations (region of interest). The following are the steps you are supposed to provide: 

1. Cropping the WSI to get only the “tissue regions”. In fact, a major part of the WSIs is composed of background and other artefacts, see the image below:

![Screenshot 2023-03-29 at 14.49.25.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aadac7b5-9b35-4330-a709-64ca89a47ab1/Screenshot_2023-03-29_at_14.49.25.png)

In this example, you should save the 6 cropped tissue samples (left) to a local folder as `filename_i.png` at a **downsample scale of 32** (the number of samples in each image is `i>=1`).

1. The annotations provided are generally used to train a semantic segmentation model, e.g. [U-Net](https://fr.wikipedia.org/wiki/U-Net). On a single tissue sample, you should **provide the “mask”** needed as input for this segmentation network.
2. This mask should fit into the following model, which should be **completed and functional** (**ready to train on a large scale,** if sufficient data are available)

```python
import segmentation_models_pytorch as smp

net = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=,                      # model output channels (number of classes in your dataset)
)
```

An **example** of `net` applied to the example of this WSI should be provided together with some data augmentation (horizontal and vertical flip, rotation, etc.).
