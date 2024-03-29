# Segmentation-WSI-andU-Net-Manipulation


# Objectives

H I 😀

>This script is provided for **part 1** & **part 2** and **tile and Augmentation Preprocessing**.

>
The description of project is provided as [Segmentation: WSI and U-Net Pipeline Manipulation](https://informationsharing.notion.siteSegmentation-WSI-and-U-Net-Pipeline-Manipulation-82a91afd8c24478f8be89c61bd04ba85).

> In summary, the mask of WSI image is provided in 'annotation.csv' with a specific format. The steps are as follows:
- Extracting ROI info from csv file, 
- Preprocessing the data to reach the desired mask win downsample version (using pyvips) (reaching to the state in which you can select tissues of you want to have their specific mask)
- Tile preprocessing to provide image patches for train validation and augmentation of data(using pyvips)
- Conducting train and validation, seeing some random predictions and applying error analysis to improve 
  


> Having gained a better understanding of its functionality, I believe it would be applicable to focus on a specific class of annotation terms or a selected tissue for further refinement. Additionally, to achieve superior results, I've prepared a pilot format for fine-tuning SMP models using predefined weights. This can be applied to an augmented dataset generated by me from tiles of the original image, especially when ample image data is available and can be processed on a robust server.





The first part of the project is to prepare the data with its label mask which results from geometry data that results from 'annotations.csv' file and the main image 'm9de8lfp.tif'. I have done it by extracting geometry data from the original WSI image and using 'pylibs' and 'shapely' libraries.
the mask is obtained as follows:

![extracted mask](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/83cab7a4-c2d8-4656-af99-2932ad1dff1d)

For a better representation of the extracted mask from 'annotation.csv' file, you can see the next image which is in smaller shape:

![m9de8lfp_MASK_new_64](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/3a0568b0-d24c-47f7-a804-d31da6a69bd7)


The next step is preparing the mask data for image segmentation. To reach the mask as ground truth with need to have a binary mask. In addition for segmentation, we can have 2 approaches. the first one could be related to annotation terms and we can use the original mask, the second one is for tissue segmentation which is mentioned in part 2.
in the following, you can see the binary masks, registered masks, and each tissue mask separately which is prepared for further processing if asked.

![filled morphed mask](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/253439e2-3b1f-45a1-97ad-57af9ffe890f)
![filled morphed registered](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/92d6e580-676d-43c0-81af-466f87e41567)

![labeling different region](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/5361a534-81cf-4071-8d88-dffacabb963b)
![each tissue registered](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/2368114d-46f3-4211-aa13-aa73e84b30ba)


For part 3, tiles action is applied to have patches for training and validation: you can find some random tiles with their masks in the next plot of tiles:
![random tiles](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/27d9aa9e-acf6-419f-a089-14178f702226)

The last step is applying segmentation using smp by pytorch. As my computational source and data are limited, (epochs = 15 and various augmentation method are scripted as comments and is tested to be applicable if needed) I have just designed a prototype version and you can see the result of applying a pre-trained model with 'imagenet' weights in the next images. The validation loss plot is also obtained. ( I have tried smp model without applying pretrained weights and results are not well enough in low values of epochs). 

![val_loss of pretrained model](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/59aea6a2-abc4-42cb-ae77-989d99424b68)

![pretrained model result5](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/d4d54e53-2b54-434b-a494-c34adfcc2923)
![pretrained model result2](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/cbd271f1-d93a-4a67-a09d-23f633a13c49)
![pretrained model result4](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/e0970604-1f3f-4d16-b508-d96e98c9ecfc)

![pretrained model result3](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/e5297a98-9083-4132-97f6-b88dd0c9f91b)


Lastly, I want to mention that consider this as a prototype and it could be modified with the objective and data and suited computational source to test.
You can also download the prototype version of the fine-tuned pre-trained model with 15 epochs with the following link:
https://drive.google.com/file/d/1mlq5dmFtXn0TiE-1S-_iE6aD9xQ8O8fX/view?usp=sharing

Updates and comments:
I augmented data in 3 ways as there would be a higher cost in terms of time and resources if I want to apply various augmentation methods with strides that produce more images and there can be an improvement in image segmentation (binary classification) of tissues ROI, and in the following I am providing more images. However, as the data diversity is not high enough, we may have some defects in the cost function of the validation set. This issue can be solved by applying more diverse WSI for training and validation, and also by applying more augmented data which needs more computational sources for conducting trial and error on training and validation in order to optimize. (considering dropout if needed)

The updated version of fine-tuned pre-trained in 30 epochs has the cost function as follows which is not stable in some epochs that might have root in not enjoying diverse data both in training and testing and might have been overfitted somehow in training set.
![image](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/3bc5da16-b7b8-43e9-905a-b87e797f9b77)

some result images of this updated algorithm from test data:
![image](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/b6fff04d-8260-47e4-96d7-8c256bcc224c)
![image](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/f4e6073a-8ef9-44bc-8292-82fd79568618)
![image](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/d8531480-ea0e-429f-976f-2b403cca04ca)

and there is another update: (the picks are in cost function might be solved by more diversified data for training and validating.
tile_size = 512
stride = 256
171 images for train:
45 for validation:
Total tiles: 216 
augmentations = [HorizontalFlip(p=1), ElasticTransform(p=1)]

![image](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/53718643-943d-48a5-bf07-c9243af91a8d)


![6](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/83d5aef6-3eb0-445d-ba4d-ce1700ea0006)
![6](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/c1136b29-92c8-4740-9658-3940e7edb956)
![6_2](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/84872461-e16d-42c2-bf5f-4c95faa95af8)![6_4](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/1a870251-3c4d-471d-a02e-c303bf077627)
![7](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/2f9dd993-60d8-432e-9ec1-a943821f59f6)
![9](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/3085aa79-134d-43dc-99bb-34dcd3dc6154)





> For the next step, in my opinion, designing a pipeline to apply segmentation for tissue ROI is the first step. Having reached the desired algorithm which can be reliable enough for this objective, the next is the segmentation via annotation terms that have been provided in 'annotation.csv' file to segment each selected tissue into 5 classes as follows:
    'Dermal component of melanoma'
    'Intra-epidermal component of melanoma'
    'Normal dermis'
    'Normal sub-cutaneous tissue'
    'Normal epidermis (with papillary dermis)'


 

Regards,
M Najafi


