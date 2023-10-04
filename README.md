# Segmentation-WSI-andU-Net-Manipulation
This is the private project assigned by DiaDeep

# Objectives

H I 😀

>This script is provided for **task 1** & **task 2** and **tile and Augmentation Preprocessing**.
>This script is privately provided by Mohamad NAJAFI for the DiaDeep PhD applicant task. The task consists of three parts, and all the necessary scripts have been provided to achieve the requested results.
>
The description of task is provided as [Segmentation: WSI and U-Net Pipeline Manipulation](https://informationsharing.notion.siteSegmentation-WSI-and-U-Net-Pipeline-Manipulation-82a91afd8c24478f8be89c61bd04ba85).

> It should be mentioned that the main challenges and important points are as follows for me:

- The 'annotation.csv' file cannot be read as a dataframe because all the data is stored in the first column of the CSV file. Additionally, there is an issue with reading the LINESTRING format of the geometry data. This requires parsing the data to obtain WKT or OPENCV geometry data, which needs to be addressed.
- I have experience in TensorFlow and Keras for DML and DL applications, however, I tried to not use tf libraries and just work with **torch** in this project.


> I am available and open to optimizing this script for various WSI image segmentation purposes. Having gained a better understanding of its functionality, I believe it would be applicable to focus on a specific class of annotation terms or a selected tissue for further refinement. Additionally, to achieve superior results, I've prepared a pilot format for fine-tuning SMP models using predefined weights. This can be applied to an augmented dataset generated by me from tiles of the original image, especially when ample image data is available and can be processed on a robust server.


> I am available to offer explanations or make modifications to this script based on your specific inquiry.

> Mohamad Najafi


The first part of the task is to prepare the data with its label mask which results from geometry data that results from 'annotations.csv' file and the main image 'm9de8lfp.tif'. I have done it by extracting geometry data from the original WSI image and using 'pylibs' and 'shapely' libraries.
the mask is obtained as follows:

![extracted mask](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/83cab7a4-c2d8-4656-af99-2932ad1dff1d)


The next step is preparing the mask data for image segmentation. To reach the mask as ground truth with need to have a binary mask. In addition for segmentation, we can have 2 approaches. the first one could be related to annotation terms and we can use the original mask, the second one is for tissue segmentation which is mentioned in task 2.
in the following, you can see the binary masks, registered masks, and each tissue mask separately which is prepared for further processing if asked.

![filled morphed mask](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/253439e2-3b1f-45a1-97ad-57af9ffe890f)
![filled morphed registered](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/92d6e580-676d-43c0-81af-466f87e41567)

![labeling different region](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/5361a534-81cf-4071-8d88-dffacabb963b)
![each tissue registered](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/2368114d-46f3-4211-aa13-aa73e84b30ba)


For task 3, tiles action is applied to have patches for training and validation: you can find some random tiles with their masks in the next plot of tiles:
![random tiles](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/27d9aa9e-acf6-419f-a089-14178f702226)

The last step is applying segmentation using smp using pytorch. As my computational source and data are limited, (epochs = 15 and various augmentation method are scripted as comments and is tested to be applicable if needed) I have just designed a prototype version and you can see the result of applying a pre-trained model with 'imagenet' weights in the next images. The validation loss plot is also obtained. ( I have tried smp model without applying pretrained weights and results are not well enough in low values of epochs). 

![val_loss of pretrained model](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/59aea6a2-abc4-42cb-ae77-989d99424b68)

![pretrained model result5](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/d4d54e53-2b54-434b-a494-c34adfcc2923)
![pretrained model result2](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/cbd271f1-d93a-4a67-a09d-23f633a13c49)
![pretrained model result4](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/e0970604-1f3f-4d16-b508-d96e98c9ecfc)

![pretrained model result3](https://github.com/Moh-najafi/Segmentation-WSI-andU-Net-Manipulation/assets/93668623/e5297a98-9083-4132-97f6-b88dd0c9f91b)


Lastly, I want to mention that consider this as a prototype and it could be modified with the objective and data and suited computational source to test.

Regards,
M Najafi


