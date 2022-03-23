### Current experiement

#### Binary organ segmentation
1. UNet
2. AttUnet

#### Multiorgan segmentation
1. UNet - Dice_coeff
2. UNet ++ - Dice_coeff
3. AttUNet - Dice
4. ResAttUNet - Dice
5. DeepLabV3+ - (encoders : restnet18, 34,152)
>> Note: I perform two times the resnet18 the one is using a pretrained model
6. ResAttUnet - Focal Loss
7. Integration of k-fold cross validation
8. K-fold (k=5) on ResAttUnet - Double Focal Loss (Name= 22_3_ResAttU_Net_200_4_fold_{the number of the fold})

