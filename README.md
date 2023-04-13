##### The experiment steps and the instruction of running the code are as follows:
## 1.Clean the dat

Run the python file “check_data.py”
The python file reads each sample in the LB folder, set label 2 to 255 (background), and then save the data to the new_LB folder.

## 2.Train SegNet

The name of the training code is "train_segnet_2label_seg_fc_hrnet.py". 

The trained weights are saved in "trained_weights/seg_fc_hrnet_avgpooling-model_50_only_1.h5". 

The training record is saved in "training_logs/TrainHistory-Seg_fc_hrnet-1_label-50_epoch.data". 

To visualize the training results, use "display_results/read_history_data_seg_fc_hrnet.py". The corresponding images can be found in "display_results/result_loss_seg_fc_hrnet.png" and "display_results/result_acc_seg_fc_hrnet.png".


## 3.TrainUNet (VGG backbone)

The VGG part is located in "qt_model/Model/my_vgg.py", and the UNet part is located in "qt_model/Model/my_unet.py". 

The training code is "unet_vgg_train.py". Except for the model, all training-related parameters are the same as before. 

The trained weights are saved in "trained_weights/seg_vgg_unet-model_50_only_1.h5". 

The training data is saved in "training_logs/TrainHistory-Seg_vgg_unet-1_label-50_epoch.data". 

To visualize the training results, use "display_results/read_history_data_seg_vgg_unet.py". The visualization results are "display_results/result_loss_unet_vgg.png" and "display_results/result_acc_unet_vgg.png".


## 4.UNet (ResNet50 Backbone)

The ResNet part is located in "qt_model/Model/my_resnet.py", and the UNet part is located in "qt_model/Model/my_unet.py". 

The training code is "unet_resnet_train.py". Except for the model, all training-related parameters are the same as before. 

The trained weights are saved in "trained_weights/seg_resnet50_unet-model_50_only_1.h5". 

The training data during the training process is saved in "training_logs/TrainHistory-Seg_resnet50_unet-1_label-50_epoch.data".



## 5. Performance Comparison

To compare the performance of different models, the code is located in "display_results/data_compare.py". 

The results are saved in "display_results/training loss compare.png" and "display_results/training accuracy compare.png".

"check_data.py" is used to display the performance indicators of different models.

The link of the pretrained model is: https://pan.baidu.com/s/1N6Y75PowF45kc24ivPIpMw?pwd=8bud


