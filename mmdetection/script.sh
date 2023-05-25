# # python train_confusion.py \
# # --config_path '/opt/ml/baseline/mmdetection/configs/_custom_/convnext/cascade_mask_rcnn_convnext_xlarge_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in22k_detr.py' \
# # --valid_num 1 \
# # --seed 2022 \
# # --batch_size 4 \
# # --lr 0.000005 \
# # --epoch 36 \
# # --name Cascade_RCNN+ConvNext_xlarge

# python train_confusion.py \
# --config_path '/opt/ml/baseline/mmdetection/configs/_custom_/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco.py' \
# --valid_num 1 \
# --seed 2022 \
# --batch_size 4 \
# --lr 0.000005 \
# --epoch 12 \
# --name ATSS+Swin-l

output1=$(python print_key.py)
python tools/test.py $output1

output2=$(python ./print_confusion.py)
python tools/analysis_tools/confusion_matrix.py $output2

