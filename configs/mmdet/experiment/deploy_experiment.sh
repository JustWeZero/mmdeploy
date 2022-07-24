export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn

# VisDrone模型部署
export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
    ../mmdetection/configs2/VisDrone/lka_fpn/faster_rcnn_18_lka_fpn_1x_VisDrone640_singletest.py \
    ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    ../data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --test-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00611_d_0000002.jpg \
    --work-dir ../VisDrone_cache/deployment/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g \
    --calib-dataset-cfg ../configs2/VisDrone/dataset/VisDrone_detection_640x640_overlap40_singletest.py \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info

# 标准模型部署
# 目前还有报错
python tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
    ../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ../mmdetection/demo/demo.jpg \
    --test-img ../mmdetection/demo/demo.jpg \
    --work-dir ../mmdetection/demo/demo_deploy \
    --calib-dataset-cfg ../mmdetection/configs/_base_/datasets/coco_detection.py \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info