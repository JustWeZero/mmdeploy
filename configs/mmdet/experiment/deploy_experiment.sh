export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn

# VisDrone模型部署
# 换docker后转换成功
export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
    ../mmdetection/configs2/VisDrone/lka_fpn/faster_rcnn_18_lka_fpn_1x_VisDrone640_singletest.py \
    ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    ../data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --test-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00611_d_0000002.jpg \
    --work-dir ../VisDrone_cache/deployment/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g \
    --calib-dataset-cfg ../mmdetection/configs2/VisDrone/dataset/VisDrone_detection_640x640_overlap40_singletest.py \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info

# 重新转换一下模型
export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
    ../mmdetection/configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
    ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth \
    ../data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg \
    --test-img data/VisDrone2019/VisDrone2019-DET-test-dev/images/0000006_00611_d_0000002.jpg \
    --work-dir ../VisDrone_cache/deployment/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g \
    --calib-dataset-cfg ../mmdetection/configs2/VisDrone/dataset/VisDrone_detection_640x640_overlap40.py \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info

# 显示没有检测结果，平均FPS 45.74。去掉了最下面的参数后仍然没有检测结果，回mmdetection排除一下
# 从mmrazor中训练出来的结果，和分离模型后调用mmdetection的结果不一致（0.2130，0.1760）
export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640" \
&& METRICS=bbox && SHOW_SCORE_THR=0.7 && work_dir=../VisDrone_cache/deployment/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g
python tools/test.py \
configs/mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py \
../mmdetection/configs2/VisDrone/lka_fpn/faster_rcnn_r18_lka_fpn_1x_VisDrone640.py \
--model ${work_dir}/end2end.engine \
--out ${work_dir}/output.pkl \
--metrics ${METRICS} \
--show-dir ${work_dir}/test_result \
--device cuda:0 \
--log2file ${work_dir}/output.txt \
--speed-test \
--cfg-options optimizer.lr=${LR} data.samples_per_gpu=16

