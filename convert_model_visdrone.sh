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

docker run -it --rm -w /root/workspace/mmdeploy \
    -v /root/software:/software --network host 172.18.18.222:5000/schinper/ai-train:schiper_deploy_xavier_v1.0  python3 tools/deploy.py /software/model/cls_139_1182/model_deploy_config.py \
    /software/model/cls_139_1182/mmcls_model_config.py /software/model/cls_139_1182/latest.pth /software/model/cls_139_1182/demo.jpg \
    --test-img /software/model/cls_139_1182/demo.jpg \
    --work-dir /software/model/cls_139_1182/convert \
    --device cuda:0 --log-level INFO --dump-info

docker run --gpus all \
 -v /home/group5/lzj/mmdetection:/root/workspace/mmdetection \
 -v /home/group5/lzj/data:/root/workspace/data \
 -v /home/group5/lzj/TOV_mmdetection_cache:/root/workspace/TOV_mmdetection_cache \
 -v /home/group5/lzj/VisDrone_cache:/root/workspace/VisDrone_cache \
 -tid mmdeploy:inside
