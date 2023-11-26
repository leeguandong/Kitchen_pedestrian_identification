# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('--video', default='/home/ivms/local_disk/Kitchen_pedestrian_identification/data/202302071541_11.mp4',help='Video file')
    parser.add_argument('--config', default='/home/ivms/local_disk/Kitchen_pedestrian_identification/yolox_s_8x8_300e_coco.py',help='Config file')
    parser.add_argument('--checkpoint', default='/home/ivms/local_disk/Kitchen_pedestrian_identification/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='Bbox score threshold')
    parser.add_argument('--out', default='/home/ivms/local_disk/Kitchen_pedestrian_identification/results/out.mp4',type=str, help='Output video file')
    parser.add_argument('--show', default=False,action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)
#     capture = cv2.VideoCapture(args.video)
    video_reader = mmcv.VideoReader(args.video)
    
    video_writer = None  
#     width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(capture.get(cv2.CAP_PROP_FPS))
#     frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     print("video fps: %d, frame_count: %d" % (fps, frame_count))

    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video_writer = cv2.VideoWriter(
#             args.out, fourcc, fps, (width, height))
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    
#     if capture.isOpened():
#         print("####")

#     while (1):
#         ret, frame = capture.read()
#         if not ret:
#             break
    for frame in mmcv.track_iter_progress(video_reader):
#         import pdb;pdb.set_trace()
        result = inference_detector(model, frame)
        
#         if isinstance(result, tuple):
#             bbox_result, segm_result = result
#             if isinstance(segm_result, tuple):
#                 segm_result = segm_result[0]  # ms rcnn
#         else:
#             bbox_result, segm_result = result, None
#         bboxes = np.vstack(bbox_result)
#         labels = [
#             np.full(bbox.shape[0], i, dtype=np.int32)
#             for i, bbox in enumerate(bbox_result)
#         ]
#         labels = np.concatenate(labels)
        
        result = [result[0]]
            
#         import pdb;pdb.set_trace()
            
        frame = model.show_result(frame, result, score_thr=args.score_thr)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
