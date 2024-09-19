# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

import csv
import cv2
from mmengine import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=1,
        type=int,
        help='specify fps value of the output video when using rawframes to '
             'generate file')
    parser.add_argument(
        '--font-scale',
        default=None,
        type=float,
        help='font scale of the text in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the text in output video')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
             'video as input. If either dimension is set to -1, the frames are '
             'resized by keeping the existing aspect ratio')
    parser.add_argument('--csv-filename', default='output.csv', help='CSV filename to save results')
    args = parser.parse_args()
    return args


def split_video(video_path, window_size, overlap, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunks = []

    start_frame = 0
    chunk_idx = 0
    while start_frame < total_frames:
        end_frame = min(start_frame + window_size * fps, total_frames)
        chunk_filename = osp.join(output_dir, f"chunk_{chunk_idx}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(chunk_filename, fourcc, fps, (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(int(start_frame), int(end_frame) ):
            success, frame = cap.read()
            if not success:
                break
            out.write(frame)

        out.release()
        chunks.append((chunk_filename, start_frame / fps, end_frame / fps))
        start_frame += (window_size - overlap) * fps
        chunk_idx += 1

    cap.release()
    return chunks

def post_process_predictions(args, chunk_results, overlap):
    with open(args.csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Second', 'Label', 'Score'])
        
        num_chunks = len(chunk_results)
        
        for i in range(num_chunks):
            chunk_filename, start_time, end_time, result, score = chunk_results[i]
            
            # Check the previous and next chunks
            if i > 0 and i < num_chunks - 1:
                _, _, _, prev_result, _ = chunk_results[i - 1]
                _, _, _, next_result, _ = chunk_results[i + 1]
                
                # If the current results are different from both previous and next
                # but previous and next are the same, assign current to previous/next
                if (result != prev_result and result != next_result and 
                    prev_result == next_result):
                    result = prev_result
            if i != num_chunks -1:
                # Write a row for each second in the chunk
                for sec in range(int(start_time), int(end_time-overlap)):
                    writer.writerow([sec, result, score])
            else:
                # Write a row for each second in the chunk
                for sec in range(int(start_time), int(end_time)):
                    writer.writerow([sec, result, score])

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)

    # Define a list of window and overlap sizes to experiment with
    window_sizes = [150]  # in seconds
    overlap_percentages = [15]  # in percentage
    
    chunk_results = []

    for window_size in window_sizes:
        for overlap_percentage in overlap_percentages:
            overlap = window_size * (overlap_percentage / 100)  # calculate overlap in seconds
            output_dir = f'video_chunks_window{window_size}_overlap{overlap_percentage}'
            os.makedirs(output_dir, exist_ok=True)
            chunks = split_video(args.video, window_size, overlap, output_dir)

            labels = open(args.label).readlines()
            labels = [x.strip() for x in labels]

            for chunk_filename, start_time, end_time in chunks:
                pred_result = inference_recognizer(model, chunk_filename)

                pred_scores = pred_result.pred_score.tolist()
                score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
                score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
                top5_label = score_sorted[:5]

                results = [(labels[k[0] - 1], k[1]) for k in top5_label]
                print('The top-5 labels with corresponding scores are:')
                for result in results:
                    print(f'{result[0]}: {result[1]}')
                    
                chunk_results.append((chunk_filename, start_time, end_time, results[0][0], results[0][1]))
                
                post_process_predictions(args, chunk_results, overlap)

                   
if __name__ == '__main__':
    main()
