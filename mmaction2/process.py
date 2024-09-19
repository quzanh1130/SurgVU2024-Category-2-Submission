import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List
import json
from operator import itemgetter
from mmengine import Config
from typing import Optional, Tuple
from mmaction.apis import inference_recognizer, init_recognizer
import os
import os.path as osp

execute_in_docker = True  # Change to True if running in Docker

class SurgVUClassify:
    def __init__(self):
        self.index_key = 'input_video'
        self.input_path = Path("/input/") if execute_in_docker else Path("./test/")
        self.output_file = Path("/output/surgical-step-classification.json") if execute_in_docker else Path("./output/surgical-step-classification.json")
        
        # Slowfast config and checkpoint
        config_path = 'configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_rgb_1fps_video_final.py'
        checkpoint_path = 'work_dirs/slow_fast/best_model.pth'
        
        self.window_size = 60  # seconds
        self.overlap_percentage = 15  # percent
        
        self.device = 'cuda:0'
        
        self.cfg = Config.fromfile(config_path)
        self.model = init_recognizer(self.cfg, checkpoint_path, device=self.device)
    
    def split_video(self, video_path: str, window_size: int, overlap: int, output_dir: str) -> List[Tuple[str, float, float]]:
        """
        Splits a video into smaller chunks based on the specified window size and overlap.

        Args:
            video_path (str): Path to the input video file.
            window_size (int): Size of each video chunk in seconds.
            overlap (int): Overlap between consecutive chunks in seconds.
            output_dir (str): Directory where the video chunks will be saved.

        Returns:
            List[Tuple[str, float, float]]: A list of tuples where each tuple contains:
            - The filename of the video chunk.
            - The start time of the chunk in seconds.
            - The end time of the chunk in seconds.
        """
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
            for _ in range(int(start_frame), int(end_frame)):
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

    
    def post_process_predictions(self, chunk_results, overlap):
        """
        Post-processes the predictions by applying a smoothing technique and saves the results to a JSON file.

        Args:
            chunk_results (list): A list of tuples where each tuple contains:
            - chunk_filename (str): The filename of the chunk.
            - start_time (float): The start time of the chunk.
            - end_time (float): The end time of the chunk.
            - result (str): The predicted result for the chunk.
            - score (float): The confidence score of the prediction.
            overlap (int): The number of overlapping frames between consecutive chunks.

        Returns:
            None
        """
        num_chunks = len(chunk_results)
        frame_predictions = []

        # Store all the results for easier access
        all_results = [chunk[3] for chunk in chunk_results]

        # Define the smoothing window size
        smoothing_window = 8

        for i in range(len(chunk_results)):
            chunk_filename, start_time, end_time, result, score = chunk_results[i]

            # Apply majority voting in a small window around the current prediction
            window_start = max(i - smoothing_window // 2, 0)
            window_end = min(i + smoothing_window // 2 + 1, num_chunks)

            # Get the most frequent prediction in the window
            window_results = all_results[window_start:window_end]
            smoothed_result = max(set(window_results), key=window_results.count)

            # Store frame predictions
            if i == len(chunk_results) - 1:
                for frame_nr in range(int(start_time), int(end_time)):
                    frame_predictions.append({
                        "frame_nr": frame_nr,
                        "surgical_step": smoothed_result
                    })
            else:
                for frame_nr in range(int(start_time), int(end_time - overlap)):
                    frame_predictions.append({
                        "frame_nr": frame_nr,
                        "surgical_step": smoothed_result
                    })

        # Save the results to a JSON file
        with open(self.output_file, 'w') as f:
            json.dump(frame_predictions, f, indent=4)

    def predict(self, fname: str) -> List[Dict[str, int]]:
        """
        Predicts the labels for video chunks generated from the input video file.
        Args:
            fname (str): The filename of the input video.
        Returns:
            List[Dict[str, int]]: A list of dictionaries containing the prediction results for each video chunk.
            Each dictionary contains:
                - 'chunk_filename' (str): The filename of the video chunk.
                - 'start_time' (float): The start time of the video chunk.
                - 'end_time' (float): The end time of the video chunk.
                - 'label' (int): The predicted label for the video chunk.
                - 'score' (float): The confidence score of the predicted label.
        """
        overlap = self.window_size * (self.overlap_percentage / 100)
        output_dir = f'video_chunks_window{self.window_size}_overlap{self.overlap_percentage}'
        os.makedirs(output_dir, exist_ok=True)
        chunks = self.split_video(fname, self.window_size, overlap, output_dir)

        chunk_results = []
        for i, (chunk_filename, start_time, end_time) in enumerate(chunks):
            pred_result = inference_recognizer(self.model, chunk_filename)

            pred_scores = pred_result.pred_score.tolist()
            score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
            score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
            top5_label = score_sorted[:5]

            # Convert the top prediction index to the corresponding label string
            label = top5_label[0][0]
            score = top5_label[0][1]
            
            if score < 0.88:
                label = 7
            
            chunk_results.append((chunk_filename, start_time, end_time, label, score))
        
        self.post_process_predictions(chunk_results, overlap)


if __name__ == "__main__":
    detector = SurgVUClassify()
    video_file = detector.input_path / "endoscopic-robotic-surgery-video.mp4"  # Example video file path for submit
    detector.predict(str(video_file))
