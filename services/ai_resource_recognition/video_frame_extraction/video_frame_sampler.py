import os
import re
import torch
import time
import cv2
import shutil
import logging
import requests
import imagehash
import numpy as np
from pyarrow import fs

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from video_frame_extraction.utils import dynamic_preprocess


logger = logging.getLogger(__name__)


class VideoFrameSampler:
    def __init__(self, transform=None, output_size=448):
        self.output_size = output_size
        self.transform = transform(input_size=output_size) if transform is not None else None
        # self.hdfs, _ = fs.FileSystem.from_uri('hdfs://haruna/home/')
        
        # os.makedirs(f"./tmp", exist_ok=True)
        # self.tmp_file_path = f"./tmp/tmp_video_{time.time()}.mp4"

    def tmp_clear(self):
        # 删除目录下的所有内容
        shutil.rmtree(self.tmp_path)
        # 重新创建空的目录
        os.mkdir(self.tmp_path)

    def _vr_gen(self, video_path, cpu_index=0, num_threads=1):
        if 'hdfs://' in video_path:
            with self.hdfs.open_input_file(video_path) as f:
                with open(self.tmp_file_path, 'wb') as temp_file:
                    logger.warning(f"Write file to {self.tmp_file_path}")
                    temp_file.write(f.read())
                cap = cv2.VideoCapture(self.tmp_file_path)
        elif 'http://' in video_path or 'https://' in video_path:
            with open(self.tmp_file_path, 'wb') as temp_file:
                response = requests.get(video_path, stream=True)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            cap = cv2.VideoCapture(self.tmp_file_path)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        return cap

    def _generate_pixel_values(self,
                            video_path,
                            frame_indices,
                            output_dir,
                            max_frames=40,
                            debug=True,
                            max_workers=1):
        if debug:
            print(f"Saving images to {output_dir}")

        def process_frame(frame_index):
            local_cap = self._vr_gen(video_path) # cv2.VideoCapture(video_path)  # 每个线程独立创建 VideoCapture
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = local_cap.read()
            local_cap.release()
            if not ret:
                return None, None

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            width, height = img.size
            if min(width, height) > 720:
                # if width < height:
                #     new_width = 448
                #     new_height = int(height * (new_width / width))
                # else:
                #     new_height = 448
                #     new_width = int(width * (new_height / height))
                img = img.resize((720, 1280), Image.Resampling.BICUBIC)
                
            current_hash = imagehash.phash(img)
            return frame_index, current_hash

        if debug:
            print("Calculating image phash values and deduplicating.")
        # Step 1: Calculate phash values for all frames in single process
        hash_list = []
        frame_hash_map = {}  # Map frame_index to its phash
        cap = self._vr_gen(video_path)
        # frame_indices_iter = tqdm(frame_indices, total=len(frame_indices))
        frame_indices_iter = frame_indices

        for frame_index in frame_indices_iter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            current_hash = imagehash.phash(img)
            if frame_index is not None and current_hash is not None:
                frame_hash_map[frame_index] = current_hash

        # Step 2: Deduplicate frames based on phash values
        selected_indices = []
        hash_list = []
        threshold = 2

        for frame_index, current_hash in sorted(frame_hash_map.items()):
            is_duplicate = any(abs(current_hash - prev_hash) < threshold for prev_hash in hash_list)
            if not is_duplicate:
                selected_indices.append(frame_index)
                hash_list.append(current_hash)

        # Step 3: Gradually increase threshold if frames exceed max_frames
        while len(selected_indices) > max_frames:
            # if debug:
            #     print(f"Increasing threshold to {threshold}")

            new_selected_indices = [selected_indices[0]]
            new_hash_list = [hash_list[0]]

            for i in range(1, len(selected_indices)):
                is_duplicate = any(abs(hash_list[i] - prev_hash) < threshold for prev_hash in new_hash_list)
                if not is_duplicate:
                    new_selected_indices.append(selected_indices[i])
                    new_hash_list.append(hash_list[i])
            
            # 当提高阈值导致最终帧数不够时，从头部进行补齐, 6 是一个经验值，不大于 6 时相似度很高，可以去掉而少于目标图片数量
            if threshold > 6 and len(new_selected_indices) < max_frames:
                for idx in selected_indices:
                    if idx not in new_selected_indices:
                        new_selected_indices.append(idx)
                    if len(new_selected_indices) == max_frames:
                        break
                selected_indices = sorted(new_selected_indices)
                break # 补齐后可以直接退出去重循环了

            selected_indices = new_selected_indices
            hash_list = new_hash_list
            threshold += 2
            if threshold > 64:
                break

        # Truncate to max_frames
        if len(selected_indices) > max_frames:
            selected_indices = selected_indices[:max_frames]
            logger.warning(f"{video_path} haven't be deduped to {max_frames} frames. Truncated from {len(selected_indices)} frames.")

        # Step 4: Save images in parallel
        images = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # frame_indices_iter = tqdm(selected_indices, total=len(selected_indices))
        frame_indices_iter = selected_indices

        for frame_index in frame_indices_iter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                if output_dir:
                    img.save(os.path.join(output_dir, f'frame_{frame_index:04d}.jpg'))
                images.append((frame_index, img))

        # 释放视频捕获对象
        cap.release()
        # self.tmp_clear()
        images.sort(key = lambda x: x[0])
        return [img for _, img in images]

    def frames_to_tensor(self, images, max_crop_num):
        pixel_values_list, num_patches_list = [], []
        for frame_index, img in enumerate(images):
            img = dynamic_preprocess(
                img,
                image_size=self.output_size,
                use_thumbnail=True,
                max_num=max_crop_num)
            pixel_values = [self.transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values_list = torch.stack(pixel_values_list).reshape([-1, 3, self.output_size, self.output_size])
        
        return pixel_values_list, num_patches_list

    def head_first_sample_frames(self,
                                video_path,
                                head_duration=5,
                                num_segments=8,
                                save_path="./images"):
        """
        Non-uniform frame sampling. Sample half of the frames in the first 5 seconds,
        and the other half in the remaining duration.

        Args:
            video_path (str): Path to the video file.
            head_duration (int, optional): First half frames sampling duration. Defaults to 5.
            num_segments (int, optional): Number of segments (frames) to sample. Defaults to 8.
            save_path (str, optional): Path to save the frames. Defaults to "./images".
        Returns:
            images: List of sampled frames.
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        debug = True if save_path else False

        if video_path.startswith("hdfs://"):
            self.tmp_file_path = os.path.join("./tmp", Path(video_path).name)
            local_video_path = self.tmp_file_path
        elif video_path.startswith("http"):
            self.tmp_file_path = f"./tmp/tmp_video_{time.time()}.mp4"
            local_video_path = self.tmp_file_path
        else:
            local_video_path = video_path

        cap = self._vr_gen(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        head_frame_count = min(int(fps * head_duration), frame_count)

        num_first_half_frames = num_segments // 2
        num_remaining_half_frames = num_segments - num_first_half_frames

        first_half_indices = list(range(0, head_frame_count, max(1, head_frame_count // num_first_half_frames)))
        remaining_duration_frame_count = frame_count - head_frame_count
        remaining_half_indices = list(range(head_frame_count, frame_count, max(1, remaining_duration_frame_count // num_remaining_half_frames)))

        frame_indices = first_half_indices + remaining_half_indices
        frame_indices = sorted(frame_indices[:num_segments])
        
        images = self._generate_pixel_values(
                    local_video_path,
                    frame_indices,
                    save_path,
                    debug=debug,
                    max_frames=num_segments)
        if video_path.startswith("hdfs://"):
            os.remove(self.tmp_file_path)

        return images

    def uniformed_sample_frames(self,
                                video_path,
                                target_fps=1,
                                num_segments=8,
                                save_path="./images"):
        assert target_fps > 0, "target_fps must be greater than 0"
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        debug = True if save_path else False

        if video_path.startswith("hdfs://"):
            self.tmp_file_path = os.path.join("./tmp", Path(video_path).name)
            local_video_path = self.tmp_file_path
        elif video_path.startswith("http"):
            self.tmp_file_path = f"./tmp/tmp_video_{time.time()}.mp4"
            local_video_path = self.tmp_file_path
        else:
            local_video_path = video_path

        cap = self._vr_gen(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # # Only consider first 10min videos for efficiency
        # frame_count = min(int(fps * 600), frame_count)
        print(f"video {video_path} has {fps} fps")
        
        step = int(fps / target_fps) if target_fps is not None else 1
        step = step if step > 1 else 1
        frame_indices = range(0, frame_count, step)
        images = self._generate_pixel_values(
                    local_video_path,
                    frame_indices,
                    save_path,
                    debug=debug,
                    max_frames=num_segments)
        if video_path.startswith("hdfs://"):
            os.remove(self.tmp_file_path)
        return images
    
    def slowfast_sample_frames(self, 
                               video_path, 
                               slow_frame_num, 
                               fast_frame_num, 
                               max_frames,
                               sample_strategy="uniform", 
                               save_path="./images", 
                               with_phash=False):
        # If with_phash = True, fast_frame_num + slow_frame_num can be larger than max_frames, otherwise, these two should be the same.
        if not with_phash:
            if max_frames!= fast_frame_num + slow_frame_num:
                raise ValueError("If with_phash = False, max_frames should be equal to slow_frame_num plus fast_frame_num.")
        vr = self._vr_gen(video_path)
        fps = vr.get_avg_fps()
        if sample_strategy == "uniform":
            frame_interval = int(len(vr) / max_frames)
            if frame_interval == 0:
                frame_interval = 1
            frame_indices = range(0, len(vr), frame_interval)

        elif sample_strategy == "fix":
            max_frame = len(vr) - 1
            frame_indices = self.get_index(None, fps, max_frame, first_idx=0, num_segments=max_frames)
        else:
            raise ValueError("Invalid sample strategy. Sample strategy should be either 'fix' or 'uniform'.")
        
        debug = True if save_path else False
        frame_imgs = self._generate_pixel_values(vr, frame_indices, save_path, 
                                                 debug=debug, max_frames=max_frames, with_phash=with_phash)
        if not frame_imgs:
            return {}

        # Previous strategy: slow_frame_imgs are uniformly sampled from fast_frame_imgs, following stragtegy from SF-Llava.
        # Current stratgy: Adding interleaving slow & fast frame strategy from "LLAVA-VIDEO_slowfast".
        slow_img_internval = int(len(frame_imgs) / slow_frame_num)
        slow_frame_ids = list(range(0, len(frame_imgs), slow_img_internval))[:slow_frame_num]
        fast_frame_ids = [i for i in range(len(frame_imgs)) if i not in slow_frame_ids]

        return {"frame_imgs": frame_imgs, "slow_frame_ids": slow_frame_ids, "fast_frame_ids": fast_frame_ids}

        

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        '''
        Args
            bound：start, end 是起止时间/秒
            num_segments: 将视频切分的片段数，同时也是采样帧数
        Return:
            List of frame index
        '''
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def fixed_sample_frames(self, video_path, bound=None, num_segments=32,
                            save_path="./images"):
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        debug = True if save_path else False
        
        if video_path.startswith("hdfs://"):
            self.tmp_file_path = os.path.join("./tmp", Path(video_path).name)
            local_video_path = self.tmp_file_path
        elif video_path.startswith("http"):
            self.tmp_file_path = f"./tmp/tmp_video_{time.time()}.mp4"
            local_video_path = self.tmp_file_path
        else:
            local_video_path = video_path

        cap = self._vr_gen(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_indices = self.get_index(bound, fps, frame_count, first_idx=0, num_segments=num_segments)
        
        images = self._generate_pixel_values(
                    local_video_path,
                    frame_indices,
                    save_path,
                    debug=debug,
                    max_frames=num_segments)
        if video_path.startswith("hdfs://"):
            os.remove(self.tmp_file_path)
        return images

    def frames_resampler(self, img_path, frames, num_segments=32, debug=False, max_workers=1):
        def get_image_phash(img):
            try:
                with open(img, 'rb') as f:
                    current_hash = imagehash.phash(Image.open(f))
                frame_index = re.match(r'frame_(\d+)\.jpg', Path(img).name).group(1)
                return int(frame_index), current_hash
            except Exception as e:
                logger.error(f"Error calculating phash for image {img}: {e}")
                return None, None
        
        imgs = [f"{img_path}/{frame}" for frame in frames]
        # Step 1: Calculate phash values for all frames in parallel
        hash_list = []
        frame_hash_map = {}  # Map frame_index to its phash
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {executor.submit(get_image_phash, img): img for img in imgs}
            for future in tqdm(as_completed(future_to_frame), total=len(frames)):
                frame_index, current_hash = future.result()
                if frame_index is not None and current_hash is not None:
                    frame_hash_map[frame_index] = current_hash

        # Step 2: Deduplicate frames based on phash values
        selected_indices = []
        hash_list = []
        threshold = 2

        for frame_index, current_hash in sorted(frame_hash_map.items()):
            is_duplicate = any(abs(current_hash - prev_hash) < threshold for prev_hash in hash_list)
            if not is_duplicate:
                selected_indices.append(frame_index)
                hash_list.append(current_hash)

        # Step 3: Gradually increase threshold if frames exceed num_segments
        while len(selected_indices) > num_segments:
            if debug: 
                logger.info(f"Increasing threshold to {threshold}")

            new_selected_indices = [selected_indices[0]]
            new_hash_list = [hash_list[0]]

            for i in range(1, len(selected_indices)):
                is_duplicate = any(abs(hash_list[i] - prev_hash) < threshold for prev_hash in new_hash_list)
                if not is_duplicate:
                    new_selected_indices.append(selected_indices[i])
                    new_hash_list.append(hash_list[i])
            
            # 当提高阈值导致最终帧数不够时，从头部进行补齐, 6 是一个经验值，不大于 6 时相似度很高，可以去掉而少于目标图片数量
            if threshold > 6 and len(new_selected_indices) < num_segments:
                for idx in selected_indices:
                    if idx not in new_selected_indices:
                        new_selected_indices.append(idx)
                    if len(new_selected_indices) == num_segments:
                        break
                selected_indices = sorted(new_selected_indices)
                break # 补齐后可以直接退出去重循环了

            selected_indices = new_selected_indices
            hash_list = new_hash_list
            threshold += 2
            if threshold > 64:
                break

        # Truncate to num_segments
        if len(selected_indices) > num_segments:
            selected_indices = selected_indices[:num_segments]
            logger.warning(f"{img_path} haven't be deduped to {num_segments} frames. Truncated from {len(selected_indices)} frames.")

        return [f'frame_{idx:04d}.jpg' for idx in selected_indices]