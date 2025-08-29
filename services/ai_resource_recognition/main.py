import argparse
import os
import json
from collections import Counter

import numpy as np

from ai_resource_recognition.faiss_index_service import FaissIndexGPU
from ai_resource_recognition.dinov2_resource_recognition import DinoV2Encoder
from ai_resource_recognition.doubao_vision_resource_recognition import recognize_entity_with_doubao
from ai_resource_recognition.video_frame_extraction.video_frame_sampler import VideoFrameSampler
from ai_resource_recognition.video_frame_extraction.utils import build_transform
from ai_resource_recognition.video_frame_extraction.video_retrieval import download_video_by_gid


IMAGE_RESIZE=448


def parse_args():
    parser = argparse.ArgumentParser(description="资源识别提取器")

    # DinoV2参数
    parser.add_argument("--dino-model-type", type=str, required=True, choices=["dinov2", "dinov2_mlp"], help="DinoV2模型类型")
    parser.add_argument("--dino-config-file", type=str, required=True, help="DinoV2配置文件路径")
    parser.add_argument("--dino-pretrained-weight", type=str, required=True, help="DinoV2预训练权重路径")
    parser.add_argument("--dino-out-dim", type=int, default=128, help="DinoV2输出维度")
    
    # Faiss参数
    parser.add_argument("--faiss-index-path", type=str, default="", help="Faiss索引文件路径")
    parser.add_argument("--faiss-embedding-path", type=str, default="", help="Faiss索引原始embedding文件路径")
    parser.add_argument("--faiss-label-path", type=str, required=True, help="Faiss标签文件路径")
    parser.add_argument("--faiss-index-type", type=str, default="FlatIP", choices=["FlatL2", "FlatIP"], help="Faiss索引类型")
    
    # Doubao Vision参数
    parser.add_argument("--doubao-model-ep", type=str, required=True, help="Doubao模型端点")
    parser.add_argument("--doubao-system-prompt-fp", type=str, required=True, help="Doubao系统提示词")
    
    # 数据处理参数
    parser.add_argument("--title2id-fp", type=str, required=True, help="标题到ID的映射文件路径")
    parser.add_argument("--id2title-fp", type=str, required=True, help="ID到标题的映射文件路径")
    parser.add_argument("--freq-thresh", type=int, default=5, help="实体频率阈值")
    parser.add_argument("--score-thresh", type=float, default=0, help="检索分数阈值")

    # 待评测数据参数
    parser.add_argument("--image-paths", type=str, default="", help="图像路径，逗号分隔")
    parser.add_argument("--video-gid", type=str, default="", help="video gid")
    parser.add_argument("--num-frame", type=int, default=48, help="视频采样帧数")
    parser.add_argument("--doubao-num-images", type=int, default=3, help="Doubao模型采样帧数")
    args = parser.parse_args()
    return args


class ResourceExtractor:
    def __init__(self, dino_model_type, dino_config_file, dino_pretrained_weight, dino_out_dim,
                 faiss_index_path, faiss_embedding_path, faiss_label_path, faiss_index_type,
                 doubao_model_ep, doubao_system_prompt_fp, title2id_fp, id2title_fp):
        
        # 加载Faiss索引
        if faiss_index_path:
            self.faiss_index = FaissIndexGPU.load_index(
                index_type=faiss_index_type,
                load_path=faiss_index_path,
                label_path=faiss_label_path
            )
        elif faiss_embedding_path:
            self.faiss_index = FaissIndexGPU(
                embedding_path=faiss_embedding_path,
                label_path=faiss_label_path,
                index_type=faiss_index_type,
            )
        else:
            raise ValueError("Either faiss_index_path or faiss_embedding_path must be provided")
        
        print("*****Finish init faiss index*****")

        # 初始化DinoV2编码器
        self.dino_encoder = DinoV2Encoder(
            model_type=dino_model_type,
            config_file=dino_config_file,
            pretrained_weights=dino_pretrained_weight,
            out_dim=dino_out_dim
        )
        print("*****Finish init dino encoder*****")
        
        # 存储Doubao相关参数
        self.doubao_model_ep = doubao_model_ep
        self.doubao_system_prompt = open(doubao_system_prompt_fp, 'r', encoding='utf-8').read()
        
        # 存储标题映射文件路径
        self.title2id = json.load(open(title2id_fp, 'r', encoding='utf-8'))
        self.id2title = json.load(open(id2title_fp, 'r', encoding='utf-8'))

    def extract(self, image_paths, freq_thresh=5, score_thresh=0, doubao_num_images=3, retrieval_batch_size=16):
        # 1. 使用DinoV2提取图像embedding
        embeddings = self.dino_encoder.infer_batch(image_paths)
        
        # 2. 使用Faiss进行批量检索
        faiss_results = []
        for i in range(0, len(embeddings), retrieval_batch_size):
            batch_embeddings = np.array(embeddings[i:i+retrieval_batch_size])
            batch_results = self.faiss_index.batch_search(batch_embeddings)
            faiss_results.extend(batch_results)
        
        resource_pred_list = [item["retrieval"] for item in faiss_results]
        
        # 3. 提取DinoV2实体识别结果
        dinov2_pred_entity_id, _ = extract_dinov2_entity(resource_pred_list, freq_thresh, score_thresh)
        
        # 4. 使用Doubao Vision进行实体识别
        doubao_entity, valid_flag = recognize_entity_with_doubao(
            image_path_list=image_paths,
            model_ep=self.doubao_model_ep,
            system_prompt=self.doubao_system_prompt,
            num_images=doubao_num_images,
        )
        print(f"Doubao entity: {doubao_entity}, valid_flag: {valid_flag}")
        
        # 5. 融合两个模型的识别结果
        merged_result = merge_image_recognition_v2_result(
            doubao_pred_entity=doubao_entity,
            valid_flag=valid_flag,
            dinov2_pred_entity_id=dinov2_pred_entity_id,
            title2id=self.title2id,
            id2title=self.id2title
        )
        
        return merged_result

    @staticmethod
    def extract_video_frames(gid, save_video_fp, save_frames_dir):
        # download video
        download_video_by_gid(gid, out_fp=save_video_fp)
        video_reader = VideoFrameSampler(build_transform, output_size=IMAGE_RESIZE)
        
        # extract frames
        if not os.path.exists(save_frames_dir):
            os.makedirs(save_frames_dir)
        
        print("Start extracting video frames")
        video_reader.uniformed_sample_frames(
            save_video_fp,
            num_segments=args.num_frame,
            save_path=save_frames_dir
        )
        print("Finish extracting video frames")


def extract_dinov2_entity(resource_pred_list, freq_thresh=5, score_thresh=0):
    # only keep top1 pred for each resource_pred
    top1_resource_pred_list = [r[0] for r in resource_pred_list if len(r) > 0]
    filter_score = [pred for pred in top1_resource_pred_list if float(pred["distance"]) >= score_thresh]
    if not filter_score:
        pred_entity_id = "UNK"
        pred_freq = ""
    else:
        pred_entity_ids = [pred["label"].split("_")[-1] for pred in filter_score]
        pred_entity_id = Counter(pred_entity_ids).most_common(1)[0][0]
        pred_freq = Counter(pred_entity_ids).most_common(1)[0][1]
        # filter by freq_thresh
        if pred_freq < freq_thresh:
            pred_entity_id = "UNK"
    print(f"dinov2 pred entity id={pred_entity_id} with freq={pred_freq}")
    return pred_entity_id, pred_freq


def merge_image_recognition_v2_result(doubao_pred_entity, valid_flag, dinov2_pred_entity_id, title2id, id2title):

    def normalize_punctuation(text):
        if not text:
            return ""
        # 英文转中文标点
        punctuation_map = {
            ',': '，', '.': '。', '!': '！', '?': '？', ':': '：', ';': '；',
            '(': '（', ')': '）', '[': '【', ']': '】', '{': '｛', '}': '｝',
            '<': '＜', '>': '＞', '"': '“', "'": '’', '-': '－', '_': '＿'
        }
        for eng, chn in punctuation_map.items():
            text = text.replace(eng, chn)
        return text.strip()
    
    # 构建规范化标题到ID的映射
    normalized_title2id = {}
    for title, ids in title2id.items():
        normalized_title = normalize_punctuation(title)
        normalized_title2id[normalized_title] = ids
    
    doubao_normalized = normalize_punctuation(doubao_pred_entity)
    doubao_id = normalized_title2id.get(doubao_normalized, [])

    if dinov2_pred_entity_id == "UNK":
        dinov2_pred_entity = "UNK"
    else:
        dinov2_pred_entity = id2title[dinov2_pred_entity_id]
        
    # 融合策略
    pred_entity = "UNK"
    pred_id = "UNK"
    strategy = ""

    if valid_flag == "否":
        strategy = "filter"

    elif doubao_pred_entity and doubao_id:
        # if multiple ids are mapped for doubao vision entity, keep dinov2 result.
        if len(doubao_id) > 1:
            pred_entity = dinov2_pred_entity
            pred_id = dinov2_pred_entity_id
            strategy = "dinov2"
        else:
            pred_entity = doubao_pred_entity
            pred_id = doubao_id[0]
            if doubao_pred_entity == dinov2_pred_entity and pred_id == dinov2_pred_entity_id:
                strategy = "both"
            else:
                strategy = "doubao_vision"
    else:
        pred_entity = dinov2_pred_entity
        pred_id = dinov2_pred_entity_id
        strategy = "dinov2"
    result = {
        "pred_entity": pred_entity,
        "pred_entity_id": pred_id,
        "pred_strategy": strategy
    }
    return result


if __name__ == "__main__":
    
    args = parse_args()
    # 初始化资源提取器
    extractor = ResourceExtractor(
        dino_model_type=args.dino_model_type,
        dino_config_file=args.dino_config_file,
        dino_pretrained_weight=args.dino_pretrained_weight,
        dino_out_dim=args.dino_out_dim,
        faiss_index_path=args.faiss_index_path,
        faiss_embedding_path=args.faiss_embedding_path,
        faiss_label_path=args.faiss_label_path,
        faiss_index_type=args.faiss_index_type,
        doubao_model_ep=args.doubao_model_ep,
        doubao_system_prompt_fp=args.doubao_system_prompt_fp,
        title2id_fp=args.title2id_fp,
        id2title_fp=args.id2title_fp
    )
    
    # 处理图像并输出结果
    if args.image_paths:
        image_paths = args.image_paths.split(",")
        result = extractor.extract(image_paths, args.freq_thresh, args.score_thresh, doubao_num_images=args.doubao_num_images)
    elif args.video_gid:
        save_frames_dir = f"./temp/{args.video_gid}"
        os.makedirs(save_frames_dir, exist_ok=True)
        extractor.extract_video_frames(args.video_gid, os.path.join(save_frames_dir, f"{args.video_gid}.mp4"), save_frames_dir)
        image_fp_list = sorted([os.path.join(save_frames_dir, i) for i in os.listdir(save_frames_dir) if i.endswith((".png", ".jpeg", ".jpg"))])
        if not image_fp_list:
            print("Frame extraction fails")
            reuslt = ""
        else:
            image_type = image_fp_list[0].rsplit(".", 1)[1]
            result = extractor.extract(image_fp_list, args.freq_thresh, args.score_thresh, doubao_num_images=args.doubao_num_images)
    else:
        raise ValueError("Either image_paths or video_gid must be provided")
    print("Final result:", json.dumps(result, ensure_ascii=False, indent=2))