import json
from collections import Counter
import os

import pandas as pd
from tqdm import tqdm

import argparse  # 添加argparse导入


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate retrieval metrics for frame or video mode')
    parser.add_argument('--mode', type=str, required=True, choices=['frame', 'video'], 
                        help='Evaluation mode: frame or video')
    
    parser.add_argument('--out_metrics_fp', type=str, required=True, 
                        help='Output Excel file path to store metrics')
    
    subparsers = parser.add_subparsers(dest='subparser_name')
    
    frame_parser = subparsers.add_parser('frame')
    frame_parser.add_argument('--data_dir', type=str, required=True, 
                             help='Directory containing val_infer_ip_step_xxx.jsonl files')
    frame_parser.add_argument('--steps', type=str, required=True, 
                             help='Comma-separated list of steps (e.g., "12499,24999")')
    frame_parser.add_argument('--epoch_nums', type=str, required=True, 
                             help='Comma-separated list of epoch numbers (e.g., "10,20")')
    frame_parser.add_argument('--k', type=int, default=20, 
                             help='Deprecated, no longer useful: Top k value for retrieval (default: 20)')

    video_parser = subparsers.add_parser('video')
    video_parser.add_argument('--freq_threshs', type=str, required=True, 
                             help='Comma-separated list of frequency thresholds (e.g., "5,8")')
    video_parser.add_argument('--retrieval_infer_fp', type=str, required=True, 
                             help='Path to retrieval inference JSONL file')
    video_parser.add_argument('--id2title_fp', type=str, 
                             help='Path to id2title JSON file (optional)')

    args = parser.parse_args()
    return args


def format_metrics_video(metrics_dict):
    data = []
    frame_threshs = sorted(metrics_dict.keys())
    thresholds = sorted(list(set([thresh for frame_thresh in frame_threshs for thresh in metrics_dict[frame_thresh].keys()])))

    columns = ["frame_threshs"] + thresholds

    for frame_thresh in frame_threshs:
        row = [frame_thresh]
        for thresh in thresholds:
            if thresh in metrics_dict[frame_thresh]:
                metric = metrics_dict[frame_thresh][thresh]
                row.append(f"Precision = {metric['precision']}\nRecall = {metric['recall']}\nF1 = {metric['f1']}")
            else:
                row.append("")
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df


def format_metrics_frame(metrics_dict):
    data = []
    epochs = sorted(metrics_dict.keys())
    thresholds = sorted(list(set([thresh for epoch in epochs for thresh in metrics_dict[epoch].keys()])))

    columns = ["Epoch"] + thresholds

    for epoch in epochs:
        row = [epoch]
        for thresh in thresholds:
            if thresh in metrics_dict[epoch]:
                metric = metrics_dict[epoch][thresh]
                # 这里只选择展示top1_acc和cover_ratio，可根据需求调整
                row.append(f"acc = {metric['top1_acc']}\nCover rate = {metric['cover_ratio']}")
            else:
                row.append("")
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

def calculate_vllm_metrics(file_path):
    """
    计算不同匹配设置下的准确率、总数据量和无效预测数量，同时按体裁统计准确率。

    :param file_path: 输入文件的路径，文件每一行是一个 JSON 字符串
    :return: 包含各种指标的字典
    """
    total_count = 0
    invalid_num = 0
    both_matched_count = 0
    name_matched_count = 0
    # 新增体裁相关统计
    genre_stats = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_count += 1
            data = json.loads(line)
            image_dir_name = data['image_dir_name']
            raw_answer = data['raw_answer']

            # 解析 ground truth
            parts = image_dir_name.split('_')
            if len(parts) >= 2:
                gt_genre = parts[0]
                gt_name = parts[1]
                # gt_name = '_'.join(parts[1:-1])
            else:
                gt_genre = ""
                gt_name = ""
            
            gt_genre = "短剧" if "短剧" in gt_genre else gt_genre

            # 初始化体裁统计
            if gt_genre not in genre_stats:
                genre_stats[gt_genre] = {
                    "total": 0,
                    "both_matched": 0,
                    "name_matched": 0
                }
            genre_stats[gt_genre]["total"] += 1

            # 解析 pred
            pred_genre = ""
            pred_name = ""
            try:
                start_index = raw_answer.find('来自') + 2
                end_index = raw_answer.find('《')
                if start_index < end_index:
                    pred_genre = raw_answer[start_index:end_index]
                start_index = raw_answer.find('《') + 1
                end_index = raw_answer.find('》')
                if start_index < end_index:
                    pred_name = raw_answer[start_index:end_index]
            except IndexError:
                invalid_num += 1
                continue

            if not pred_genre or not pred_name:
                invalid_num += 1
                continue

            # 同时考虑体裁和作品名的匹配
            if pred_genre == gt_genre and pred_name == gt_name:
                both_matched_count += 1
                genre_stats[gt_genre]["both_matched"] += 1

            # 只考虑作品名匹配
            if pred_name == gt_name:
                name_matched_count += 1
                genre_stats[gt_genre]["name_matched"] += 1
            # if pred_name == gt_name and pred_genre != gt_genre:
            #     print(pred_name, gt_name, pred_genre, gt_genre)

    both_acc = round(both_matched_count / total_count, 4) if total_count > 0 else 0
    name_acc = round(name_matched_count / total_count, 4) if total_count > 0 else 0

    # 计算各体裁的准确率
    genre_metrics = {}
    for genre, stats in genre_stats.items():
        genre_both_acc = round(stats["both_matched"] / stats["total"], 4) if stats["total"] > 0 else 0
        genre_name_acc = round(stats["name_matched"] / stats["total"], 4) if stats["total"] > 0 else 0
        genre_metrics[genre] = {
            "both_acc": genre_both_acc,
            "name_acc": genre_name_acc,
            "count": stats["total"]
        }

    return {
        "both_acc": both_acc,
        "name_acc": name_acc,
        "total_count": total_count,
        "invalid_num": invalid_num,
        "genre_metrics": genre_metrics
    }


def calculate_retrieval_metrics_for_video(retrieval_infer_fp, freq_threshs="5", id2title_fp=None):
    id2title = {}  # fix book_title is empty cases.
    if id2title_fp:
        with open(id2title_fp, "r") as f:
            id2title = json.load(f)

    all_data = []
    with open(retrieval_infer_fp, 'r', encoding='utf-8') as file:
        for l in tqdm(file, desc="Loading retrieval inference data"):
            try:
                all_data.append(json.loads(l))
            except:
                pass
    video2pred_dict = {}
    for d in all_data:
        video_name = d["gt_label"].split("/")[-2].replace("_frames", "")
        video2pred_dict[video_name] = video2pred_dict.get(video_name, []) + [d["retrieval"][0]]
    
    metrics_dict = {}
    freq_thresh_list = freq_threshs.split(",")
    for freq_thresh in tqdm(freq_thresh_list, desc="Calculating retrieval metrics"):
        freq_thresh = int(freq_thresh)
        metrics_dict[freq_thresh] = {}
        for min_thresh in [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            total_count = 0
            cover_count = 0
            correct_count = 0
            for video_name, preds in video2pred_dict.items():
                gt = video_name.split("_")[-1]
                if gt in id2title:
                    print(gt, id2title[gt])
                    gt = id2title[gt]
                total_count += 1
                preds = [pred for pred in preds if float(pred["distance"]) >= min_thresh]
                if not preds:
                    continue
                preds = [pred["label"].split("_")[-1] for pred in preds]
                pred_val = Counter(preds).most_common(1)[0][0]
                pred_freq = Counter(preds).most_common(1)[0][1]
                if pred_freq < freq_thresh:
                    continue
                cover_count += 1
                if pred_val == gt:
                    correct_count += 1

            precision = correct_count / cover_count if cover_count > 0 else 0
            recall = correct_count / total_count if total_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            cover_ratio = round(cover_count / total_count, 4) if total_count > 0 else 0
            metrics_dict[freq_thresh][min_thresh] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "total_count": round(total_count, 4),
                "cover_ratio": cover_ratio,
                "thresh": min_thresh
            }
    return metrics_dict


def calculate_retrieval_metrics(retrieval_infer_fp, k=5):
    # for frame
    all_data = []
    with open(retrieval_infer_fp, 'r', encoding='utf-8') as file:
        for l in file:
            try:
                all_data.append(json.loads(l))
            except:
                pass
    metrics_dict = {}
    for min_thresh in [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        total_count = 0
        cover_count = 0
        top1_correct_count = 0
        topk_correct_count = 0
        for data in all_data:
            retrieval_results = data['retrieval']
            gt_label = data['gt_label'].split('/')[-3].split("_")[-1]
            total_count += 1
            # 获取 top1 的 label
            top1_label = retrieval_results[0]['label'].split("_")[-1]
            top1_score = float(retrieval_results[0]['distance'])
            if min_thresh and top1_score < min_thresh:
                continue
            cover_count += 1
            if top1_label == gt_label:
                top1_correct_count += 1

            # 获取 topk 的 labels 并找出众数
            topk_labels = [item['label'] for item in retrieval_results[:k]]
            counter = Counter(topk_labels)
            topk_pred_label = counter.most_common(1)[0][0]
            if topk_pred_label == gt_label:
                topk_correct_count += 1

        top1_acc = top1_correct_count / cover_count if cover_count > 0 else 0
        topk_acc = topk_correct_count / cover_count if cover_count > 0 else 0
        metrics_dict[min_thresh] = {
            "top1_acc": round(top1_acc, 4),
            "topk_acc": round(topk_acc, 4),
            "total_count": round(total_count, 4),
            "cover_ratio": round(cover_count / total_count, 4) if total_count > 0 else 0,
            "thresh": min_thresh
        }
    return metrics_dict


if __name__ == "__main__":
    # 创建命令行参数解析器
    # 解析参数
    args = parse_args()
    
    # 根据模式执行相应的评测逻辑
    if args.mode == 'frame':
        """
        python metrics.py --mode frame --out_metrics_fp "./frame_metrics.xlsx" frame --data_dir "/path/to/data_dir" --steps "12499,24999" --epoch_nums "10,20" --k 20
        """
        # 处理frame模式参数
        step_list = args.steps.split(',')
        epoch_num_list = list(map(int, args.epoch_nums.split(',')))
        
        if len(step_list) != len(epoch_num_list):
            raise ValueError("steps and epoch_nums must have the same length")
        
        overall_metrics = {}
        for step, epoch in zip(step_list, epoch_num_list):
            retrieval_infer_fp = os.path.join(args.data_dir, f"val_infer_ip_step_{step}.jsonl")
            res = calculate_retrieval_metrics(retrieval_infer_fp, k=args.k)
            overall_metrics[epoch] = res
        
        df = format_metrics_frame(overall_metrics)
        df.to_excel(args.out_metrics_fp, index=False)
        print(f"Frame metrics saved to {args.out_metrics_fp}")
        
    elif args.mode == 'video':
        """
        python metrics.py --mode video --out_metrics_fp "./video_metrics.xlsx" video --freq_threshs "5,8" --retrieval_infer_fp "/path/to/retrieval.jsonl" --id2title_fp "/path/to/id2title.json" 
        """
        # 处理video模式参数
        overall_metrics = calculate_retrieval_metrics_for_video(
            retrieval_infer_fp=args.retrieval_infer_fp,
            freq_threshs=args.freq_threshs,
            id2title_fp=args.id2title_fp
        )
        df = format_metrics_video(overall_metrics)
        df.to_excel(args.out_metrics_fp, index=False)
        print(f"Video metrics saved to {args.out_metrics_fp}")
