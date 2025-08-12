import argparse
import os
import glob
import torch

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Merge embedding and label files.')
    parser.add_argument('--data_dir', required=True, help='Directory containing embedding and label files')
    parser.add_argument('--embed_suffix', required=True, help='Common prefix for embedding files')
    parser.add_argument('--label_suffix', required=True, help='Common prefix for label files')
    parser.add_argument("--data_prefix", required=True, help='Common prefix for data files')
    parser.add_argument('--embed_out_fn', required=True, help='Output filename for merged embeddings')
    parser.add_argument('--label_out_fn', required=True, help='Output filename for merged labels')
    
    args = parser.parse_args()
    
    # 获取所有embedding和label文件
    embed_files = glob.glob(os.path.join(args.data_dir, f"{args.data_prefix}*{args.embed_suffix}"))
    label_files = glob.glob(os.path.join(args.data_dir, f"{args.data_prefix}*{args.label_suffix}"))
    
    # 提取文件的后缀部分用于匹配
    def get_prefix(filename, suffix):
        return filename[:-len(suffix)]
    
    # 创建label文件的映射：前缀 -> 文件路径
    label_prefix_map = {}
    for label_file in label_files:
        prefix = get_prefix(os.path.basename(label_file), args.label_suffix)
        label_prefix_map[prefix] = label_file
    
    # 按顺序匹配embedding和label文件
    merged_embeddings = []
    merged_labels = []
    
    # 按文件名排序，确保处理顺序一致
    embed_files.sort()
    
    for embed_file in embed_files:
        # 获取embedding文件的前缀
        prefix = get_prefix(os.path.basename(embed_file), args.embed_suffix)
        
        # 查找对应的label文件
        if prefix in label_prefix_map:
            label_file = label_prefix_map[prefix]
            
            # 加载embedding和label
            embedding = torch.load(embed_file)
            with open(label_file, "r") as f:
                label = [l.strip() for l in f]
            
            # 添加到合并列表
            merged_embeddings.append(embedding)
            merged_labels.extend(label)
            
            # 从映射中移除已处理的label，避免重复处理
            del label_prefix_map[prefix]
        else:
            print(f"Warning: No matching label file found for embedding file {embed_file}")
    
    # 检查是否有未匹配的label文件
    if label_prefix_map:
        print(f"Warning: The following label files have no matching embedding files:")
        for prefix, label_file in label_prefix_map.items():
            print(f"  {label_file}")
    
    # 合并embeddings和labels
    if merged_embeddings:
        merged_embeddings = torch.cat(merged_embeddings, dim=0)
        
        # 保存合并后的文件
        embed_out_path = os.path.join(args.data_dir, args.embed_out_fn)
        label_out_path = os.path.join(args.data_dir, args.label_out_fn)
        
        torch.save(merged_embeddings, embed_out_path)
        with open(label_out_path, "w") as f:
            f.write("\n".join(merged_labels))
        
        print(f"Successfully merged {len(merged_embeddings)} embeddings and labels.")
        print(f"Merged embeddings saved to: {embed_out_path}")
        print(f"Merged labels saved to: {label_out_path}")
    else:
        print("Error: No embeddings and labels were merged.")

if __name__ == "__main__":
    main()
