import argparse
import os
import glob

from tqdm import tqdm

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Merge validation files.')
    parser.add_argument('--data_dir', required=True, help='Directory containing embedding and label files')
    parser.add_argument('--val_suffix', required=True, help='Common suffix for embedding files')
    parser.add_argument('--val_out_fn', required=True, help='Output filename for merged labels')
    
    args = parser.parse_args()
    
    # 获取所有embedding和label文件
    val_files = glob.glob(os.path.join(args.data_dir, f"*{args.val_suffix}"))
    
    # 按文件名排序，确保处理顺序一致
    val_files.sort()
    merge_val_data = []
    for val_file in tqdm(val_files, desc="Merging validation files"):
        with open(val_file, "r") as f:
            val_data = [l.strip() for l in f]
            merge_val_data.extend(val_data)
    val_out_fp = os.path.join(args.data_dir, args.val_out_fn)
    with open(val_out_fp, "w") as f:
        f.write("\n".join(merge_val_data))
    print(f"Successfully merged {len(merge_val_data)} validation result")

if __name__ == "__main__":
    main()
