import json
import matplotlib.pyplot as plt
import os

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss comparison from training metrics')
    parser.add_argument('--fps', required=True, help='Comma-separated file paths to training metrics JSON files')
    parser.add_argument('--out_fp', required=True, help='Output file path for the plot image')
    parser.add_argument('--legend_names', help='Comma-separated legend names for each file path')
    
    args = parser.parse_args()
    return args


def read_total_loss(file_path):
    total_losses = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            total_losses.append(data['total_loss'])
    return total_losses

def plot_loss_comparison(file_paths, out_fp, legend_names=None):
    if legend_names is None:
        legend_names = [os.path.basename(os.path.dirname(fp)) for fp in file_paths]
    elif len(legend_names) != len(file_paths):
        raise ValueError("图例名称的数量必须与文件路径的数量相同")

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    for file_path, legend_name in zip(file_paths, legend_names):
        losses = read_total_loss(file_path)
        plt.plot(losses, label=legend_name)

    # 设置图表标题和坐标轴标签
    plt.title('Total Loss Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 保存图表
    plt.savefig(out_fp)
    # 显示图表
    plt.show()
    # 关闭图表
    plt.close()

if __name__ == "__main__":
    args = parse_args()

    fps = args.fps.split(',')
    out_fp = args.out_fp
    legend_names = args.legend_names.split(',') if args.legend_names else None
    
    plot_loss_comparison(fps, out_fp, legend_names)