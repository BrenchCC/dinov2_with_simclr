import json
import matplotlib.pyplot as plt

def read_total_loss(file_path):
    total_losses = []
    with open(file_path, "r", encoding = 'utf-8') as file:
        for line in file:
            data = json.loads(line)
            total_losses.append(data["total_loss"])
    return total_losses

def plot_loss_comparison(file_path1, file_path2, out_fp):
    # read two path json data to get pre total loss data
    a_losses = read_total_loss(file_path1)
    b_losses = read_total_loss(file_path2)

    # set image clarity
    plt.rcParams['figure.dpi'] = 300

    # start to draw image
    plt.plot(a_losses, label = "A")
    plt.plot(b_losses, label = "B")

    # set image title and labels
    plt.title("Total Loss Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.legend()

    # show the grid
    plt.grid(True)

    # show and save the image
    plt.show()
    plt.savefig(out_fp)
    plt.close()

if __name__ == "__main__":
    fp1 = "demo1/training_metrics.json"
    fp2 = "demo2/training_metrics.json"
    out_fp = "./total_loss.png"
    plot_loss_comparison(fp1, fp2, out_fp)




