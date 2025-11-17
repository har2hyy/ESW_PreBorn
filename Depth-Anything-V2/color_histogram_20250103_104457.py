
import cv2
import matplotlib.pyplot as plt
import os

# Path to your color depth map
img_path = '/home/harshyy/Desktop/20250103_104457.png'

# Load image in color
img = cv2.imread(img_path)

def save_histogram(fig, out_path):
    fig.savefig(out_path)
    print(f"Histogram saved to {out_path}")

def plot_color_histogram(img, save_path=None):
    channels = ('b', 'g', 'r')
    colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
    plt.figure(figsize=(8, 5))
    for i, channel in enumerate(channels):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=colors[channel], label=f'{channel.upper()} channel')
        plt.xlim([0, 256])
    plt.title(f'Color Histogram for {os.path.basename(img_path)}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_histogram(plt, save_path)
    plt.show()

def plot_grayscale_histogram(img, save_path=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(8, 5))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist, color='black', label='Grayscale')
    plt.xlim([0, 256])
    plt.title(f'Grayscale Histogram for {os.path.basename(img_path)}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_histogram(plt, save_path)
    plt.show()

# Choose histogram type: 'color' or 'grayscale'
hist_type = 'grayscale'  # Change to 'grayscale' for black and white histogram

out_img = f"{os.path.splitext(img_path)[0]}_histogram_{hist_type}.png"

if hist_type == 'color':
    plot_color_histogram(img, save_path=out_img)
else:
    plot_grayscale_histogram(img, save_path=out_img)
