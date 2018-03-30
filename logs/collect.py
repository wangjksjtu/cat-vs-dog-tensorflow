import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--quality', type=int, default=15, help='Image quality [default: 15]')
parser.add_argument('--setting', type=int, default=9, help='Model architecture (0-9) [default: 9]')
parser.add_argument('--model', type=str, default="", help='Model architecture description (0-9) [default: ""]')
FLAGS = parser.parse_args()

SETTING = FLAGS.setting
MODEL = FLAGS.model
QUALITY = FLAGS.quality

model_list = ["io-units_x1+cnn_pooling2",
              "io-units_x1+cnn_pooling",
              "io-units_x2",
              "io-layer",
              "io-units_x1",
              "io-units_x1+layer",
              "io-units_x1+units_x1",
              "io-units_x1+units_x2",
              "baseline",
              "original",
             ]

if MODEL == "":
    SETTING = model_list[SETTING]
else:
    SETTING = MODEL

log_path = SETTING + "/quality_" + str(QUALITY) + "/train/log_train.txt"
k = 10

def max_acc(log_path):
    os.system("cat %s | grep eval\ accuracy | awk '{print $3}' | sort | tail -n 1" % log_path)

def max_avg_cls_acc(log_path):
    os.system("cat %s | grep eval\ avg | awk '{print $5}' | sort | tail -n 1" % log_path)

def top_k_acc(log_path, k):
    os.system("cat %s | grep eval\ accuracy | awk '{print $3}' | sort | tail -n %s | awk '{ sum += $0 } END { print sum / NR }'" % (log_path, str(k)))

def top_k_avg_cls_acc(log_path, k):
    os.system("cat %s | grep eval\ avg | awk '{print $5}' | sort | tail -n %s | awk '{ sum += $0 } END { print sum / NR }'" % (log_path, str(k)))

if __name__ == "__main__":
    print log_path
    # max_acc(log_path)
    top_k_acc(log_path, k)
    # max_avg_cls_acc(log_path)
    # top_k_avg_cls_acc(log_path, k)

