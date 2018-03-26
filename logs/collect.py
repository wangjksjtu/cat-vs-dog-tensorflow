import os

log_path = "quality_0/train/log_train.txt"
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
    max_acc(log_path)
    # top_k_acc(log_path, k)
    # max_avg_cls_acc(log_path)
    # top_k_avg_cls_acc(log_path, k)

