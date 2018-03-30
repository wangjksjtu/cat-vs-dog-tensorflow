from PIL import Image
import glob
import os

def quantize_img(img_dir, out_dir, quality, suffix=".jpg"):
    for inpath in glob.glob(os.path.join(img_dir, "*" + suffix)):
        filename = inpath.split("/")[-1].strip()
        img = Image.open(inpath)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outpath = os.path.join(out_dir, filename)
        img.save(outpath, quality=quality)
        # break

def list_dirs(dataset_path):
    dir_list = glob.glob(os.path.join(dataset_path, "train/n*/images"))
    return dir_list

def quantize_all(dataset_path, save_path, quality):
    os.system("cp -r %s %s" % (dataset_path, save_path))
    dir_list = list_dirs(dataset_path)
    for img_dir in dir_list:
        out_dir = os.path.join(save_path, img_dir[img_dir.index("/")+1 : ])
        quantize_img(img_dir, out_dir, quality, suffix=".JPEG")
        # quantize_image(img_dir, )

if __name__ == "__main__":

    for quality in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75]:
        # img_dir = "train/"
        # out_dir = "quality_" + str(quality)
        # print out_dir
        output_dir = "quality_" + str(quality)
        print output_dir
        quantize_all("tiny-imagenet-200", output_dir, quality)
