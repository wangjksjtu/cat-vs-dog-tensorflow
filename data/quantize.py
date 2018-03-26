from PIL import Image
import glob
import os

def quantize_img(img_dir, out_dir, quality):
    for inpath in glob.glob(os.path.join(img_dir, "*.jpg")):
        filename = inpath.split("/")[-1].strip()
        img = Image.open(inpath)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outpath = os.path.join(out_dir, filename)
        img.save(outpath, quality=quality)
        # break

if __name__ == "__main__":
    for quality in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75]:
        img_dir = "train/"
        out_dir = "quality_" + str(quality)
        print out_dir
        quantize_img(img_dir, out_dir, quality)
