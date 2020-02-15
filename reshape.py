import glob
import argparse
import os

from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="make images half")

    parser.add_argument("input_dir", type=str, help="input image dir")
    parser.add_argument("output_dir", type=str, help="output image dir")
    parser.add_argument("--resolution", type=str, default="480,640")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    height, width = map(int, args.resolution.split(','))

    filepath = os.path.join(args.input_dir, "**")
    files = glob.glob(filepath, recursive=True)

    for file in tqdm(files):
        if ".png" in file or ".jpg" in file:
            img = Image.open(file)
            # trainのpng(depth画像)とtestのdepth画像はグレースケールで読み込む
            if (".png" in file and "train" in file) or ("depth" in file and "test" in file):
                # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img = Image.open(file).convert('L')
            size = (height, width)

            resized = img.resize(size)
            output_name = os.path.join(args.output_dir, os.path.relpath(file, start=args.input_dir))
            os.makedirs(os.path.dirname(output_name), exist_ok=True)
            resized.save(output_name)
