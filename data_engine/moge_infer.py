import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import cv2
import torch
import numpy as np


original_cwd = os.getcwd()
moge_dir = os.path.join(original_cwd, 'MoGe')
try:
    os.chdir(moge_dir)
    if moge_dir not in sys.path:
        sys.path.insert(0, moge_dir)
    from moge.model.v1 import MoGeModel
finally:
    os.chdir(original_cwd)


def main(image_dir, output_dir):    
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)  
    model.eval()

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    image_paths = sorted(itertools.chain(*(Path(image_dir).rglob(f'*.{suffix}') for suffix in include_suffices)))


    # Check for existing NPY files in output directory
    output_npy_files = list(Path(output_dir).glob('*.npy'))
    if len(output_npy_files) >= len(image_paths):
        return

    for image_path in image_paths:
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        # Inference
        output = model.infer(image_tensor, fov_x=None, resolution_level=9, num_tokens=None, use_fp16=True)
        depth = output['depth'].cpu().numpy()

        npy_output_dir = Path(output_dir)
        npy_output_dir.mkdir(exist_ok=True, parents=True)

        # Construct filename using image_path stem
        filename = f"{image_path.stem}.npy"
        
        # Path joining
        save_file = npy_output_dir.joinpath(filename)  
        
        # Save depth map as NPY
        np.save(str(save_file), depth)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MoGe depth estimation.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)
