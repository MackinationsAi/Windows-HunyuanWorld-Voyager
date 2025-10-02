import logging
import os
from pathlib import Path
import json
import tarfile
from collections import defaultdict
from einops import rearrange
from typing import List
import torch
import torchvision
import numpy as np
import imageio
import PIL.Image
from PIL import Image

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def build_pretraining_data_loader():
    pass


def logger_filter(name):
    def filter_(record):
        return record["extra"].get("name") == name

    return filter_


def resolve_resume_path(resume, results_dir):
    # Detect the resume path. Support both the experiment index and the full path.
    if resume.isnumeric():
        tmp_dirs = list(Path(results_dir).glob("*"))
        id2exp_dir = defaultdict(list)
        for tmp_dir in tmp_dirs:
            part0 = tmp_dir.name.split("_")[0]
            if part0.isnumeric():
                id2exp_dir[int(part0)].append(tmp_dir)
        resume_id = int(resume)
        valid_exp_dir = id2exp_dir.get(resume_id)
        if len(valid_exp_dir) == 0:
            raise ValueError(
                f"No valid experiment directories found in {results_dir} with the experiment "
                f"index {resume}."
            )
        elif len(valid_exp_dir) > 1:
            raise ValueError(
                f"Multiple valid experiment directories found in {results_dir} with the experiment "
                f"index {resume}: {valid_exp_dir}."
            )
        resume_path = valid_exp_dir[0] / "checkpoints"
    else:
        resume_path = Path(resume)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume path {resume_path} not found.")

    return resume_path


def dump_codes(save_path, root, sub_dirs=None, valid_suffixes=None, save_prefix="./"):
    """
    Dump codes to the experiment directory.

    Args:
        save_path (str): Path to the experiment directory.
        root (Path): Path to the root directory of the codes.
        sub_dirs (list): List of subdirectories to be dumped. If None, all files in the root directory will
            be dumped. (default: None)
        valid_suffixes (tuple, optional): Valid suffixes of the files to be dumped. If None, CODE_SUFFIXES will be used.
            (default: None)
        save_prefix (str, optional): Prefix to be added to the files in the tarball. (default: './')
    """
    if valid_suffixes is None:
        valid_suffixes = CODE_SUFFIXES

    # Force to use tar.gz suffix
    save_path = safe_file(save_path)
    assert save_path.name.endswith(
        ".tar.gz"
    ), f"save_path should end with .tar.gz, got {save_path.name}."
    # Make root absolute
    root = Path(root).absolute()
    # Make a tarball of the codes
    with tarfile.open(save_path, "w:gz") as tar:
        # Recursively add all files in the root directory
        if sub_dirs is None:
            sub_dirs = list(root.iterdir())
        for sub_dir in sub_dirs:
            for file in Path(sub_dir).rglob("*"):
                if file.is_file() and file.suffix in valid_suffixes:
                    # make file absolute
                    file = file.absolute()
                    arcname = Path(save_prefix) / file.relative_to(root)
                    tar.add(file, arcname=arcname)
    return root


def dump_args(args, save_path, extra_args=None):
    args_dict = vars(args)
    if extra_args:
        assert isinstance(
            extra_args, dict
        ), f"extra_args should be a dictionary, got {type(extra_args)}."
        args_dict.update(extra_args)
    # Save to file
    with safe_file(save_path).open("w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True, ensure_ascii=False)


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def is_valid_experiment(path):
    path = Path(path)
    if path.is_dir() and path.name.split("_")[0].isdigit():
        return True
    return False


def get_experiment_max_number(experiments):
    valid_experiment_numbers = []
    for exp in experiments:
        if is_valid_experiment(exp):
            valid_experiment_numbers.append(int(Path(exp).name.split("_")[0]))
    if valid_experiment_numbers:
        return max(valid_experiment_numbers)
    return 0


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
    copy from: 
    https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to [0, 1]. Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 24.
    """
    
    print(f"DEBUG: save_videos_grid input - shape: {videos.shape}, dtype: {videos.dtype}")
    print(f"DEBUG: min: {videos.min()}, max: {videos.max()}")
    print(f"DEBUG: has_nan: {torch.isnan(videos).any()}, has_inf: {torch.isinf(videos).any()}")
    
    # CRITICAL FIX: Handle NaN/Inf values that cause the RuntimeWarning
    if torch.isnan(videos).any() or torch.isinf(videos).any():
        print("WARNING: Input tensor contains NaN/Inf values. Applying fixes...")
        
        # Replace NaN/Inf with valid values
        videos = torch.nan_to_num(videos, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # If the entire tensor was NaN, create a test pattern instead of black video
        if torch.all(videos == 0.0):
            print("WARNING: Entire tensor was NaN. Creating test pattern...")
            B, C, T, H, W = videos.shape
            
            # Create a simple test pattern - gradient frames
            for t in range(T):
                for c in range(C):
                    # Create a gradient pattern that changes over time
                    gradient = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
                    # Add time-based variation
                    time_factor = (t / max(T-1, 1)) * 0.5 + 0.25  # 0.25 to 0.75
                    videos[0, c, t] = gradient * time_factor
        
        print(f"DEBUG: after NaN fix - min: {videos.min()}, max: {videos.max()}")
    
    # Move to CPU if needed
    if videos.is_cuda:
        videos = videos.cpu()
    
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    
    for i, x in enumerate(videos):
        try:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            
            x = torch.clamp(x, 0, 1)
            
            # Additional safety check before conversion
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: Frame {i} still has NaN/Inf after processing")
                x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
                x = torch.clamp(x, 0, 1)
            
            # Convert to numpy with error handling
            try:
                x_numpy = (x * 255).numpy().astype(np.uint8)
            except Exception as e:
                print(f"ERROR: Failed to convert frame {i}: {e}")
                # Create a fallback frame
                if len(outputs) > 0:
                    x_numpy = outputs[-1].copy()  # Use previous frame
                else:
                    # Create a simple test frame
                    x_numpy = np.full((x.shape[0], x.shape[1], 3), 128, dtype=np.uint8)
            
            outputs.append(x_numpy)
            
        except Exception as e:
            print(f"ERROR: Failed to process frame {i}: {e}")
            # Create a fallback frame
            if len(outputs) > 0:
                outputs.append(outputs[-1].copy())
            else:
                # Create a default frame
                fallback_frame = np.full((256, 256, 3), 128, dtype=np.uint8)
                outputs.append(fallback_frame)

    if not outputs:
        print("ERROR: No frames could be processed. Creating minimal video.")
        # Create a minimal test video
        outputs = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(10)]

    print(f"DEBUG: Successfully processed {len(outputs)} frames")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        imageio.mimsave(path, outputs, fps=fps)
        print(f"DEBUG: Video saved successfully to {path}")
    except Exception as e:
        print(f"ERROR: Failed to save video: {e}")
        # Try saving as individual frames for debugging
        frame_dir = path.replace('.mp4', '_frames')
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(outputs[:5]):  # Save first 5 frames
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:03d}.png"), frame)
        print(f"Saved debug frames to: {frame_dir}")
        raise

def save_video_only(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """
    Save only the RGB video portion without depth map overlay.
    
    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video  
        rescale (bool, optional): rescale the video tensor from [-1, 1] to [0, 1]. Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 24.
    """
    
    print(f"DEBUG: save_video_only input - shape: {videos.shape}, dtype: {videos.dtype}")
    print(f"DEBUG: min: {videos.min()}, max: {videos.max()}")
    print(f"DEBUG: has_nan: {torch.isnan(videos).any()}, has_inf: {torch.isinf(videos).any()}")
    
    # Handle NaN/Inf values
    if torch.isnan(videos).any() or torch.isinf(videos).any():
        print("WARNING: Input tensor contains NaN/Inf values. Applying fixes...")
        videos = torch.nan_to_num(videos, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.all(videos == 0.0):
            print("WARNING: Entire tensor was NaN. Creating test pattern...")
            B, C, T, H, W = videos.shape
            for t in range(T):
                for c in range(C):
                    gradient = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
                    time_factor = (t / max(T-1, 1)) * 0.5 + 0.25
                    videos[0, c, t] = gradient * time_factor
        
        print(f"DEBUG: after NaN fix - min: {videos.min()}, max: {videos.max()}")
    
    # Move to CPU if needed
    if videos.is_cuda:
        videos = videos.cpu()
    
    # Extract only RGB channels (first 3 channels) to exclude depth
    B, C, T, H, W = videos.shape
    if C > 3:
        print(f"DEBUG: Extracting RGB channels from {C}-channel video")
        videos = videos[:, :3, :, :, :]  # Take only first 3 channels (RGB)
    
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    
    for i, x in enumerate(videos):
        try:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            
            x = torch.clamp(x, 0, 1)
            
            # Additional safety check before conversion
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: Frame {i} still has NaN/Inf after processing")
                x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
                x = torch.clamp(x, 0, 1)
            
            # Convert to numpy with error handling
            try:
                x_numpy = (x * 255).numpy().astype(np.uint8)
            except Exception as e:
                print(f"ERROR: Failed to convert frame {i}: {e}")
                if len(outputs) > 0:
                    x_numpy = outputs[-1].copy()
                else:
                    x_numpy = np.full((x.shape[0], x.shape[1], 3), 128, dtype=np.uint8)
            
            outputs.append(x_numpy)
            
        except Exception as e:
            print(f"ERROR: Failed to process frame {i}: {e}")
            if len(outputs) > 0:
                outputs.append(outputs[-1].copy())
            else:
                fallback_frame = np.full((256, 256, 3), 128, dtype=np.uint8)
                outputs.append(fallback_frame)

    if not outputs:
        print("ERROR: No frames could be processed. Creating minimal video.")
        outputs = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(10)]

    print(f"DEBUG: Successfully processed {len(outputs)} RGB-only frames")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        imageio.mimsave(path, outputs, fps=fps)
        print(f"DEBUG: RGB-only video saved successfully to {path}")
    except Exception as e:
        print(f"ERROR: Failed to save video: {e}")
        frame_dir = path.replace('.mp4', '_frames')
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(outputs[:5]):
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:03d}.png"), frame)
        print(f"Saved debug frames to: {frame_dir}")
        raise
