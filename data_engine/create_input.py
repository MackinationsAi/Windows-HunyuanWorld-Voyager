import numpy as np
from PIL import Image
import torch
import argparse
import os
import json
import imageio
import cv2

try:
    from moge.model.v1 import MoGeModel
except:
    from MoGe.moge.model.v1 import MoGeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./example.png")
    parser.add_argument("--render_output_dir", type=str, default="../demo/example/")
    parser.add_argument("--type", type=str, default="forward",
        choices=["forward", "backward", "left", "right", "turn_left", "turn_right"])
    return parser.parse_args()


def camera_list(
    num_frames=49,
    type="forward",
    Width=512,
    Height=512,
    fx=256,
    fy=256,
    movement_speed=1.0,
    rotation_speed=1.0,
    arc_height=0.0,
    zoom_factor=1.0,
    custom_params=None
):
    """
    Advanced camera trajectory generation with support for combined movements.
    
    Args:
        num_frames: Number of frames in the trajectory
        type: Movement type or combination
        Width, Height: Image dimensions
        fx, fy: Focal lengths
        movement_speed: Speed of translational movement (0.0 to 3.0)
        rotation_speed: Speed of rotational movement (0.0 to 3.0)
        arc_height: Height of arc for curved movements (0.0 to 2.0)
        zoom_factor: Zoom effect intensity (0.5 to 2.0)
        custom_params: Dict with custom trajectory parameters
    
    Supported types:
        Basic: "forward", "backward", "left", "right", "up", "down"
        Rotations: "turn_left", "turn_right", "tilt_up", "tilt_down",
                 "360_rotation_left", "360_rotation_right"
        Combined: "forward_left", "forward_right", "orbit_left", "orbit_right", 
                 "dolly_zoom_in", "dolly_zoom_out", "spiral_up", "spiral_down", "arc_left", "arc_right"
        Advanced: "figure_eight", "sine_wave", "custom"
    """
    
    # Validate inputs
    valid_types = [
        "forward", "backward", "left", "right", "up", "down", "turn_left", "turn_right", 
        "tilt_up", "tilt_down", "360_rotation_left", "360_rotation_right", 
        "forward_left", "forward_right", "forward_turn_left", "forward_turn_right", "orbit_left", 
        "orbit_right", "dolly_zoom_in", "dolly_zoom_out", "spiral_up", "spiral_down", "arc_left", "arc_right", 
        "figure_eight", "sine_wave", "custom"
    ]
    
    if type not in valid_types:
        raise ValueError(f"Invalid camera type. Must be one of: {valid_types}")
    
    # Setup intrinsics
    cx = Width // 2
    cy = Height // 2
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsic] * num_frames)
    
    # Time parameter for smooth interpolation
    t = np.linspace(0, 1, num_frames)
    
    # Initialize positions and orientations
    positions = np.zeros((num_frames, 3))
    look_at_points = np.zeros((num_frames, 3))
    up_vectors = np.tile([0, 1, 0], (num_frames, 1))
    
    # Generate trajectories based on type
    if type == "forward":
        positions[:, 2] = t * movement_speed
        look_at_points[:, 2] = positions[:, 2] + 1
        
    elif type == "backward":
        positions[:, 2] = -t * movement_speed
        look_at_points[:, 2] = positions[:, 2] - 1
        
    elif type == "left":
        positions[:, 0] = -t * movement_speed
        look_at_points[:, 2] = 1
        
    elif type == "right":
        positions[:, 0] = t * movement_speed
        look_at_points[:, 2] = 1
        
    elif type == "up":
        positions[:, 1] = t * movement_speed
        look_at_points[:, 2] = 1
        
    elif type == "down":
        positions[:, 1] = -t * movement_speed
        look_at_points[:, 2] = 1
        
    elif type == "turn_left":
        # Pure rotation around Y axis
        angles = t * np.pi * (rotation_speed * 0.75)
        look_at_points[:, 0] = -np.sin(angles)
        look_at_points[:, 2] = np.cos(angles)
        
    elif type == "turn_right":
        angles = t * np.pi * (rotation_speed * 0.75)
        look_at_points[:, 0] = np.sin(angles)
        look_at_points[:, 2] = np.cos(angles)

    elif type == "tilt_up":
        angles = t * np.pi * 0.25 * (rotation_speed * 0.5)
        look_at_points[:, 1] = -np.sin(angles)
        look_at_points[:, 2] = np.cos(angles)

    elif type == "tilt_down":
        angles = t * np.pi * 0.25 * (rotation_speed * 0.5)
        look_at_points[:, 1] = np.sin(angles)
        look_at_points[:, 2] = np.cos(angles)

    elif type == "360_rotation_left":
        # Camera spins counterclockwise while moving forward
        positions[:, 2] = t * movement_speed * 0.5  # Move forward slowly
        # Rotate the look-at direction around the camera position
        angles = t * 2 * np.pi * (rotation_speed * 0.5)  # Full 360 degree spin
        look_at_points[:, 0] = positions[:, 0] - np.sin(angles)
        look_at_points[:, 2] = positions[:, 2] + np.cos(angles)
        # Keep up vector normal
        up_vectors[:, :] = [0, 1, 0]
        
    elif type == "360_rotation_right":
        # Camera spins clockwise while moving forward
        positions[:, 2] = t * movement_speed * 0.5  # Move forward slowly
        # Rotate the look-at direction around the camera position
        angles = t * 2 * np.pi * (rotation_speed * 0.5)  # Full 360 degree spin
        look_at_points[:, 0] = positions[:, 0] + np.sin(angles)
        look_at_points[:, 2] = positions[:, 2] + np.cos(angles)
        # Keep up vector normal
        up_vectors[:, :] = [0, 1, 0]

    elif type == "forward_left":
        # Forward movement with leftward drift
        positions[:, 2] = t * movement_speed
        positions[:, 0] = -t * movement_speed
        look_at_points[:, 2] = positions[:, 2] + 2
        look_at_points[:, 0] = positions[:, 0]
        
    elif type == "forward_right":
        positions[:, 2] = t * movement_speed
        positions[:, 0] = t * movement_speed
        look_at_points[:, 2] = positions[:, 2] + 2
        look_at_points[:, 0] = positions[:, 0]
        
    elif type == "forward_turn_left":
        positions[:, 2] = t * movement_speed
        angles = t * np.pi * 0.5 * rotation_speed  # 90 degree turn over sequence
        look_at_points[:, 0] = positions[:, 0] - np.sin(angles)
        look_at_points[:, 2] = positions[:, 2] + np.cos(angles)
        
    elif type == "forward_turn_right":
        positions[:, 2] = t * movement_speed
        angles = t * np.pi * 0.5 * rotation_speed
        look_at_points[:, 0] = positions[:, 0] + np.sin(angles)
        look_at_points[:, 2] = positions[:, 2] + np.cos(angles)
        
    elif type == "orbit_left":
        radius = 2.0 * movement_speed
        angles = t * 2 * np.pi * rotation_speed
        positions[:, 0] = radius * np.cos(angles)
        positions[:, 2] = radius * np.sin(angles)
        look_at_points[:, :] = [0, 0, 0]  # Always look at center
        
    elif type == "orbit_right":
        radius = 2.0 * movement_speed
        angles = -t * 2 * np.pi * rotation_speed
        positions[:, 0] = radius * np.cos(angles)
        positions[:, 2] = radius * np.sin(angles)
        look_at_points[:, :] = [0, 0, 0]
        
    elif type == "dolly_zoom_in":
        positions[:, 2] = t * movement_speed * 2
        look_at_points[:, 2] = 1
        
    elif type == "dolly_zoom_out":
        positions[:, 2] = -t * movement_speed * 2
        look_at_points[:, 2] = 1
        
    elif type == "spiral_up":
        radius = 2.0 * movement_speed
        angles = t * 4 * np.pi * rotation_speed
        positions[:, 0] = radius * np.cos(angles) * (1 - t * 0.5)
        positions[:, 2] = radius * np.sin(angles) * (1 - t * 0.5)
        positions[:, 1] = t * movement_speed * 2
        look_at_points[:, :] = [0, 0, 0]
        
    elif type == "spiral_down":
        radius = 2.0 * movement_speed
        angles = t * 4 * np.pi * rotation_speed
        positions[:, 0] = radius * np.cos(angles) * (1 - t * 0.5)
        positions[:, 2] = radius * np.sin(angles) * (1 - t * 0.5)
        positions[:, 1] = -t * movement_speed * 2
        look_at_points[:, :] = [0, 0, 0]
        
    elif type == "arc_left":
        positions[:, 0] = -t * movement_speed
        positions[:, 1] = arc_height * np.sin(t * np.pi)  # Arc shape
        positions[:, 2] = t * movement_speed * 0.5
        look_at_points[:, 2] = positions[:, 2] + 1
        
    elif type == "arc_right":
        positions[:, 0] = t * movement_speed
        positions[:, 1] = arc_height * np.sin(t * np.pi)
        positions[:, 2] = t * movement_speed * 0.5
        look_at_points[:, 2] = positions[:, 2] + 1
        
    elif type == "figure_eight":
        angles = t * 4 * np.pi * rotation_speed
        positions[:, 0] = movement_speed * np.sin(angles)
        positions[:, 2] = movement_speed * np.sin(angles * 2) * 0.5 + 2.0
        positions[:, 1] = movement_speed * np.cos(angles * 2) * 0.3
        look_at_points[:, :] = [0, 0, 0]
        
    elif type == "sine_wave":
        # Sinusoidal path
        positions[:, 0] = movement_speed * np.sin(t * 2 * np.pi * rotation_speed)
        positions[:, 2] = t * movement_speed
        positions[:, 1] = arc_height * np.sin(t * 4 * np.pi)
        look_at_points[:, 2] = positions[:, 2] + 1
        
    elif type == "custom" and custom_params:
        # Custom trajectory from parameters
        if "positions" in custom_params:
            positions = np.array(custom_params["positions"])
        if "look_at_points" in custom_params:
            look_at_points = np.array(custom_params["look_at_points"])
        if "up_vectors" in custom_params:
            up_vectors = np.array(custom_params["up_vectors"])
    
    else:
        # Default fallback
        positions[:, 2] = t * movement_speed
        look_at_points[:, 2] = positions[:, 2] + 1
    
    # Convert to extrinsic matrices
    extrinsics = []
    for i in range(num_frames):
        pos = positions[i]
        target = look_at_points[i]
        up = up_vectors[i]

        # Compute camera coordinate system
        z = target - pos  # Forward direction
        z = z / (np.linalg.norm(z) + 1e-8)

        x = np.cross(up, z)  # Right direction
        x = x / (np.linalg.norm(x) + 1e-8)

        y = np.cross(z, x)  # Up direction
        y = y / (np.linalg.norm(y) + 1e-8)

        # Build rotation matrix and extrinsic matrix
        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ pos

        extrinsics.append(w2c)

    extrinsics = np.stack(extrinsics)
    return intrinsics, extrinsics

# Helper function to create smooth interpolated trajectories
def create_custom_trajectory(keyframes, num_frames, interpolation="cubic"):
    """
    Create custom trajectory from keyframes.
    
    Args:
        keyframes: List of (frame_index, position, look_at, up) tuples
        num_frames: Total number of frames
        interpolation: "linear", "cubic", or "bezier"
    
    Returns:
        positions, look_at_points, up_vectors arrays
    """
    from scipy.interpolate import interp1d
    
    # Extract keyframe data
    kf_indices = [kf[0] for kf in keyframes]
    kf_positions = np.array([kf[1] for kf in keyframes])
    kf_look_ats = np.array([kf[2] for kf in keyframes])
    kf_ups = np.array([kf[3] for kf in keyframes])
    
    # Create interpolation functions
    frame_indices = np.linspace(0, num_frames-1, num_frames)
    
    if interpolation == "cubic":
        kind = "cubic"
    elif interpolation == "linear":
        kind = "linear"
    else:
        kind = "linear"  # fallback
    
    # Interpolate each component
    pos_interp = interp1d(kf_indices, kf_positions, axis=0, kind=kind, 
                         bounds_error=False, fill_value="extrapolate")
    look_interp = interp1d(kf_indices, kf_look_ats, axis=0, kind=kind,
                          bounds_error=False, fill_value="extrapolate")
    up_interp = interp1d(kf_indices, kf_ups, axis=0, kind=kind,
                        bounds_error=False, fill_value="extrapolate")
    
    positions = pos_interp(frame_indices)
    look_at_points = look_interp(frame_indices)
    up_vectors = up_interp(frame_indices)
    
    return positions, look_at_points, up_vectors


# from VGGT: https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points


def render_from_cameras_videos(points, colors, extrinsics, intrinsics, height, width):
    
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    render_list = []
    mask_list = []
    depth_list = []
    # Render from each camera
    for frame_idx in range(len(extrinsics)):
        # Get corresponding camera parameters
        extrinsic = extrinsics[frame_idx]
        intrinsic = intrinsics[frame_idx]
        
        camera_coords = (extrinsic @ homogeneous_points.T).T[:, :3]
        projected = (intrinsic @ camera_coords.T).T
        uv = projected[:, :2] / projected[:, 2].reshape(-1, 1)
        depths = projected[:, 2]    
        
        pixel_coords = np.round(uv).astype(int)  # pixel_coords (h*w, 2)      
        valid_pixels = (  # valid_pixels (h*w, )      valid_pixels is the valid pixels in width and height
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < width) & 
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < height)
        )
        
        pixel_coords_valid = pixel_coords[valid_pixels]  # (h*w, 2) to (valid_count, 2)
        colors_valid = colors[valid_pixels]
        depths_valid = depths[valid_pixels]
        uv_valid = uv[valid_pixels]
        
        
        valid_mask = (depths_valid > 0) & (depths_valid < 60000) # & normal_angle_mask
        colors_valid = colors_valid[valid_mask]
        depths_valid = depths_valid[valid_mask]
        pixel_coords_valid = pixel_coords_valid[valid_mask]

        # Initialize depth buffer
        depth_buffer = np.full((height, width), np.inf)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized depth buffer update
        if len(pixel_coords_valid) > 0:
            rows = pixel_coords_valid[:, 1]
            cols = pixel_coords_valid[:, 0]
                            
            # Sort by depth (near to far)
            sorted_idx = np.argsort(depths_valid)
            rows = rows[sorted_idx]
            cols = cols[sorted_idx]
            depths_sorted = depths_valid[sorted_idx]
            colors_sorted = colors_valid[sorted_idx]

            # Vectorized depth buffer update
            depth_buffer[rows, cols] = np.minimum(
                depth_buffer[rows, cols], 
                depths_sorted
            )
            
            # Get the minimum depth index for each pixel
            flat_indices = rows * width + cols  # Flatten 2D coordinates to 1D index
            unique_indices, idx = np.unique(flat_indices, return_index=True)
            
            # Recover 2D coordinates from flattened indices
            final_rows = unique_indices // width
            final_cols = unique_indices % width
            
            image[final_rows, final_cols] = colors_sorted[idx, :3].astype(np.uint8)

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255
        
        render_list.append(image)
        mask_list.append(mask)
        depth_list.append(depth_buffer)
    
    return render_list, mask_list, depth_list


def create_video_input(
    render_list, mask_list, depth_list, render_output_dir,
    separate=True, ref_image=None, ref_depth=None,
    Width=512, Height=512,
    min_percentile=2, max_percentile=98
):
    video_output_dir = os.path.join(render_output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    video_input_dir = os.path.join(render_output_dir, "video_input")
    os.makedirs(video_input_dir, exist_ok=True)

    value_list = []
    for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):

        # Sky part is the region where depth_max is, also included in mask
        mask = mask > 0
        
        # CRITICAL: Convert to inverse depth BEFORE normalization
        # This is what was happening in the original .exr pipeline
        depth[mask] = 1 / (depth[mask] + 1e-6)
        depth_values = depth[mask]
        
        # Compute the percentile ranges for THIS specific video sequence
        min_percentile_val = np.percentile(depth_values, min_percentile)
        max_percentile_val = np.percentile(depth_values, max_percentile)
        value_list.append([min_percentile_val, max_percentile_val])

        # Normalize using the computed ranges
        depth[mask] = (depth[mask] - min_percentile_val) / (max_percentile_val - min_percentile_val)
        depth[~mask] = depth[mask].min()

        # resize to target dimensions
        render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), \
            (Width, Height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)

        # Save mask as png
        mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
        imageio.imwrite(mask_path, mask)
        
        if separate:
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)
            depth_path = os.path.join(video_input_dir, f"depth_{i:04d}.npy")
            np.save(depth_path, depth)
        else:
            render = np.concatenate([render, depth], axis=-3)
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)

        if i == 0:
            if separate:
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)
                ref_depth_path = os.path.join(video_output_dir, f"ref_depth.npy")
                np.save(ref_depth_path, depth)
            else:
                ref_image = np.concatenate([ref_image, depth], axis=-3)
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)

    # CRITICAL: Save the depth ranges for the inference pipeline to use
    with open(os.path.join(video_output_dir, f"depth_range.json"), "w") as f:
        json.dump(value_list, f)


if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl", local_files_only=False).to(device)

    image = np.array(Image.open(args.image_path).convert("RGB").resize((1280, 720)))
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    output = model.infer(image_tensor)
    depth = np.array(output['depth'].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
    
    Height, Width = image.shape[:2]
    intrinsics, extrinsics = camera_list(
        num_frames=1, type=args.type, Width=Width, Height=Height, fx=256, fy=256
    )

    # Backproject point cloud
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    intrinsics, extrinsics = camera_list(
        num_frames=49, type=args.type, Width=Width//2, Height=Height//2, fx=128, fy=128
    )
    render_list, mask_list, depth_list = render_from_cameras_videos(
        points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
    )
    
    create_video_input(
        render_list, mask_list, depth_list, args.render_output_dir, separate=True, 
        ref_image=image, ref_depth=depth, Width=Width, Height=Height)
