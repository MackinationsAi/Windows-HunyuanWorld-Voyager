import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image
import uuid
import warnings
# import struct

# Suppress warnings that can cause crashes
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Set to your GPU architecture
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from voyager.utils.file_utils import save_videos_grid, save_video_only
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler

from moge.model.v1 import MoGeModel
from data_engine.create_input import camera_list, depth_to_world_coords_points, render_from_cameras_videos, create_video_input

def make_divisible_by_16(value):
    """Ensure dimension is divisible by 16 for video compatibility"""
    return ((value // 16) * 16)

def load_models(args):
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    model = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    return model


def generate_video(temp_path, prompt):
    condition_path = temp_path
    output_path = os.path.join(os.path.dirname(condition_path), "output")
    
    # No need to convert here - conversion already happened during condition generation
    
    # Use local models from ckpts folder with proper Windows paths
    ckpts_path = os.path.abspath("ckpts")
    
    # Convert all paths to Windows format and escape them properly
    condition_path_win = os.path.normpath(condition_path)
    output_path_win = os.path.normpath(output_path)
    ckpts_path_win = os.path.normpath(ckpts_path)
    
    cmd = f'''python sample_image2video.py --model HYVideo-T/2 --model-base "{ckpts_path_win}" --input-path "{condition_path_win}" --prompt "{prompt}" --i2v-stability --infer-steps 50 --flow-reverse --flow-shift 7.0 --seed 0 --embedded-cfg-scale 6.0 --use-cpu-offload --save-path "{output_path_win}"'''
    
    print(f"Executing command: {cmd}")
    os.system(cmd)
    
    if os.path.exists(output_path) and os.listdir(output_path):
        video_name = os.listdir(output_path)[0]
        return os.path.join(output_path, video_name)
    else:
        print(f"No video generated in {output_path}")
        return None


def create_condition(model, image_path, direction, save_path, 
                    movement_speed=1.0, rotation_speed=1.0, arc_height=0.0):
    try:
        print(f"Starting condition generation for {image_path}")
        
        # Load and process image with dimensions divisible by 16
        original_image = Image.open(image_path)
        original_width, original_height = original_image.size
        
        # Make dimensions divisible by 16 for video compatibility
        target_width = make_divisible_by_16(1280)
        target_height = make_divisible_by_16(720)
        
        image = np.array(original_image.resize((target_width, target_height)))
        print(f"Image loaded and resized: {image.shape} (was {original_width}x{original_height})")
        
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device="cuda:0").permute(2, 0, 1)    
        print("Running depth estimation...")
        
        output = model.infer(image_tensor)
        depth = np.array(output['depth'].detach().cpu())
        
        # Clean up depth values
        print(f"Depth range before cleanup: {depth.min()} to {depth.max()}")
        depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
        depth[np.isnan(depth)] = depth[~np.isnan(depth)].mean()
        print(f"Depth range after cleanup: {depth.min()} to {depth.max()}")
        
        Height, Width = image.shape[:2]
        print(f"Image dimensions: {Height}x{Width}")

        # Generate camera parameters
        print("Generating camera parameters...")
        intrinsics, extrinsics = camera_list(
            num_frames=1, type=direction, Width=Width, Height=Height, fx=256, fy=256
        )

        # Backproject point cloud with error handling
        print("Backprojecting point cloud...")
        try:
            point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
            
            # Clean up any invalid coordinates before proceeding
            print("Cleaning up point cloud data...")
            print(f"Point map shape: {point_map.shape}")
            
            # point_map should be (H, W, 3), so check along the last axis
            valid_mask = np.isfinite(point_map).all(axis=-1)  # This gives us (H, W) boolean mask
            print(f"Valid mask shape: {valid_mask.shape}")
            print(f"Valid points: {valid_mask.sum()} out of {valid_mask.size}")
            
            if valid_mask.sum() == 0:
                raise ValueError("No valid points in point cloud after depth processing")
            
            # Reshape to match the flattened arrays
            valid_mask_flat = valid_mask.flatten()  # Convert (H, W) to (H*W,)
            
            # Filter points and colors to only include valid ones
            points = point_map.reshape(-1, 3)[valid_mask_flat]  # Apply flat mask to reshaped points
            colors = image.reshape(-1, 3)[valid_mask_flat]      # Apply flat mask to reshaped colors
            print(f"Filtered point cloud generated: {points.shape}")
            
        except Exception as e:
            print(f"Error in point cloud generation: {e}")
            raise
        
        # Generate video frames (configurable frame count)
        print("Generating video frames...")
        frame_count = 49  # Changed to 49 frames - WARNING: This significantly increases processing time
        
        # Make render dimensions divisible by 16 to avoid FFMPEG warnings
        render_width = make_divisible_by_16(Width // 2)
        render_height = make_divisible_by_16(Height // 2)
        
        print(f"Render dimensions: {render_width}x{render_height} (16-divisible)")
        
        intrinsics, extrinsics = camera_list(
            num_frames=frame_count, type=direction, 
            Width=render_width, Height=render_height, fx=128, fy=128,
            movement_speed=movement_speed, rotation_speed=rotation_speed, arc_height=arc_height
        )
        
        try:
            render_list, mask_list, depth_list = render_from_cameras_videos(
                points, colors, extrinsics, intrinsics, 
                height=render_height, width=render_width
            )
            print(f"Rendered {len(render_list)} frames at {render_width}x{render_height}")
        except Exception as e:
            print(f"Error in video rendering: {e}")
            raise
        
        # Create video input files
        print("Creating video input files... (this may take several minutes)")
        condition_path = os.path.join(save_path, "condition")
        try:
            # Add comprehensive error handling and progress tracking
            print(f"Render list length: {len(render_list)}")
            print(f"Condition path: {condition_path}")
            print(f"Image shape: {image.shape}, Depth shape: {depth.shape}")
            
            # Create the directory structure first
            os.makedirs(condition_path, exist_ok=True)
            video_input_dir = os.path.join(condition_path, "video_input")
            os.makedirs(video_input_dir, exist_ok=True)
            
            print("Processing frames individually...")
            
            value_list = []  # Initialize depth range list
            
            for i, (render, mask, depth_buffer) in enumerate(zip(render_list, mask_list, depth_list)):
                try:
                    print(f"Processing frame {i+1}/{len(render_list)}")
                    
                    # Save render frame
                    render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
                    render_clean = np.clip(render, 0, 255).astype(np.uint8)
                    imageio.imwrite(render_path, render_clean)
                    
                    # Save mask frame
                    mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
                    mask_clean = np.clip(mask, 0, 255).astype(np.uint8)
                    imageio.imwrite(mask_path, mask_clean)
                    
                    # Process depth with range computation
                    depth_path = os.path.join(video_input_dir, f"depth_{i:04d}.npy")
                    depth_clean = depth_buffer.copy().astype(np.float32)
                    
                    # Create mask for valid depth values
                    mask_bool = mask_clean > 0
                    
                    # Handle infinite values
                    inf_count = np.sum(np.isinf(depth_clean))
                    if inf_count > 0:
                        finite_values = depth_clean[np.isfinite(depth_clean)]
                        if len(finite_values) > 0:
                            replacement_value = finite_values.max()
                        else:
                            replacement_value = 1000.0
                        depth_clean[np.isinf(depth_clean)] = replacement_value
                    
                    # Convert to inverse depth and compute ranges (matching original pipeline)
                    depth_clean[mask_bool] = 1.0 / (depth_clean[mask_bool] + 1e-6)
                    depth_values = depth_clean[mask_bool]
                    
                    if len(depth_values) > 0:
                        # Compute percentile ranges for this frame
                        min_percentile_val = np.percentile(depth_values, 2)
                        max_percentile_val = np.percentile(depth_values, 98)
                        value_list.append([min_percentile_val, max_percentile_val])
                        
                        # Normalize depth using computed ranges
                        depth_clean[mask_bool] = (depth_clean[mask_bool] - min_percentile_val) / (max_percentile_val - min_percentile_val)
                        depth_clean[~mask_bool] = depth_clean[mask_bool].min() if np.any(mask_bool) else 0.0
                    else:
                        # Fallback if no valid depth values
                        value_list.append([0.004, 12.0])  # Default range
                        depth_clean[:] = 0.5  # Neutral value
                    
                    # Ensure depth is in [0, 1] range
                    depth_clean = np.clip(depth_clean, 0.0, 1.0)
                    
                    np.save(depth_path, depth_clean)

                    if i % 15 == 0:  # Progress update every 15 frames (was 10, adjusted for 49 frames)
                        print(f"Completed {i+1}/{len(render_list)} frames")
                        
                except Exception as frame_error:
                    print(f"Error processing frame {i}: {frame_error}")
                    import traceback
                    traceback.print_exc()
                    # Add fallback range for failed frames
                    value_list.append([0.004, 12.0])
                    continue
            
            print(f"All {len(render_list)} frames processed successfully")
            
            # Save the depth_range.json file
            print("Saving depth_range.json...")
            try:
                depth_range_path = os.path.join(condition_path, "depth_range.json")
                import json
                with open(depth_range_path, 'w') as f:
                    json.dump(value_list, f)
                print(f"Depth range saved to: {depth_range_path}")
                print(f"Sample ranges: {value_list[:3]}...")  # Show first 3 ranges for debugging
            except Exception as range_error:
                print(f"Error saving depth_range.json: {range_error}")
                import traceback
                traceback.print_exc()
            
            # Save reference image and depth
            print("Saving reference files...")
            try:
                ref_image_path = os.path.join(condition_path, "ref_image.png")
                imageio.imwrite(ref_image_path, image)
                print("Reference image saved")
                
                ref_depth_path = os.path.join(condition_path, "ref_depth.npy")
                depth_clean = depth.copy().astype(np.float32)
                depth_clean[np.isinf(depth_clean)] = depth_clean[~np.isinf(depth_clean)].max() if np.any(~np.isinf(depth_clean)) else 1000.0
                np.save(ref_depth_path, depth_clean)
                print("Reference depth saved")
                
            except Exception as ref_error:
                print(f"Error saving reference files: {ref_error}")
                import traceback
                traceback.print_exc()
            
            print("Video input files created successfully")
            
        except Exception as e:
            print(f"Error creating video input: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Create condition video with FFMPEG fix
        print("Creating condition video...")
        image_list = []
        video_input_dir = os.path.join(save_path, "condition", "video_input")
        
        if not os.path.exists(video_input_dir):
            raise FileNotFoundError(f"Video input directory not found: {video_input_dir}")
            
        for i in range(frame_count):  # Use the same frame_count as above
            render_file = os.path.join(video_input_dir, f"render_{i:04d}.png")
            if os.path.exists(render_file):
                image_list.append(np.array(Image.open(render_file)))
            else:
                print(f"Warning: Missing render file {render_file}")
        
        if not image_list:
            raise FileNotFoundError("No render files found")
            
        condition_video_path = os.path.join(save_path, "condition.mp4")
        # Apply FFMPEG fix: use macro_block_size=1 to prevent resizing warnings
        imageio.mimsave(condition_video_path, image_list, fps=8, macro_block_size=1)
        print(f"Condition video saved: {condition_video_path}")
            
        return condition_video_path
        
    except Exception as e:
        print(f"Error in create_condition: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_uploaded_image(image, save_dir="temp_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    image_path = os.path.join(save_dir, "input_image.png")
    pil_image.save(image_path)
    return image_path


def create_video_demo():
    # Load MoGe model - use local_files_only to prevent downloading
    try:
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl", local_files_only=True).to("cuda:0")
    except:
        # If local model not found, download it once
        print("Downloading MoGe model (one-time setup)...")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda:0")

    def process_condition_generation(image, direction, mov_speed, rot_speed, arc_h):
        temp_path = os.path.join("temp", uuid.uuid4().hex[:8])
        image_path = save_uploaded_image(image, temp_path)
        assert image_path is not None, "Please upload image"
        condition_video_path = create_condition(
            moge_model, image_path, direction, temp_path,
            movement_speed=mov_speed, rotation_speed=rot_speed, arc_height=arc_h
        )
        return os.path.join(temp_path, "condition"), condition_video_path
    
    def process_video_generation(temp_path, prompt):
        if temp_path is None or prompt is None:
            return None
        
        final_video_path = generate_video(temp_path, prompt)
        
        return final_video_path
    
    with gr.Blocks(title="Voyager Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Windows-HunyuanWorld-Voyager")
        gr.Markdown("Upload an image, input description text, select movement direction, and generate exciting videos!")

        temp_path = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                direction_choice = gr.Dropdown(
                    choices=[
                        # Basic movements
                        "forward", "backward", "left", "right", "up", "down",
                        
                        # Pure rotations
                        "turn_left", "turn_right", "tilt_up", "tilt_down", 
                        
                        # Combined movements (most cinematic)
                        "forward_left", "forward_right", 
                        "forward_turn_left", "forward_turn_right",  # These create the zoom+pan effect
                        
                        # Advanced movements
                        "orbit_left", "orbit_right",
                        "dolly_zoom_in", "dolly_zoom_out",  # Classic Vertigo effect
                        "spiral_up", "spiral_down",
                        "arc_left", "arc_right",
                        
                        # Complex patterns
                        "figure_eight", "sine_wave"
                    ],
                    label="Choose Camera Movement",
                    value="forward_turn_left"  # Set this as default to match your example
                )

                with gr.Row():
                    movement_speed = gr.Slider(
                        minimum=0.1, maximum=3.0, value=1.0, step=0.1,
                        label="Movement Speed"
                    )
                    rotation_speed = gr.Slider(
                        minimum=0.1, maximum=3.0, value=1.0, step=0.1,
                        label="Rotation Speed"
                    )
                    arc_height = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                        label="Arc [curve path]"
                    )

                condition_video_output = gr.Video(
                    label="Condition Video",
                    height=300
                )
                
                condition_btn = gr.Button(
                    "Generate Condition",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                input_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Please input video description",
                    lines=5
                )
                
                gr.Markdown("### Generating Final Video")
                final_video_output = gr.Video(
                    label="Generated Video", 
                    height=600
                )

                generate_btn = gr.Button(
                    "Generate Video",
                    variant="primary",
                    size="lg"
                )
        
        examples = []
        for i in range(1, 11):
            example_dir = os.path.join("examples", f"case{i}")
            if os.path.exists(example_dir):
                items = [
                    os.path.join(example_dir, "ref_image.png"), 
                    os.path.join(example_dir, "condition.mp4")
                ]
                prompt_file = os.path.join(example_dir, "prompt.txt")
                if os.path.exists(prompt_file):
                    with open(prompt_file, "r") as f:
                        prompt = f.readline().strip()
                        items.append(prompt)
                items.append(example_dir)
                examples.append(items)

        def update_state(hidden_input):
            return str(hidden_input)

        hidden_input = gr.Textbox(visible=False)
        
        if examples:
            with gr.Accordion("Example Gallery", open=False):
                gr.Examples(
                    examples=examples,
                    inputs=[input_image, condition_video_output, input_prompt, hidden_input],
                    outputs=[temp_path]
                )

        hidden_input.change(fn=update_state, inputs=hidden_input, outputs=temp_path)
        
        condition_btn.click(
            fn=process_condition_generation,
            inputs=[input_image, direction_choice, movement_speed, rotation_speed, arc_height],
            outputs=[temp_path, condition_video_output]
        )
        generate_btn.click(
            fn=process_video_generation,
            inputs=[temp_path, input_prompt],
            outputs=[final_video_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_video_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
