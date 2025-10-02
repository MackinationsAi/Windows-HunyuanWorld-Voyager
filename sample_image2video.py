import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from voyager.utils.file_utils import save_videos_grid, save_video_only
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler

def main():
    args = parse_args()
    
    # Set model_base to use local ckpts folder
    args.model_base = os.path.abspath("ckpts")
    
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        i2v_mode=args.i2v_mode,
        i2v_resolution=args.i2v_resolution,
        i2v_image_path=args.i2v_image_path,
        i2v_condition_type=args.i2v_condition_type,
        i2v_stability=args.i2v_stability,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        ref_images=[(os.path.join(args.input_path, "ref_image.png"),
                     os.path.join(args.input_path, "ref_depth.npy"))],
        partial_cond=[(os.path.join(args.input_path, "video_input", f"render_{j:04d}.png"), os.path.join(
            args.input_path, "video_input", f"depth_{j:04d}.npy")) for j in range(49)],
        partial_mask=[(os.path.join(args.input_path, "video_input", f"mask_{j:04d}.png"), os.path.join(
            args.input_path, "video_input", f"mask_{j:04d}.png")) for j in range(49)]
    )
    samples = outputs['samples']

    # Save generated videos to disk
    # Only save on the main process in distributed settings
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H_%M_%S")
            
            # Clean prompt for Windows filename
            clean_prompt = outputs['prompts'][i][:100]
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                clean_prompt = clean_prompt.replace(char, '')
            
            cur_save_path = f"{save_path}{os.sep}{time_flag}_seed{outputs['seeds'][i]}_{clean_prompt}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')

            clean_save_path = f"{save_path}{os.sep}{time_flag}_seed{outputs['seeds'][i]}_{clean_prompt}_clean.mp4"
            save_video_only(sample, clean_save_path, fps=24)

if __name__ == "__main__":
    main()
