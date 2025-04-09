import runpod
import logging
import sys
from datetime import datetime
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video


def generate_video(prompt):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )

    task = "i2v-14B"
    size = "832*480"
    ckpt_dir = "/runpod-volume/Wan2.1-I2V-14B-480P"
    image_path = "examples/i2v_input.JPG"

    cfg = WAN_CONFIGS[task]
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Loading image from: {image_path}")

    img = Image.open(image_path).convert("RGB")

    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    logging.info("Generating video...")
    video = wan_i2v.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=81,
        shift=3.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        seed=42,
        offload_model=True
    )

    save_path = f"/runpod-volume/output/generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.info(f"Saving video to {save_path}")
    cache_video(
        tensor=video[None],
        save_file=save_path,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    logging.info("Finished.")
    return save_path
 # assuming you saved it as a separate module

def handler(event):
    print("Worker Start")
    input_data = event['input']

    prompt = input_data.get('prompt')
    print(f"Received prompt: {prompt}")

    try:
        output_path = generate_video(prompt)
        return {
            "status": "success",
            "output_path": output_path
        }
    except Exception as e:
        print("Generation failed:", str(e))
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
