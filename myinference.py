import os, gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, Blip2ForConditionalGeneration, T5EncoderModel
from diffusers import (
    AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
    DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler, PNDMScheduler
)

from models.infer import DepthCrafterDemo
from models.utils import Warper, read_video_frames, save_video
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from models.crosstransformer3d import CrossTransformer3DModel
from models.autoencoder_magvit import AutoencoderKLCogVideoX
from scipy.spatial.transform import Rotation as R, Slerp


# ===== 配置参数 =====
class SimpleOpts:
    device = "cuda"
    unet_path = "tencent/DepthCrafter"
    pre_train_path = "stabilityai/stable-video-diffusion-img2vid-xt"
    cpu_offload = "model"
    near, far = 0.0001, 10000.0
    depth_inference_steps = 5
    depth_guidance_scale = 1.0
    window_size, overlap, max_res = 110, 25, 1024
    mask = False
    sample_size = (384, 672)
    fps, video_length, stride = 10, 49, 2
    sampler_name = "DDIM_Origin"
    cut = 20
    seed = 42
    model_name = "alibaba-pai/CogVideoX-Fun-V1.1-5b-InP"
    transformer_path = "TrajectoryCrafter/TrajectoryCrafter"
    blip_path = "Salesforce/blip2-opt-2.7b"
    save_dir = "/home/tione/notebook/TrajectoryCrafter/data/mytest"
    refine_prompt = ""
    negative_prompt = ""
    diffusion_guidance_scale = 7.5
    diffusion_inference_steps = 50
    weight_dtype = torch.float16

opts = SimpleOpts()

# ===== 工具函数 =====
def get_caption(opts, image):
    image_array = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_array)
    inputs = caption_processor(images=pil_image, return_tensors="pt").to(opts.device, torch.float16)
    generated_ids = captioner.generate(**inputs)
    text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text + opts.refine_prompt

def load_c2w(path):
    extr = np.load(path)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :] = extr
    return mat

def interpolate_camera_poses(pose_s, pose_t, num_steps):
    """
    插值从 pose_s 到 pose_t 的相机位姿序列（SE(3)）
    输入: pose_s, pose_t 为 [4, 4] numpy 或 torch 张量
    返回: list[np.ndarray]，每个是 [4, 4]
    """
    # 转 numpy
    if isinstance(pose_s, torch.Tensor):
        pose_s = pose_s.cpu().numpy()
    if isinstance(pose_t, torch.Tensor):
        pose_t = pose_t.cpu().numpy()

    R_s, t_s = pose_s[:3, :3], pose_s[:3, 3]
    R_t, t_t = pose_t[:3, :3], pose_t[:3, 3]

    # 转四元数
    rot_s = R.from_matrix(R_s)
    rot_t = R.from_matrix(R_t)

    key_times = [0, 1]
    key_rots = R.from_quat([rot_s.as_quat(), rot_t.as_quat()])
    slerp = Slerp(key_times, key_rots)

    interpolated_poses = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        R_interp = slerp([alpha])[0].as_matrix()
        t_interp = (1 - alpha) * t_s + alpha * t_t

        pose_interp = np.eye(4, dtype=np.float32)
        pose_interp[:3, :3] = R_interp
        pose_interp[:3, 3] = t_interp
        interpolated_poses.append(pose_interp)

    return interpolated_poses

# ===== 1. 加载视频与相机参数 =====
video_path = "/home/tione/notebook/TrajectoryCrafter/data/mytest/video.mp4"
intrinsics_path = "/home/tione/notebook/TrajectoryCrafter/data/mytest/intrinsics.npy"
src_extrinsics_path = "/home/tione/notebook/TrajectoryCrafter/data/mytest/extrinsics1.npy"
tgt_extrinsics_path = "/home/tione/notebook/TrajectoryCrafter/data/mytest/extrinsics2.npy"

frames = read_video_frames(video_path, opts.video_length, opts.stride, opts.max_res)

K = torch.tensor(np.load(intrinsics_path)).float().unsqueeze(0).repeat(opts.cut, 1, 1)
pose_s_np = load_c2w(src_extrinsics_path)  # shape [3, 4] → load_c2w 自动转成 [4, 4]
pose_t_np = load_c2w(tgt_extrinsics_path)
pose_seq_np = interpolate_camera_poses(pose_s_np, pose_t_np, opts.cut)
pose_s = torch.tensor(pose_s_np).unsqueeze(0).repeat(opts.cut, 1, 1).float().to(opts.device)  # [T, 4, 4]
pose_t = torch.tensor(pose_seq_np).float().to(opts.device)  # [T, 4, 4]
# from IPython import embed; embed()  # 调试用

# ===== 2. 获取文本描述（caption） =====
caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path, torch_dtype=torch.float16).to(opts.device)
prompt = get_caption(opts, frames[opts.video_length // 2])

# ===== 3. 预测深度图 =====
depth_model = DepthCrafterDemo(opts.unet_path, opts.pre_train_path, opts.cpu_offload, opts.device)
depths = depth_model.infer(
    frames,
    opts.near, opts.far, opts.depth_inference_steps,
    opts.depth_guidance_scale, opts.window_size, opts.overlap
).to(opts.device)
frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0

# ===== 4. 使用 Warper 合成新视角视频 =====
warper = Warper(device=opts.device)
warped_images, masks = [], []
for i in tqdm(range(opts.video_length), desc="Rendering new view"):
    if i<opts.cut:
        warped, mask, _, _ = warper.forward_warp(
            frames[0:1], None, depths[0:1],
            pose_s[0:1], pose_t[i:i+1], K[0:1],
            None, opts.mask, twice=False
        )
        warped_images.append(warped)
        masks.append(mask)
    else:
        warped, mask, _, _ = warper.forward_warp(
            frames[i-opts.cut:i-opts.cut+1], None, depths[i-opts.cut:i-opts.cut+1],
            pose_s[0:1], pose_t[-1:], K[0:1],
            None, opts.mask, twice=False
        )
        warped_images.append(warped)
        masks.append(mask)

cond_video = (torch.cat(warped_images) + 1.0) / 2.0
cond_masks = torch.cat(masks)

# ===== 5. 尺寸调整并保存中间结果视频 =====
frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')

os.makedirs(opts.save_dir, exist_ok=True)
save_video((frames[: opts.video_length - opts.cut].permute(0, 2, 3, 1) + 1.0) / 2.0, os.path.join(opts.save_dir, 'input.mp4'), fps=opts.fps)
save_video(cond_video.permute(0, 2, 3, 1).cpu().numpy(), os.path.join(opts.save_dir, 'render.mp4'), fps=opts.fps)
save_video(cond_masks[opts.cut:].repeat(1, 3, 1, 1).permute(0, 2, 3, 1), os.path.join(opts.save_dir, 'mask.mp4'), fps=opts.fps)

# ===== 6. 生成条件准备（调整 shape） =====
frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
frames_ref = frames[:, :, :10]
cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
print("cond_masks shape:", cond_masks.shape)
generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

# ===== 清理内存（释放模型） =====
del depth_model, caption_processor, captioner
gc.collect()
torch.cuda.empty_cache()

# ===== 7. 加载模型组件 =====
transformer = CrossTransformer3DModel.from_pretrained(opts.transformer_path).to(opts.weight_dtype)
vae = AutoencoderKLCogVideoX.from_pretrained(opts.model_name, subfolder="vae").to(opts.weight_dtype)
text_encoder = T5EncoderModel.from_pretrained(opts.model_name, subfolder="text_encoder", torch_dtype=opts.weight_dtype)

sampler_cls = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}[opts.sampler_name]
scheduler = sampler_cls.from_pretrained(opts.model_name, subfolder="scheduler")

pipeline = TrajCrafter_Pipeline.from_pretrained(
    opts.model_name, vae=vae, text_encoder=text_encoder,
    transformer=transformer, scheduler=scheduler,
    torch_dtype=opts.weight_dtype
)

if getattr(opts, "low_gpu_memory_mode", False):  # 兼容无该字段情况
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

# ===== 8. 文本条件生成视频 =====
with torch.no_grad():
    sample = pipeline(
        prompt, num_frames=opts.video_length,
        negative_prompt=opts.negative_prompt,
        height=opts.sample_size[0],
        width=opts.sample_size[1],
        generator=generator,
        guidance_scale=opts.diffusion_guidance_scale,
        num_inference_steps=opts.diffusion_inference_steps,
        video=cond_video,
        mask_video=cond_masks,
        reference=frames_ref,
    ).videos

save_video(sample[0].permute(1, 2, 3, 0)[opts.cut:], os.path.join(opts.save_dir, 'gen.mp4'), fps=opts.fps)

# ===== 9. 拼接生成结果和原始输入可视化 =====
viz = True
if viz:
    left = frames[0][:, :opts.video_length - opts.cut].to(opts.device)
    right = sample[0][:, opts.cut:].to(opts.device)
    interval = torch.ones(3, opts.video_length - opts.cut, 384, 30).to(opts.device)
    result = torch.cat((left, interval, right), dim=3)
    final_result = torch.cat((result, torch.flip(result, dims=[1])[:, 1:]), dim=1)
    save_video(final_result.permute(1, 2, 3, 0).cpu(), os.path.join(opts.save_dir, 'viz.mp4'), fps=opts.fps * 2)