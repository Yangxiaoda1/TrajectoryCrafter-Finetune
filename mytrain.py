# train_trajcrafter_deepspeed.py
import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from accelerate import Accelerator
from einops import rearrange
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from models.crosstransformer3d import CrossTransformer3DModel
from models.utils import read_video_frames
from torch.cuda.amp import autocast
import argparse

# Utility functions

def get_resize_crop_region_for_grid(src, tgt_w, tgt_h):
    h, w = src
    r = h / w
    if r > (tgt_h / tgt_w):
        rh, rw = tgt_h, int(round(tgt_h / h * w))
    else:
        rw, rh = tgt_w, int(round(tgt_w / w * h))
    top  = (tgt_h - rh) // 2
    left = (tgt_w - rw) // 2
    return (top, left), (top + rh, left + rw)


def prepare_mask_latents(
    mask, masked_image,
    vae, scaling, temporal_comp,
    device
):
    # mask, masked_image: [B,C,F,H,W] pixel values
    # 参考pipeline的实现方式
    if mask is not None:
        mask = mask.to(device=device, dtype=vae.dtype)
        bs = 1
        new_mask = []
        with torch.no_grad():
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()  # 使用mode()而不是sample()
                new_mask.append(mask_bs)
        mask_latents = torch.cat(new_mask, dim=0)
        mask_latents = mask_latents * vae.config.scaling_factor
    else:
        mask_latents = None

    if masked_image is not None:
        masked_image = masked_image.to(device=device, dtype=vae.dtype)
        bs = 1
        new_mask_pixel_values = []
        with torch.no_grad():
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()  # 使用mode()而不是sample()
                new_mask_pixel_values.append(mask_pixel_values_bs)
        masked_image_latents = torch.cat(new_mask_pixel_values, dim=0)
        masked_image_latents = masked_image_latents * vae.config.scaling_factor
    else:
        masked_image_latents = None

    return mask_latents, masked_image_latents


def compute_rotary_pos_emb(
    height, width, num_frames,
    vae_sf_spatial, patch_size, attention_head_dim, device
):
    gh = height // (vae_sf_spatial * patch_size)
    gw = width  // (vae_sf_spatial * patch_size)
    base_w = 720 // (vae_sf_spatial * patch_size)
    base_h = 480 // (vae_sf_spatial * patch_size)
    coords = get_resize_crop_region_for_grid((gh, gw), base_w, base_h)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=coords,
        grid_size=(gh, gw),
        temporal_size=num_frames,
        use_real=True
    )
    return freqs_cos.to(device), freqs_sin.to(device)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False,
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False,
            )
            resized_mask = torch.cat(
                [first_frame_resized, remaining_frames_resized], dim=2
            )
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask, size=target_size, mode='trilinear', align_corners=False
        )
    return resized_mask

# Dataset class
class TrajCrafterTripletDataset(Dataset):
    def __init__(self, json_path, tokenizer, num_frames=49, image_size=(384,672)):
        with open(json_path) as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cond = read_video_frames(s['pointcloud_render_path'], self.num_frames, 2, 1024)
        mask = read_video_frames(s['mask_video_path'], self.num_frames, 2, 1024)
        ref  = read_video_frames(s['reference_video_path'], self.num_frames, 2, 1024)
        gt   = read_video_frames(s['target_video_path'], self.num_frames, 2, 1024)
        prompt = s['prompt']
        # print(f"mask.shape: {mask.shape}")
        txt = self.tokenizer(
            prompt, max_length=226, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        def to_tensor(v):
            t = torch.from_numpy(v).permute(3,0,1,2)
            return F.interpolate(
                t, size=self.image_size, mode='bilinear', align_corners=False
            )
        return {
            'cond_video': to_tensor(cond),
            'mask_video': to_tensor(mask).long(),
            'ref_video' : to_tensor(ref),
            'gt_video'  : to_tensor(gt),
            'input_ids' : txt.input_ids[0],
            'attention_mask': txt.attention_mask[0]
        }

# 包装所有模型组件的类
class TrajCrafterModel(torch.nn.Module):
    def __init__(self, model_config, batch_size=1):
        super().__init__()
        # 加载所有模型组件
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_config['model_path'], subfolder='text_encoder'
        ).eval()
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_config['model_path'], subfolder='vae'
        ).eval()
        self.transformer = CrossTransformer3DModel.from_pretrained(
            model_config['transformer_path']
        ).train()
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_config['model_path'], subfolder='scheduler'
        )
        # 冻结VAE参数
        for param in self.vae.parameters():
            param.requires_grad = False
        # 冻结text_encoder参数
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # Freeze strategy for transformer
        stage = model_config.get('stage', 1)
        for n, p in self.transformer.named_parameters():
            p.requires_grad = True
            if stage == 1 and 'ref_dit' in n:
                p.requires_grad = False
            if stage == 2 and not any(k in n for k in ('ref_dit.cross_attn','ref_dit.proj')):
                p.requires_grad = False
        # 计算VAE缩放因子
        self.vae_sf = 2**(len(self.vae.config.block_out_channels)-1)
        self.batch_size = batch_size
        
    def forward(self, batch):
        device = next(self.parameters()).device
        
        # 确保输入数据类型与模型参数类型匹配
        model_dtype = next(self.parameters()).dtype
        
        # Encode text and videos
        with torch.no_grad():
            txt_emb = self.text_encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )[0]
            
            gt   = batch['gt_video'].to(dtype=model_dtype)
            cond = batch['cond_video'].to(dtype=model_dtype)
            ref  = batch['ref_video'].to(dtype=model_dtype)
            mask = batch['mask_video']
            
            def enc(x):
                # 分帧处理，避免一次性爆显存
                outs = []
                for i in range(x.shape[0]):
                    # 确保输入数据类型与VAE参数类型匹配
                    x_i = x[i:i+1].to(dtype=self.vae.dtype)
                    outs.append(self.vae.encode(x_i)[0].sample() * self.vae.config.scaling_factor)
                z = torch.cat(outs, dim=0)
                return rearrange(z,'b c f h w->b f c h w')
                
            z_gt  = enc(gt)
            z_ref = enc(ref)
            
            # 清理显存
            del gt, cond, ref, mask
            torch.cuda.empty_cache()

        # Add noise to z_gt using prepare_latents function
        def prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            video_length,
            dtype,
            video=None,
        ):
            video = video.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_video = []
            with torch.no_grad():
                for i in range(0, video.shape[0], bs):
                    video_bs = video[i : i + bs]
                    # 确保输入数据类型与VAE参数类型匹配
                    video_bs = video_bs.to(dtype=self.vae.dtype)
                    video_bs = self.vae.encode(video_bs)[0]
                    video_bs = video_bs.sample()
                    new_video.append(video_bs)
            video = torch.cat(new_video, dim=0)
            video = video * self.vae.config.scaling_factor

            video_latents = video.repeat(batch_size // video.shape[0], 1, 1, 1, 1)
            video_latents = video_latents.to(device=device, dtype=dtype)
            video_latents = rearrange(video_latents, "b c f h w -> b f c h w")

            # Calculate shape based on actual encoded video frames
            actual_frames = video_latents.shape[1]
            shape = (
                batch_size,
                actual_frames,  # Use actual frame count from VAE encoding
                num_channels_latents,
                height // self.vae_sf,
                width // self.vae_sf,
            )

            # Add noise to latents
            noise = torch.randn(shape, device=device, dtype=model_dtype)
            # 生成随机timestep
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (shape[0],), device=device
            ).long()
            latents = self.scheduler.add_noise(video_latents, noise, timesteps)#这个地方应该加正态分布的noise还是加noise过vae的结果
            latents = latents * self.scheduler.init_noise_sigma

            return latents, noise

        # Use prepare_latents function
        z_gt_noisy, noise = prepare_latents(
            batch_size=self.batch_size,
            num_channels_latents=16,  # VAE output channels
            height=384,
            width=672,
            video_length=49,  # Original video length
            dtype=model_dtype,
            video=batch['gt_video'].to(dtype=model_dtype),  # Use gt video with correct dtype
        )
        
        # 生成新的timestep用于transformer
        tsteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (1,), device=device
        ).long()

        # 处理mask和masked_video
        mask = batch['mask_video']
        cond = batch['cond_video'].to(dtype=model_dtype)
        mask_f = mask.float()  # mask已经是单通道0/1，直接转float即可
        masked_video = cond * (mask_f < 0.5) + (-1) * (mask_f >= 0.5)
        
        # 编码mask和masked_video到latent
        _, masked_video_latents = prepare_mask_latents(
            None, masked_video.to(dtype=model_dtype), self.vae, self.vae.config.scaling_factor, 
            self.vae.config.temporal_compression_ratio, device
        )
        mask_latents = resize_mask((1 - mask[:,0,:,:,:]).unsqueeze(1).float(), masked_video_latents)
        
        # 重新排列维度: [B,C,F,H,W] -> [B,F,C,H,W]
        mask_latents = rearrange(mask_latents, 'b c f h w -> b f c h w')
        masked_video_latents = rearrange(masked_video_latents, 'b c f h w -> b f c h w')
            
        # 拼接mask和masked_video
        inpaint = torch.cat([mask_latents, masked_video_latents], dim=2)

        # Rotary positional embeddings
        cos, sin = compute_rotary_pos_emb(
            height=384, width=672,
            num_frames=z_gt_noisy.shape[1],
            vae_sf_spatial=self.vae_sf,
            patch_size=self.transformer.config.patch_size,
            attention_head_dim=self.transformer.config.attention_head_dim,
            device=device
        )
        img_rot = (cos, sin)

        # from IPython import embed;embed()
        print(f"z_gt_noisy.shape: {z_gt_noisy.shape}")
        print(f"inpaint.shape: {inpaint.shape}")
        print(f"z_ref.shape: {z_ref.shape}")
        z_gt_noisy = z_gt_noisy.to(dtype=self.transformer.dtype)
        inpaint = inpaint.to(dtype=self.transformer.dtype)
        z_ref = z_ref.to(dtype=self.transformer.dtype)
        # Forward & loss
        pred = self.transformer(
            hidden_states=z_gt_noisy,
            encoder_hidden_states=txt_emb,
            timestep=tsteps,
            inpaint_latents=inpaint,
            cross_latents=z_ref,
            image_rotary_emb=img_rot,
            return_dict=False
        )[0]
        loss = F.mse_loss(pred, noise)

        return loss

# Main training
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='训练的batch size')
    args = parser.parse_args()

    # 直接用默认构造
    accelerator = Accelerator()
    
    # 模型配置
    model_config = {
        'model_path': '/mnt/bn/xdatahl/yangxiaoda/TrajectoryCrafter-Finetune/TrajectoryCrafter/checkpoints/CogVideoX-Fun-V1.1-5b-InP',
        'transformer_path': '/mnt/bn/xdatahl/yangxiaoda/TrajectoryCrafter-Finetune/TrajectoryCrafter/checkpoints/TrajectoryCrafter',
        'stage': 1
    }
    
    # 创建模型
    model = TrajCrafterModel(model_config, batch_size=args.batch_size)
    
    # Data & optimizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    ds = TrajCrafterTripletDataset(
        '/mnt/bn/xdatahl/yangxiaoda/TrajectoryCrafter-Finetune/mytest/data.json', tokenizer
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5 if model_config['stage'] == 1 else 2e-6,
        weight_decay=0.01
    )
    
    # 使用accelerator包装所有组件
    model, opt, dl = accelerator.prepare(model, opt, dl)

    # Training loop
    for ep in range(10):
        for batch in tqdm(dl, desc=f'Epoch {ep}'):
            # Forward pass
            loss = model(batch)
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                opt.step()
                opt.zero_grad()

        accelerator.print(f'Epoch {ep} Loss {loss.item():.4f}')
        if accelerator.is_main_process:
            ckpt_dir = '/mnt/bn/xdatahl/yangxiaoda/TrajectoryCrafter-Finetune/TrajectoryCrafter/data/ckpts'
            os.makedirs(ckpt_dir, exist_ok=True)
            if ep % 5 == 0:
                torch.save(
                    accelerator.unwrap_model(model).transformer.state_dict(),
                    os.path.join(ckpt_dir, f'transformer_stage{model_config["stage"]}_ep{ep}.pth')
                )
