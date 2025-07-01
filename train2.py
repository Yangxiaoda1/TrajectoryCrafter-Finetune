# train_trajcrafter.py
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
    # 1) Flatten frames
    def encode_video_pixel(x):
        flat = rearrange(x, "b c f h w -> (b f) c h w")
        with torch.no_grad():
            lat = vae.encode(flat.to(device, vae.dtype))[0].mode()
        lat = lat * scaling
        B = x.shape[0]
        F = x.shape[2]
        return rearrange(lat, "(b f) c h w -> b c f h w", b=B, f=F)

    mask_latents = encode_video_pixel(mask) if mask is not None else None
    mv_latents   = encode_video_pixel(masked_image) if masked_image is not None else None

    # 2) Temporal compression
    def compress(x):
        B,C,F,H,W = x.shape
        F_lat = (F - 1) // temporal_comp + 1
        idx = [i * temporal_comp for i in range(F_lat)]
        return x[:, :, idx]

    if mask_latents is not None:
        mask_latents = compress(mask_latents)
    if mv_latents is not None:
        mv_latents = compress(mv_latents)

    return mask_latents, mv_latents


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

# Main training
if __name__ == '__main__':
    height,width=384,672
    accelerator = Accelerator()
    device = accelerator.device

    # Load models
    tokenizer    = T5Tokenizer.from_pretrained('t5-base')
    text_encoder = T5EncoderModel.from_pretrained(
        'alibaba-pai/CogVideoX-Fun-V1.1-5b-InP', subfolder='text_encoder'
    ).eval().to(device)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        'alibaba-pai/CogVideoX-Fun-V1.1-5b-InP', subfolder='vae'
    ).eval().to(device)
    transformer = CrossTransformer3DModel.from_pretrained(
        'TrajectoryCrafter/TrajectoryCrafter'
    ).to(device).train()
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        'alibaba-pai/CogVideoX-Fun-V1.1-5b-InP', subfolder='scheduler'
    )

    # Freeze strategy
    stage = 1
    for n,p in transformer.named_parameters():
        p.requires_grad = True
        if stage == 1 and 'ref_dit' in n:
            p.requires_grad = False
        if stage == 2 and not any(k in n for k in ('ref_dit.cross_attn','ref_dit.proj')):
            p.requires_grad = False

    # Data & optimizer
    ds = TrajCrafterTripletDataset(
        '/home/tione/notebook/TrajectoryCrafter/data/mytest/data.json', tokenizer
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
    opt = torch.optim.AdamW(
        filter(lambda p:p.requires_grad, transformer.parameters()),
        lr=1e-5 if stage == 1 else 2e-6
    )
    transformer, opt, dl = accelerator.prepare(transformer,opt,dl)

    # Preprocessors
    vae_sf = 2**(len(vae.config.block_out_channels)-1)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_sf,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True
    )

    # Training loop
    for ep in range(10):
        for batch in tqdm(dl, desc=f'Epoch {ep}'):
            # Encode text and videos
            with torch.no_grad():
                txt_emb = text_encoder(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )[0]
                gt   = batch['gt_video'].to(device)
                cond = batch['cond_video'].to(device)
                ref  = batch['ref_video'].to(device)
                mask = batch['mask_video'].to(device)
                def enc(x):
                    z = vae.encode(x)[0].sample() * vae.config.scaling_factor
                    return rearrange(z,'b c f h w->b f c h w')
                z_gt  = enc(gt)
                z_ref = enc(ref)

            # Add noise to z_gt
            noise = torch.randn_like(z_gt)
            tsteps= torch.randint(
                0, scheduler.config.num_train_timesteps,
                (1,), device=device
            ).long()
            z_noisy = scheduler.add_noise(z_gt, noise, tsteps)

                                    # Prepare inpaint_latents: use zeros to match z_gt shape for training
            B, F_lat, C_lat, H_lat, W_lat = z_gt.shape
            mask_latents = torch.zeros((B, F_lat, 1, H_lat, W_lat), device=device, dtype=z_gt.dtype)
            masked_video_latents = torch.zeros((B, F_lat, C_lat, H_lat, W_lat), device=device, dtype=z_gt.dtype)
            inpaint = torch.cat([mask_latents, masked_video_latents], dim=2)

            # Rotary positional embeddings
            cos, sin = compute_rotary_pos_emb(
                height=height, width=width,
                num_frames=z_gt.shape[1],
                vae_sf_spatial=vae_sf,
                patch_size=transformer.config.patch_size,
                attention_head_dim=transformer.config.attention_head_dim,
                device=device
            )
            img_rot = (cos, sin)

            # Forward & loss
            pred = transformer(
                hidden_states=z_noisy,
                encoder_hidden_states=txt_emb,
                cross_latents=z_ref,
                inpaint_latents=inpaint,
                timestep=tsteps,
                image_rotary_emb=img_rot
            )[0]
            loss = F.mse_loss(pred.float(), noise)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

        accelerator.print(f'Epoch {ep} Loss {loss.item():.4f}')
        if accelerator.is_main_process:
            ckpt_dir = '/home/tione/notebook/TrajectoryCrafter/data/mytest/ckpts'
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                accelerator.unwrap_model(transformer).state_dict(),
                os.path.join(ckpt_dir, f'transformer_stage{stage}_ep{ep}.pth')
            )
