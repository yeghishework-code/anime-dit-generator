# ==============================================================
# ANIME FACE DiT DIFFUSION TRAINING (SCALED + adaLN-Zero)
# Streaming first epoch + RAM cache + EMA
# Multi-GPU DataParallel + Checkpoints
# ==============================================================

import os
import math
import random
import copy
import gc
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image, ImageOps
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# ==============================================================
# 1. HYPERPARAMETERS
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "anime-faces-180k")

# 0.01 = 1%, 0.05 = 5%, 0.30 = 30%, 1.0 = 100%
DATASET_FRACTION = 1.0

IMAGE_SIZE = 64
PATCH_SIZE = 4
IN_CHANNELS = 3

# Увеличенная архитектура DiT
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 10
MLP_RATIO = 4

TIMESTEPS = 1000

BATCH_SIZE = 128
EPOCHS = 300

LR = 0.00015
WEIGHT_DECAY = 0.00001

EMA_DECAY = 0.9995

SEED = 42

NUM_SAMPLE_IMAGES = 64
SAMPLE_SEED = 1234

SAVE_SAMPLE_EVERY_EPOCH = 1
SAVE_CHECKPOINT_EVERY_EPOCH = 1
SAVE_MODEL_EVERY_EPOCHS = 5

# Пустая строка = новый запуск (рекомендуется для новой архитектуры).
RESUME_PATH = ""

RUN_DIR = (
    os.path.dirname(RESUME_PATH)
    if RESUME_PATH
    else os.path.join(
        BASE_DIR,
        "runs",
        "anime_dit_xl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
)

SAMPLES_DIR = os.path.join(RUN_DIR, "samples")
MODELS_DIR = os.path.join(RUN_DIR, "models")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================
# 2. DEVICE / MULTI GPU
# ==============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
USE_AMP = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("\n========== HARDWARE ==========")
print("Device:", DEVICE)
print("GPU count:", GPU_COUNT)
for i in range(GPU_COUNT):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))
print("==============================\n")


# ==============================================================
# 3. SEED
# ==============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================================================
# 4. DATASET: STREAM FIRST EPOCH INTO RAM
# ==============================================================

class StreamingCachedDataset:

    def __init__(
        self,
        source_dir,
        fraction=1.0,
        resume_files=None
    ):
        self.source_dir = source_dir

        valid_formats = (".jpg", ".jpeg", ".png", ".webp")

        if resume_files is not None:
            self.files = list(resume_files)
        else:
            files = [
                f for f in os.listdir(source_dir)
                if f.lower().endswith(valid_formats)
            ]
            files.sort()

            if not files:
                raise RuntimeError("Датасет пуст!")

            if not 0 < fraction <= 1:
                raise ValueError("DATASET_FRACTION должен быть от 0 до 1")

            if fraction < 1:
                selected_count = max(1, int(len(files) * fraction))
                rng = random.Random(SEED)
                files = sorted(rng.sample(files, selected_count))

            self.files = files

        if not self.files:
            raise RuntimeError("Нет доступных изображений!")

        self.cache = torch.empty(
            (len(self.files), IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            dtype=torch.uint8
        )
        self.cached_count = 0
        self.cached_files = []
        self.loaded = False

        estimated_gb = (
            len(self.files) * IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE
        ) / (1024 ** 3)

        print("\n========== DATASET ==========")
        print("Selected images:", len(self.files))
        print("RAM allocation:", round(estimated_gb, 3), "GB")
        print("Source:", source_dir)
        print("=============================\n")

    def __len__(self):
        return self.cached_count if self.loaded else len(self.files)

    def load_image(self, path):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = img.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BICUBIC
            )
            array = np.array(img, dtype=np.uint8, copy=True)

        tensor = torch.from_numpy(array)
        return tensor.permute(2, 0, 1).contiguous()

    def load_source_image(self, idx):
        filename = self.files[idx]
        source_path = os.path.join(self.source_dir, filename)
        return self.load_image(source_path)

    def add_to_cache(self, image, filename):
        if self.cached_count >= len(self.files):
            raise RuntimeError("RAM cache переполнен!")

        cache_idx = self.cached_count
        self.cache[cache_idx].copy_(image)
        self.cached_files.append(filename)
        self.cached_count += 1
        return cache_idx

    def finish_cache(self):
        if self.cached_count == 0:
            raise RuntimeError("Не удалось загрузить изображения!")

        self.loaded = True
        memory_gb = (
            self.cached_count * IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE
        ) / (1024 ** 3)

        print("\n========== RAM CACHE READY ==========")
        print("Images:", self.cached_count)
        print("RAM:", round(memory_gb, 3), "GB")
        print("Images are cached in RAM; source files are unchanged.")
        print("=====================================\n")

    def get_ram_batch(self, indices):
        return self.cache[indices]

    def restore_from_source(self):
        print("\nВосстанавливаем RAM из исходного датасета...")
        for idx, filename in enumerate(self.files):
            image = self.load_image(os.path.join(self.source_dir, filename))
            self.cache[idx].copy_(image)
            self.cached_files.append(filename)
            self.cached_count += 1

            if (idx + 1) % 5000 == 0:
                print("Restored:", idx + 1, "/", len(self.files))

        self.finish_cache()


# ==============================================================
# 5. DIFFUSION SCHEDULER: COSINE
# ==============================================================

class DiffusionScheduler:

    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        self.device = device

        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        x = steps / timesteps

        alpha_bar = torch.cos(((x + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, min=0.000001, max=0.999)

        self.betas = betas.float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas
            * (
                1.0
                - torch.cat([
                    torch.ones(1, device=device),
                    self.alphas_cumprod[:-1]
                ])
            )
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(
            self.posterior_variance, min=1e-20)

    def q_sample(self, x_start, t, noise):
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_noise = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_noise * noise


# ==============================================================
# 6. TIME EMBEDDING & PATCH EMBEDDING
# ==============================================================

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=time.device,
                         dtype=torch.float32) * -emb_scale
        )
        emb = time[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class PatchEmbedder(nn.Module):

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ==============================================================
# 7. DiT BLOCK WITH adaLN-Zero
# ==============================================================

def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # 6 модуляций: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(
            normed_x, normed_x, normed_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        normed_mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed_mlp_in)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# ==============================================================
# 8. DiT MODEL
# ==============================================================

class DiffusionTransformer(nn.Module):

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_ratio=MLP_RATIO
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedder(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.to_pixels = nn.Linear(
            embed_dim,
            patch_size * patch_size * in_channels
        )
        nn.init.zeros_(self.to_pixels.weight)
        nn.init.zeros_(self.to_pixels.bias)

    def forward(self, x, t):
        B = x.size(0)
        tokens = self.patch_embed(x) + self.pos_embed
        c = self.time_mlp(t)

        for block in self.blocks:
            tokens = block(tokens, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        tokens = modulate(self.final_norm(tokens), shift, scale)
        x = self.to_pixels(tokens)

        p = self.patch_size
        grid = self.image_size // p

        x = x.view(B, grid, grid, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, self.in_channels, self.image_size, self.image_size)


# ==============================================================
# 9. MULTI GPU MODEL
# ==============================================================

def unwrap_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


base_model = DiffusionTransformer().to(DEVICE)

param_count = sum(p.numel() for p in base_model.parameters())
print(f"Model parameters: {param_count:,} (~{param_count / 1e6:.1f}M)")

if GPU_COUNT >= 2:
    print(f"Используем {GPU_COUNT} GPU через DataParallel")
    model = nn.DataParallel(base_model, device_ids=list(range(GPU_COUNT)))
else:
    print("Используем одну GPU")
    model = base_model


# ==============================================================
# 10. EMA
# ==============================================================

ema_model = copy.deepcopy(unwrap_model(model)).to(DEVICE)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def update_ema(ema, source, step):
    source = unwrap_model(source)
    decay = min(EMA_DECAY, (1 + step) / (10 + step))

    for ema_param, model_param in zip(ema.parameters(), source.parameters()):
        ema_param.mul_(decay).add_(model_param, alpha=1.0 - decay)

    for ema_buffer, model_buffer in zip(ema.buffers(), source.buffers()):
        ema_buffer.copy_(model_buffer)


# ==============================================================
# 11. OPTIMIZER / SCALER
# ==============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=WEIGHT_DECAY
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
criterion = nn.MSELoss()
scheduler = DiffusionScheduler(TIMESTEPS, DEVICE)


# ==============================================================
# 12. SAMPLING
# ==============================================================

@torch.no_grad()
def sample_diffusion(model, scheduler, fixed_noise):
    model.eval()
    img = fixed_noise.clone().to(DEVICE)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SAMPLE_SEED)

    for i in reversed(range(TIMESTEPS)):
        t = torch.full((img.size(0),), i, device=DEVICE, dtype=torch.long)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            predicted_noise = model(img, t)

        alpha = scheduler.alphas[i]
        alpha_bar = scheduler.alphas_cumprod[i]
        beta = scheduler.betas[i]

        mean = (
            img - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        ) / torch.sqrt(alpha)

        if i > 0:
            noise = torch.randn(img.shape, device=DEVICE, generator=generator)
            variance = scheduler.posterior_variance[i]
            img = mean + torch.sqrt(variance) * noise
        else:
            img = mean

    model.train()
    return img.clamp(-1, 1)


# ==============================================================
# 13. SAVE / PLOT
# ==============================================================

def save_samples(epoch, fixed_noise):
    print("Генерируем EMA samples...")
    samples = sample_diffusion(ema_model, scheduler, fixed_noise)
    samples = (samples.float().clamp(-1, 1) + 1.0) / 2.0

    path = os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}.png")
    save_image(samples, path, nrow=8)
    print("Samples saved:", path)
    print(
        "Sample min/max/mean/std:",
        round(samples.min().item(), 4),
        round(samples.max().item(), 4),
        round(samples.mean().item(), 4),
        round(samples.std().item(), 4)
    )
    return path


def save_loss_plot(history):
    if not history:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.xlabel("Training steps")
    plt.ylabel("MSE loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "loss.png"))
    plt.close()


# ==============================================================
# 14. CHECKPOINT
# ==============================================================

def get_expected_config():
    return {
        "dataset_fraction": DATASET_FRACTION,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "mlp_ratio": MLP_RATIO,
        "timesteps": TIMESTEPS,
        "batch_size": BATCH_SIZE,
        "beta_schedule": "cosine",
        "arch": "dit_adaln_zero"
    }


def save_checkpoint(epoch, global_step, dataset, history):
    path = os.path.join(RUN_DIR, "checkpoint_latest.pt")
    temp_path = path + ".tmp"

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "cached_files": dataset.cached_files,
        "fixed_noise": fixed_noise.cpu(),
        "config": get_expected_config()
    }, temp_path)

    os.replace(temp_path, path)
    print("EMA checkpoint сохранён. Всего шагов:", global_step)


# ==============================================================
# 15. DATASET INIT
# ==============================================================

start_epoch = 0
global_step = 0
history = []

fixed_noise = torch.randn(
    NUM_SAMPLE_IMAGES,
    IN_CHANNELS,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=DEVICE
)

if RESUME_PATH:
    if not os.path.exists(RESUME_PATH):
        raise FileNotFoundError(RESUME_PATH)

    print("Загружаем checkpoint...")
    checkpoint = torch.load(
        RESUME_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint.get("config") != get_expected_config():
        raise RuntimeError(
            "Гиперпараметры не совпадают с checkpoint. "
            "Для новой архитектуры оставь RESUME_PATH пустым."
        )

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    ema_model.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    history = checkpoint.get("history", [])
    fixed_noise = checkpoint["fixed_noise"].to(DEVICE)

    print("Продолжаем с эпохи:", start_epoch + 1)

    dataset = StreamingCachedDataset(
        source_dir=DATA_DIR,
        fraction=DATASET_FRACTION,
        resume_files=checkpoint["cached_files"]
    )
    dataset.restore_from_source()

else:
    dataset = StreamingCachedDataset(
        source_dir=DATA_DIR,
        fraction=DATASET_FRACTION
    )


# ==============================================================
# 16. TRAINING LOOP
# ==============================================================

print("\n========== TRAINING START ==========")

for epoch in range(start_epoch, EPOCHS):
    model.train()

    if dataset.loaded:
        indices = list(range(dataset.cached_count))
    else:
        indices = list(range(len(dataset.files)))

    random.shuffle(indices)

    batch_cache_indices = []
    successful_images = 0
    batch_number = 0

    for idx in indices:
        if not dataset.loaded:
            filename = dataset.files[idx]
            try:
                image = dataset.load_source_image(idx)
                cache_idx = dataset.add_to_cache(image, filename)
                batch_cache_indices.append(cache_idx)
            except Exception as e:
                print("Ошибка загрузки:", filename, str(e))
                continue
        else:
            batch_cache_indices.append(idx)

        if len(batch_cache_indices) < BATCH_SIZE:
            continue

        real_images = dataset.get_ram_batch(batch_cache_indices)
        batch_cache_indices = []

        real_images = real_images.to(DEVICE, non_blocking=True).float()
        real_images = real_images / 127.5 - 1.0

        B = real_images.size(0)

        flip_mask = torch.rand(B, device=DEVICE) < 0.5
        if flip_mask.any():
            real_images[flip_mask] = torch.flip(
                real_images[flip_mask], dims=[3])

        t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
        noise = torch.randn_like(real_images)
        noisy_images = scheduler.q_sample(real_images, t, noise)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            predicted_noise = model(noisy_images, t)
            loss = criterion(predicted_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_ema(ema_model, model, global_step)

        global_step += 1
        history.append(loss.item())
        successful_images += B

        if batch_number % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"| Batch [{batch_number}] "
                f"| Loss: {loss.item():.6f} "
                f"| Cached: {dataset.cached_count} "
                f"| EMA step: {global_step}"
            )

        batch_number += 1

    # Last partial batch
    if batch_cache_indices:
        real_images = dataset.get_ram_batch(batch_cache_indices)
        real_images = real_images.to(DEVICE, non_blocking=True).float()
        real_images = real_images / 127.5 - 1.0

        B = real_images.size(0)

        flip_mask = torch.rand(B, device=DEVICE) < 0.5
        if flip_mask.any():
            real_images[flip_mask] = torch.flip(
                real_images[flip_mask], dims=[3])

        t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
        noise = torch.randn_like(real_images)
        noisy_images = scheduler.q_sample(real_images, t, noise)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            predicted_noise = model(noisy_images, t)
            loss = criterion(predicted_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_ema(ema_model, model, global_step)

        global_step += 1
        history.append(loss.item())
        successful_images += B

    if not dataset.loaded:
        dataset.finish_cache()
        print("Первая эпоха завершена: следующие эпохи будут работать только из RAM.")

    print("\nЭпоха", epoch + 1, "завершена.")
    print("Обработано изображений:", successful_images)
    print("Всего шагов:", global_step)

    save_loss_plot(history)

    if (epoch + 1) % SAVE_SAMPLE_EVERY_EPOCH == 0:
        save_samples(epoch + 1, fixed_noise)

    if (epoch + 1) % SAVE_MODEL_EVERY_EPOCHS == 0:
        torch.save(
            unwrap_model(model).state_dict(),
            os.path.join(MODELS_DIR, f"dit_epoch_{epoch + 1}.pth")
        )
        torch.save(
            ema_model.state_dict(),
            os.path.join(MODELS_DIR, f"dit_ema_epoch_{epoch + 1}.pth")
        )

    if (epoch + 1) % SAVE_CHECKPOINT_EVERY_EPOCH == 0:
        save_checkpoint(epoch, global_step, dataset, history)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n===================================")
print("TRAINING FINISHED")
print("Results:", RUN_DIR)
print("===================================")
