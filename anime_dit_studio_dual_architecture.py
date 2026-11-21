# ============================================================
# ANIME DiT STUDIO — AnimeDiT Generator
# Model selector: AnimeDiT-384 / AnimeDiT-50M (adaLN-Zero)
# Original studio branding retained.
# Each model MUST use its matching checkpoint.
# ============================================================

import os, math, time, gc, threading, traceback, subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SR_MODEL_PATH = os.path.join(MODELS_DIR, "RealESRGAN_x4plus_anime_6B.pth")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, TIMESTEPS = 64, 4, 3, 1000
DEFAULT_SAMPLING_STEPS = 100
DDIM_ETA = 0.0
SR_SCALE = 4
OUTPUT_RESOLUTIONS = (64, 256, 512, 1024)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if USE_AMP:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

MODEL_CONFIGS = {
    "AnimeDiT-384": {
        "path": os.path.join(MODELS_DIR, "dit_ema_epoch_95.pth"),
        "embed": 384, "heads": 6, "layers": 6, "kind": "legacy"
    },
    "AnimeDiT-50M": {
        "path": os.path.join(MODELS_DIR, "dit_ema_epoch_65_50M.pth"),
        "embed": 512, "heads": 8, "layers": 10, "kind": "adaln"
    }
}
loaded_model = None
loaded_variant = None
load_lock = threading.Lock()
sr_model = None

# -------------------- BRAND COLORS --------------------
BG = "#10131c"
HEADER_BG = "#171c29"
CARD_BG = "#1c2332"
CARD_INNER = "#252e40"
ACCENT = "#647dff"
ACCENT_HOVER = "#7890ff"
TEXT = "#f1f4ff"
MUTED = "#929db4"
GREEN = "#49dba0"
BORDER = "#303b52"

# -------------------- SHARED LAYERS --------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(
            half, device=time.device, dtype=torch.float32
        ) * -scale)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class PatchEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(
            IN_CHANNELS, dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

# -------------------- OLD 384 MODEL --------------------
class LegacyDiffusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d, heads, layers = 384, 6, 6
        n = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch_embed = PatchEmbedder(d)
        self.pos_embed = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d),
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 2,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.to_pixels = nn.Linear(d, PATCH_SIZE * PATCH_SIZE * IN_CHANNELS)

    def forward(self, x, t):
        b = x.size(0)
        tokens = self.patch_embed(x) + self.pos_embed
        tokens = tokens + self.time_mlp(t).unsqueeze(1)
        tokens = self.transformer(tokens)
        x = self.to_pixels(tokens)
        p, g = PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE
        x = x.view(b, g, g, p, p, IN_CHANNELS)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

# -------------------- NEW 50M adaLN-Zero MODEL --------------------
def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden=512, heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden * mlp_ratio, hidden)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN_modulation(c).chunk(6, dim=1)
        z = modulate(self.norm1(x), s1, sc1)
        attn, _ = self.attn(z, z, z, need_weights=False)
        x = x + g1.unsqueeze(1) * attn
        z = modulate(self.norm2(x), s2, sc2)
        return x + g2.unsqueeze(1) * self.mlp(z)

class AdaLNZeroDiffusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d, heads, layers, ratio = 512, 8, 10, 4
        self.image_size = IMAGE_SIZE
        self.patch_size = PATCH_SIZE
        self.in_channels = IN_CHANNELS
        self.patch_embed = PatchEmbedder(d)
        n = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d),
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.blocks = nn.ModuleList([
            DiTBlock(d, heads, ratio) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.to_pixels = nn.Linear(d, PATCH_SIZE * PATCH_SIZE * IN_CHANNELS)
        nn.init.zeros_(self.to_pixels.weight)
        nn.init.zeros_(self.to_pixels.bias)

    def forward(self, x, t):
        b = x.size(0)
        tokens = self.patch_embed(x) + self.pos_embed
        c = self.time_mlp(t)
        for block in self.blocks:
            tokens = block(tokens, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        tokens = modulate(self.final_norm(tokens), shift, scale)
        x = self.to_pixels(tokens)
        p, g = self.patch_size, self.image_size // self.patch_size
        x = x.view(b, g, g, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, self.in_channels, self.image_size, self.image_size)

# -------------------- CHECKPOINT LOADER --------------------
def make_model(variant):
    return (LegacyDiffusionTransformer() if variant == "AnimeDiT-384"
            else AdaLNZeroDiffusionTransformer())

def get_weights(ckpt):
    if not isinstance(ckpt, dict):
        raise RuntimeError("Неподдерживаемый формат checkpoint.")
    for key in ("ema_state_dict", "model_state_dict", "state_dict"):
        if key in ckpt:
            print("Using", key)
            return ckpt[key]
    print("Using raw weights")
    return ckpt

def load_model(variant):
    global loaded_model, loaded_variant
    with load_lock:
        if loaded_model is not None and loaded_variant == variant:
            return loaded_model

        path = MODEL_CONFIGS[variant]["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint для {variant} не найден:\\n{path}\\n"
                "Проверь имя файла в MODEL_CONFIGS."
            )

        if loaded_model is not None:
            del loaded_model
            loaded_model, loaded_variant = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Loading {variant} from: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        weights = get_weights(ckpt)
        if weights and all(k.startswith("module.") for k in weights):
            weights = {k.replace("module.", "", 1): v for k, v in weights.items()}

        model = make_model(variant)
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint не соответствует {variant}. "
                "Выбери файл, обученный именно для этой архитектуры.\\n\\n"
                f"{exc}"
            ) from exc

        model.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        count = sum(p.numel() for p in model.parameters())
        print(f"{variant}: {count:,} parameters ({count / 1e6:.2f}M)")
        loaded_model, loaded_variant = model, variant
        return model

# -------------------- COSINE DDPM SCHEDULER --------------------
class DiffusionScheduler:
    def __init__(self, timesteps, device):
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        x = steps / timesteps
        abar = torch.cos(((x + s) / (1 + s)) * math.pi * 0.5) ** 2
        abar = abar / abar[0]
        betas = 1.0 - abar[1:] / abar[:-1]
        self.betas = torch.clamp(betas, 1e-6, 0.999).float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1.0 - prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance.clamp_(min=1e-20)

scheduler = DiffusionScheduler(TIMESTEPS, DEVICE)
ALPHAS_CUMPROD = scheduler.alphas_cumprod.double().cpu().tolist()

@torch.inference_mode()
def generate_one_image(model, seed, steps, progress_callback=None):
    if not 1 <= steps <= TIMESTEPS:
        raise ValueError(f"Количество шагов должно быть от 1 до {TIMESTEPS}.")

    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    images = torch.randn(
        1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
        device=DEVICE, generator=gen
    )
    timesteps = torch.linspace(
        TIMESTEPS - 1, 0, steps
    ).round().long().tolist()

    for step, timestep in enumerate(timesteps):
        alpha_bar = max(ALPHAS_CUMPROD[timestep], 1e-12)
        alpha_bar_prev = (
            ALPHAS_CUMPROD[timesteps[step + 1]]
            if step + 1 < len(timesteps)
            else 1.0
        )
        t = torch.full((1,), timestep, device=DEVICE, dtype=torch.long)
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predicted_noise = model(images, t)
        else:
            predicted_noise = model(images, t)

        predicted_noise = predicted_noise.float()
        alpha_bar_sqrt = math.sqrt(alpha_bar)
        one_minus_alpha_bar_sqrt = math.sqrt(1.0 - alpha_bar)
        predicted_original = (
            images.float() - one_minus_alpha_bar_sqrt * predicted_noise
        ) / alpha_bar_sqrt
        predicted_original = predicted_original.clamp(-1, 1)
        predicted_noise = (
            images.float() - alpha_bar_sqrt * predicted_original
        ) / max(one_minus_alpha_bar_sqrt, 1e-12)

        sigma = DDIM_ETA * math.sqrt(
            max((1.0 - alpha_bar_prev) / (1.0 - alpha_bar), 0.0)
            * max(1.0 - alpha_bar / alpha_bar_prev, 0.0)
        )
        direction_scale = math.sqrt(max(1.0 - alpha_bar_prev - sigma ** 2, 0.0))
        images = (
            math.sqrt(alpha_bar_prev) * predicted_original
            + direction_scale * predicted_noise
        )

        if sigma > 0.0 and step + 1 < len(timesteps):
            images = images + sigma * torch.randn(
                images.shape, device=DEVICE, generator=gen
            )
        if not torch.isfinite(images).all():
            raise RuntimeError(f"NaN/Inf at diffusion step {timestep}")
        if progress_callback and (
            step % max(1, len(timesteps) // 10) == 0
            or step == len(timesteps) - 1
        ):
            progress_callback(step + 1, len(timesteps))
    return images[0].float().clamp(-1, 1)

# -------------------- IMAGE FILES --------------------
def tensor_to_pil(t):
    arr = ((t.detach().cpu().float().clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(arr.byte().permute(1, 2, 0).numpy(), "RGB")

def make_run_dir(seed, variant, resolution, steps):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = os.path.join(
        RESULTS_DIR,
        f"{variant}_{stamp}_seed_{seed}_{resolution}px_ddim_{steps}"
    )
    os.makedirs(folder, exist_ok=False)
    return folder

def load_sr_model():
    global sr_model
    if sr_model is not None:
        return sr_model
    if not os.path.isfile(SR_MODEL_PATH):
        raise FileNotFoundError(
            "Для разрешений выше 64x64 положи "
            "RealESRGAN_x4plus_anime_6B.pth в папку models."
        )

    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as exc:
        raise RuntimeError(
            "Для апскейлинга установи библиотеку: "
            "python -m pip install spandrel"
        ) from exc

    descriptor = ModelLoader().load_from_file(SR_MODEL_PATH)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError("Файл RealESRGAN не является image-to-image моделью.")
    if descriptor.scale != SR_SCALE or descriptor.input_channels != IN_CHANNELS:
        raise RuntimeError(
            "Ожидается RGB RealESRGAN x4, "
            f"получена модель x{descriptor.scale} с "
            f"{descriptor.input_channels} каналами."
        )

    sr_model = descriptor.model.to(DEVICE).eval()
    for parameter in sr_model.parameters():
        parameter.requires_grad_(False)
    return sr_model

@torch.inference_mode()
def upscale_image(image, resolution):
    if resolution not in OUTPUT_RESOLUTIONS:
        raise ValueError(f"Неподдерживаемое разрешение: {resolution}x{resolution}.")
    if resolution == IMAGE_SIZE:
        return image

    model = load_sr_model()
    if resolution == 512:
        image = image.resize((128, 128), Image.Resampling.LANCZOS)

    passes = 2 if resolution == 1024 else 1
    for _ in range(passes):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = (
            torch.from_numpy(array)
            .to(DEVICE)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP
        ):
            output = model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.float()

        if not torch.isfinite(output).all():
            with torch.autocast(device_type=DEVICE.type, enabled=False):
                output = model(tensor.float()).float()
            if not torch.isfinite(output).all():
                raise RuntimeError("RealESRGAN returned NaN or Inf values.")

        expected_size = (image.height * SR_SCALE, image.width * SR_SCALE)
        if tuple(output.shape[-2:]) != expected_size:
            raise RuntimeError(
                f"RealESRGAN returned {tuple(output.shape[-2:])}, "
                f"expected {expected_size}."
            )

        output = (
            output[0].clamp(0, 1).mul(255).round().byte()
            .permute(1, 2, 0).cpu().numpy()
        )
        image = Image.fromarray(output, "RGB")

    if image.size != (resolution, resolution):
        raise RuntimeError(
            f"Ожидалось изображение {resolution}x{resolution}, "
            f"получено {image.width}x{image.height}."
        )
    return image

def save_image_tensor(tensor, folder, index, resolution):
    image = upscale_image(tensor_to_pil(tensor), resolution)
    path = os.path.join(folder, f"image_{index:05d}.png")
    image.save(path, "PNG")
    return image, path

def create_grid(folder, count):
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    cell, pad = 128, 8
    grid = Image.new("RGB", (cols * cell + (cols + 1) * pad,
                             rows * cell + (rows + 1) * pad), (24, 29, 42))
    for i in range(count):
        path = os.path.join(folder, f"image_{i + 1:05d}.png")
        if not os.path.exists(path):
            continue
        with Image.open(path) as im:
            im = im.convert("RGB").resize((cell, cell), Image.Resampling.NEAREST)
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + pad)
        grid.paste(im, (x, y))
    path = os.path.join(folder, "grid.png")
    grid.save(path)
    return path

def fmt_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

# -------------------- GUI --------------------
class AnimeGeneratorApp:
    def __init__(self, root):
        self.root = root
        root.title("ANIME DiT STUDIO")
        root.geometry("1120x900")
        root.minsize(850, 700)
        root.configure(bg=BG)
        self.running = False
        self.photos = []
        self.started = None
        self.timer_on = False
        self.completed = 0
        self.build_ui()

    def label(self, parent, text, size=10, color=TEXT, bold=False, bg=None):
        return tk.Label(parent, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                        fg=color, bg=bg or parent.cget("bg"))

    def button(self, parent, text, command, bg=ACCENT):
        return tk.Button(parent, text=text, command=command, font=("Segoe UI", 10, "bold"),
                         bg=bg, fg="white", activebackground=ACCENT_HOVER,
                         activeforeground="white", relief="flat", bd=0,
                         cursor="hand2", padx=18, pady=11)

    def build_ui(self):
        header = tk.Frame(self.root, bg=HEADER_BG, height=108)
        header.pack(fill="x")
        header.pack_propagate(False)
        left = tk.Frame(header, bg=HEADER_BG)
        left.pack(side="left", padx=28, pady=18)
        self.label(left, "ANIME DiT STUDIO", 25, TEXT, True).pack(anchor="w")
        self.label(left, "ANIME DiT GENERATOR  /  EMA CHECKPOINT  /  DDIM", 9, MUTED).pack(anchor="w", pady=4)
        device = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
        self.label(header, f"COMPUTE: {device}", 10, GREEN, True, HEADER_BG).pack(side="right", padx=28)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=18)

        settings = tk.Frame(main, bg=CARD_BG, padx=20, pady=18,
                            highlightthickness=1, highlightbackground=BORDER)
        settings.pack(fill="x")
        self.label(settings, "GENERATION SETTINGS", 12, TEXT, True).pack(anchor="w")
        self.label(settings, "Select the model checkpoint and generate images sequentially.", 9, MUTED).pack(anchor="w", pady=(4, 14))

        fields = tk.Frame(settings, bg=CARD_BG)
        fields.pack(fill="x")

        def field(title):
            f = tk.Frame(fields, bg=CARD_BG)
            f.pack(side="left", padx=(0, 20), anchor="n")
            self.label(f, title, 9, MUTED, True).pack(anchor="w")
            return f

        f = field("MODEL ARCHITECTURE")
        self.model_var = tk.StringVar(value="AnimeDiT-50M")
        self.combo = ttk.Combobox(f, textvariable=self.model_var,
                                  values=list(MODEL_CONFIGS), state="readonly", width=19)
        self.combo.pack(anchor="w", pady=(8, 0), ipady=5)
        self.info_var = tk.StringVar(value="512 / 8 heads / 10 layers / adaLN-Zero (~48.49M)")
        self.label(f, "", 8, MUTED)
        tk.Label(f, textvariable=self.info_var, bg=CARD_BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))
        self.combo.bind("<<ComboboxSelected>>", self.model_changed)

        f = field("OUTPUT RESOLUTION")
        self.resolution_var = tk.StringVar(value="64x64")
        self.resolution_combo = ttk.Combobox(
            f,
            textvariable=self.resolution_var,
            values=[f"{size}x{size}" for size in OUTPUT_RESOLUTIONS],
            state="readonly",
            width=10,
        )
        self.resolution_combo.pack(anchor="w", pady=(8, 0), ipady=5)

        f = field("NUMBER OF IMAGES")
        self.count_var = tk.StringVar(value="10")
        self.count_entry = tk.Entry(f, textvariable=self.count_var, width=12, font=("Segoe UI", 13, "bold"),
                                    bg=CARD_INNER, fg=TEXT, insertbackground="white", relief="flat")
        self.count_entry.pack(anchor="w", pady=(8, 0), ipady=8, ipadx=7)

        f = field("START SEED")
        self.seed_var = tk.StringVar(value="1234")
        self.seed_entry = tk.Entry(f, textvariable=self.seed_var, width=12, font=("Segoe UI", 13),
                                   bg=CARD_INNER, fg=TEXT, insertbackground="white", relief="flat")
        self.seed_entry.pack(anchor="w", pady=(8, 0), ipady=8, ipadx=7)

        f = field("DDIM STEPS")
        self.steps_var = tk.StringVar(value=str(DEFAULT_SAMPLING_STEPS))
        self.steps_entry = tk.Entry(
            f,
            textvariable=self.steps_var,
            width=8,
            font=("Segoe UI", 13),
            bg=CARD_INNER,
            fg=TEXT,
            insertbackground="white",
            relief="flat",
        )
        self.steps_entry.pack(anchor="w", pady=(8, 0), ipady=8, ipadx=7)

        self.start_btn = self.button(fields, "▶  START GENERATION", self.start_generation)
        self.start_btn.pack(side="left", pady=(19, 0))

        progress_card = tk.Frame(main, bg=CARD_BG, padx=20, pady=16,
                                 highlightthickness=1, highlightbackground=BORDER)
        progress_card.pack(fill="x", pady=14)
        row = tk.Frame(progress_card, bg=CARD_BG)
        row.pack(fill="x")
        self.status = tk.StringVar(value=f"Ready | {DEVICE}")
        self.label(row, "STATUS", 9, MUTED, True).pack(side="left")
        tk.Label(row, textvariable=self.status, bg=CARD_BG, fg=GREEN,
                 font=("Segoe UI", 10, "bold")).pack(side="right")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Studio.Horizontal.TProgressbar", troughcolor=CARD_INNER,
                        background=ACCENT, bordercolor=CARD_INNER, thickness=13)
        self.bar = ttk.Progressbar(progress_card, style="Studio.Horizontal.TProgressbar",
                                   maximum=100, mode="determinate")
        self.bar.pack(fill="x", pady=12)
        row = tk.Frame(progress_card, bg=CARD_BG)
        row.pack(fill="x")
        self.progress_text = tk.Label(row, text="0 / 0 images", bg=CARD_BG, fg=TEXT)
        self.progress_text.pack(side="left")
        self.step_text = tk.Label(
            row, text=f"0 / {DEFAULT_SAMPLING_STEPS} steps", bg=CARD_BG, fg=MUTED
        )
        self.step_text.pack(side="right")
        self.time_text = tk.Label(progress_card, text="ELAPSED TIME   00:00:00",
                                  bg=CARD_BG, fg=GREEN, font=("Segoe UI", 10, "bold"))
        self.time_text.pack(anchor="w", pady=(10, 0))

        row = tk.Frame(main, bg=BG)
        row.pack(fill="x", pady=(0, 10))
        self.label(row, "GENERATED IMAGES", 13, TEXT, True).pack(side="left")
        self.button(row, "OPEN RESULTS FOLDER", self.open_folder, CARD_INNER).pack(side="right")

        container = tk.Frame(main, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(container, bg=CARD_BG, highlightthickness=0)
        scroll = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.preview = tk.Frame(self.canvas, bg=CARD_BG)
        self.preview_window = self.canvas.create_window((0, 0), window=self.preview, anchor="nw")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        scroll.pack(side="right", fill="y")
        self.preview.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.preview_window, width=e.width))
        self.placeholder()

        footer = tk.Frame(self.root, bg=HEADER_BG, height=34)
        footer.pack(fill="x", side="bottom")
        self.label(footer, f"ANIME DiT STUDIO  •  64 × 64 BASE  •  DDIM DEFAULT {DEFAULT_SAMPLING_STEPS} STEPS  •  SEQUENTIAL GENERATION",
                   8, MUTED, bg=HEADER_BG).pack(pady=8)

    def model_changed(self, _event=None):
        if self.model_var.get() == "AnimeDiT-384":
            self.info_var.set("384 / 6 heads / 6 layers / Transformer")
        else:
            self.info_var.set("512 / 8 heads / 10 layers / adaLN-Zero (~48.49M)")

    def placeholder(self):
        self.label(self.preview, "Your generated images will appear here", 13, MUTED).pack(pady=90)

    def start_timer(self):
        self.started = time.perf_counter()
        self.timer_on = True
        self.tick_timer()

    def tick_timer(self):
        if self.timer_on:
            self.time_text.config(text=f"ELAPSED TIME   {fmt_time(time.perf_counter() - self.started)}")
            self.root.after(1000, self.tick_timer)

    def stop_timer(self):
        self.timer_on = False
        elapsed = fmt_time(time.perf_counter() - self.started) if self.started else "00:00:00"
        self.time_text.config(text=f"TOTAL TIME   {elapsed}")
        return elapsed

    def start_generation(self):
        if self.running:
            return
        try:
            count = int(self.count_var.get())
            seed = int(self.seed_var.get())
            steps = int(self.steps_var.get())
            resolution = int(self.resolution_var.get().split("x", 1)[0])
            if not 1 <= count <= 100000:
                raise ValueError("Количество изображений должно быть от 1 до 100000.")
            if not 0 <= seed <= 2**63 - 1:
                raise ValueError("Seed должен быть от 0 до 2^63-1.")
            if not 1 <= steps <= TIMESTEPS:
                raise ValueError(f"Количество шагов должно быть от 1 до {TIMESTEPS}.")
            if resolution not in OUTPUT_RESOLUTIONS:
                raise ValueError("Выбери одно из доступных разрешений.")
        except ValueError as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self.running, self.completed = True, 0
        self.bar["value"] = 0
        self.progress_text.config(text=f"0 / {count} images")
        self.step_text.config(text=f"0 / {steps} steps")
        self.status.set("Preparing model...")
        self.start_btn.config(state="disabled", text="GENERATING...", bg="#414d6b")
        self.combo.config(state="disabled")
        self.resolution_combo.config(state="disabled")
        self.count_entry.config(state="disabled")
        self.seed_entry.config(state="disabled")
        self.steps_entry.config(state="disabled")
        for w in self.preview.winfo_children():
            w.destroy()
        self.photos = []
        self.start_timer()
        threading.Thread(
            target=self.worker,
            args=(count, seed, self.model_var.get(), resolution, steps),
            daemon=True,
        ).start()

    def worker(self, count, seed, variant, resolution, steps):
        try:
            self.root.after(0, self.status.set, f"Loading {variant}...")
            model = load_model(variant)
            if resolution > IMAGE_SIZE:
                self.root.after(0, self.status.set, "Loading RealESRGAN Anime x4...")
                load_sr_model()
            folder = make_run_dir(seed, variant, resolution, steps)
            for idx in range(count):
                number = idx + 1
                self.root.after(
                    0, self.status.set,
                    f"Generating {resolution}x{resolution} image {number}/{count}..."
                )
                self.root.after(0, self.step_text.config, {"text": f"0 / {steps} steps"})

                def progress(step, total):
                    overall = ((number - 1) + step / total) / count * 100
                    self.root.after(0, self.update_progress, overall, number - 1, count, step, total)

                tensor = generate_one_image(model, seed + idx, steps, progress)
                image, path = save_image_tensor(tensor, folder, number, resolution)
                del tensor
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                self.root.after(0, self.add_image, image, number)
                self.root.after(0, self.set_completed, number, count)
                del image
            grid = create_grid(folder, count)
            self.root.after(0, self.finished, folder, grid, count, variant, resolution, steps)
        except Exception as e:
            self.root.after(0, self.failed, str(e), traceback.format_exc())

    def update_progress(self, value, done, total, step, total_steps):
        self.bar["value"] = value
        self.progress_text.config(text=f"{done} / {total} images completed")
        self.step_text.config(text=f"Current image: {step} / {total_steps} steps")

    def set_completed(self, done, total):
        self.completed = done
        self.bar["value"] = done / total * 100
        self.progress_text.config(text=f"{done} / {total} images completed")
        self.step_text.config(text="Image saved successfully")

    def add_image(self, image, index):
        if index == 1:
            for w in self.preview.winfo_children():
                w.destroy()
        card = tk.Frame(self.preview, bg=CARD_INNER, padx=8, pady=8,
                        highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=(index - 1) // 4, column=(index - 1) % 4, padx=8, pady=8)
        photo = ImageTk.PhotoImage(image.resize((145, 145), Image.Resampling.LANCZOS))
        self.photos.append(photo)
        tk.Label(card, image=photo, bg=CARD_INNER).pack()
        self.label(card, f"IMAGE {index:05d}", 9, TEXT, True).pack(pady=(7, 2))
        self.label(card, "SAVED", 8, GREEN, True).pack()
        for c in range(4):
            self.preview.grid_columnconfigure(c, weight=1)
        self.root.after(50, lambda: self.canvas.yview_moveto(1.0))

    def restore_controls(self):
        self.running = False
        self.start_btn.config(state="normal", text="▶  START GENERATION", bg=ACCENT)
        self.combo.config(state="readonly")
        self.resolution_combo.config(state="readonly")
        self.count_entry.config(state="normal")
        self.seed_entry.config(state="normal")
        self.steps_entry.config(state="normal")

    def finished(self, folder, grid, count, variant, resolution, steps):
        elapsed = self.stop_timer()
        self.restore_controls()
        self.bar["value"] = 100
        self.progress_text.config(text=f"{count} / {count} images completed")
        self.step_text.config(text="All diffusion steps completed")
        self.status.set(
            f"Completed | {resolution}x{resolution} | DDIM {steps} steps"
        )
        messagebox.showinfo("Generation complete",
                            f"Model: {variant}\\nGenerated: {count}\\n"
                            f"Resolution: {resolution}x{resolution}\\nDDIM steps: {steps}\\n"
                            f"Time: {elapsed}\\n\\n"
                            f"Images folder:\\n{folder}\\n\\nGrid preview:\\n{grid}")

    def failed(self, error, details):
        elapsed = self.stop_timer()
        self.restore_controls()
        self.status.set(f"Generation failed | Completed: {self.completed}")
        print("\\n" + "=" * 60 + "\\nGENERATION ERROR\\n" + "=" * 60)
        print(details)
        messagebox.showerror("Generation error",
                             f"{error}\\n\\nCompleted images: {self.completed}\\nTime: {elapsed}\\n\\n"
                             "Подробности ошибки выведены в консоль. Уже сохранённые изображения остались в results.")

    def open_folder(self):
        try:
            if os.name == "nt":
                os.startfile(RESULTS_DIR)
            elif os.uname().sysname == "Darwin":
                subprocess.Popen(["open", RESULTS_DIR])
            else:
                subprocess.Popen(["xdg-open", RESULTS_DIR])
        except Exception as e:
            messagebox.showerror("Error", f"Не удалось открыть папку:\\n{RESULTS_DIR}\\n{e}")


if __name__ == "__main__":
    print("ANIME DiT STUDIO | Device:", DEVICE)
    for name, cfg in MODEL_CONFIGS.items():
        print(name, "=>", cfg["path"])
    root = tk.Tk()
    app = AnimeGeneratorApp(root)
    root.mainloop()
