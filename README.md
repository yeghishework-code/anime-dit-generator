# Anime Diffusion

This project contains an anime-face diffusion model, a dataset archive, and a
desktop application for generating images.

## Extract the dataset

Run this command from the project root:

```powershell
python unpack_dataset.py
```

The script extracts `anime_faces_dataset.tar` to
`datasets/anime-faces-180k`, the directory used by `train_model.py`. The
archive contains approximately 185,000 files. Download it from
[Hugging Face](https://huggingface.co/Yeghishee/anime_generate_diffT_50M/tree/main).
To use a different archive or destination:

```powershell
python unpack_dataset.py "path\to\archive.tar" --destination "path\to\folder"
```

## Train the model

Install the project's dependencies: PyTorch, torchvision, NumPy, Pillow, and
Matplotlib. Then run:

```powershell
python train_model.py
```

Training starts from scratch by default and saves results to `runs`. Images
are read from the extracted dataset and cached in RAM; the source files are
not copied or modified. Caching the full dataset requires approximately
2.1 GB of RAM.

To resume training from a checkpoint, set its path in `RESUME_PATH` near the
top of `train_model.py`.

## Generate images

Launch the desktop application:

```powershell
python anime_dit_studio_dual_architecture.py
```

Choose an output resolution of 64×64, 256×256, 512×512, or 1024×1024, and set
the number of DDIM sampling steps. The default is 100 steps; valid values are
1–1000. The diffusion schedule remains at 1000 steps to match training, while
DDIM samples the requested number of steps from that schedule.

Resolutions above 64×64 use RealESRGAN through `spandrel`. Place
`RealESRGAN_x4plus_anime_6B.pth` in the `models` folder. The DiT and
RealESRGAN checkpoints are available from the
[Hugging Face repository](https://huggingface.co/Yeghishee/anime_generate_diffT_50M/tree/main).
If `spandrel` is not installed, install it with:

```powershell
python -m pip install spandrel
```

RealESRGAN performs one x4 pass for 256×256 and two passes for 1024×1024. For
512×512, the base image is first resized to 128×128 and then processed with
RealESRGAN x4. Generated images are saved in `results`.

---

**Harutyunyan Yeghishe — AI/ML Engineer**
