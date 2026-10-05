# Pre-Render Filtering of Unstable Gaussians for Improved Extreme View Synthesis in 3D Gaussian Splatting

**Sebastian Teuttli** · Honors Thesis, Barrett, The Honors College at Arizona State University (2026) · Advised by Dr. Heni Ben Amor

[Project Page](https://sebastianteu.github.io/gaussian-filtering) · [Thesis PDF](https://sebastianteu.github.io/gaussian-filtering/static/thesis.pdf)

> The exact code used for the thesis results is preserved at tag [`thesis-submission`](https://github.com/SebastianTeu/gsplat-ood-filter/tree/thesis-submission). The `ood_filter` branch may include later cleanups (e.g. default values and output-path handling).

3D Gaussian Splatting renders novel views in real time, but renders from camera poses far outside the training distribution are often obscured by noisy, unstable Gaussians. This fork of [gsplat](https://github.com/nerfstudio-project/gsplat) adds an offline, pre-render filter inspired by [EV3DGS](https://arxiv.org/abs/2510.20027) (Bowness and Poullis, 2025). Instead of scoring Gaussians every frame, cameras are sampled once from PCA-aligned ellipsoids fitted to the scene, a grid of rays is cast from each camera, and a custom gsplat CUDA kernel evaluates an EV3DGS-style sensitivity metric (with hit depth computed in each Gaussian's rotation-only local space) for every Gaussian a ray hits. Gaussians whose rejection ratio exceeds a threshold are pruned and a new checkpoint is written, so the renderer itself is unchanged and keeps its real-time performance.

### Files added / modified

| File | Status | Purpose |
| --- | --- | --- |
| `ood_filter/gaussian_filter.py` | added | CLI: PCA ellipsoid fitting, camera sampling, look-at view matrices, rejection-ratio pruning, writes the filtered checkpoint |
| `gsplat/cuda/csrc/OODfilterCUDA.cu` | added | CUDA kernel (one thread per ray): ray–Gaussian intersection and instability scoring |
| `gsplat/cuda/csrc/OODfilter.cpp` | added | Host-side entry point that allocates the count tensors and launches the kernel |
| `gsplat/cuda/csrc/OODfilter.h` | added | Kernel launcher declaration |
| `gsplat/cuda/include/Ops.h` | modified | Declares `gsplat::ood_filter` |
| `gsplat/cuda/ext.cpp` | modified | Registers the `ood_filter` Python binding |
| `gsplat/cuda/_wrapper.py` | modified | Python wrapper `gsplat.cuda._wrapper.ood_filter(...)` returning `(reject_counts, total_counts)` |

### Usage

**1. Install from source.** The filter's CUDA kernel is only in this fork, so the PyPI `gsplat` package will not work. Install [PyTorch](https://pytorch.org/get-started/locally/) first, then:

```bash
git clone --recursive https://github.com/SebastianTeu/gsplat-ood-filter.git
cd gsplat-ood-filter
pip install -e .
pip install -r examples/requirements.txt   # for training / viewing
```

**2. Train a scene** with gsplat's simple trainer (the thesis used the default strategy for 30,000 iterations on COLMAP-format data, without downscaling). This writes `results/<scene>/ckpts/ckpt_29999_rank0.pt`:

```bash
cd examples
python simple_trainer.py default --data_dir <path/to/colmap/scene> --data_factor 1 --result_dir results/<scene>
cd ..
```

**3. Run the filter** on the checkpoint (run from the repository root):

```bash
python ood_filter/gaussian_filter.py --ckpt examples/results/<scene>/ckpts/ckpt_29999_rank0.pt
```

The filtered checkpoint is saved next to the input with the parameters encoded in its name, e.g.
`ckpt_29999_rank0_ood_count-<N>_xg-0.0001_ratio-0.25_pca-0.9_slices-5x6_ellipsoids-2.0-4.0.pt`.
It contains only the `splats` dictionary (`means`, `quats`, `scales`, `opacities`, `sh0`, `shN`).

| Flag | Default | Description |
| --- | --- | --- |
| `--ckpt` | *(required)* | Path to the input `.pt` checkpoint |
| `--xg_thresh` | `1e-4` | Sensitivity threshold τ<sub>xg</sub>; a hit scoring above it is marked unstable |
| `--ratio_thresh` | `0.25` | Rejection-ratio threshold τ<sub>ratio</sub>; Gaussians above it are pruned |
| `--nx`, `--ny` | `100`, `100` | Ray grid per synthetic camera |
| `--pca_percentile` | `0.9` | Percentile of the projected Gaussian means used for the base ellipsoid radii |
| `--ellipsoid_scalars` | `2.0 4.0` | One or more multipliers on the base radii; cameras are sampled on each scaled ellipsoid |
| `--num_slices` | `5` | Slices along the ellipsoid's minor axis (poles excluded) |
| `--num_cameras_per_slice` | `6` | Cameras evenly spaced around each slice, all looking at the scene center |
| `--near_plane` | 5% of the smallest base radius | Hits closer than this are ignored |

These defaults are the parameters used for all results in the thesis (Chapter 6).

**4. View the filtered scene** with gsplat's viewer:

```bash
cd examples
python simple_viewer.py --ckpt <path/to/filtered>.pt --port 8080
```

> Note: `examples/simple_trainer.py --ckpt` expects a `step` key in the checkpoint, which the filtered checkpoint does not contain, so use `simple_viewer.py` (or load the `splats` dict yourself) for filtered scenes.

The filter can also be called from Python via `compute_instability_mask(...)` in `ood_filter/gaussian_filter.py`, or per camera via `gsplat.cuda._wrapper.ood_filter(...)`.

---

## Original gsplat README

# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## News

[Jan 2026] [PPIPS](https://research.nvidia.com/labs/sil/projects/ppisp/) is integreated as an alternative way of bilateral grid to compensate the training views.

[May 2025] Arbitrary batching (over multiple scenes and multiple viewpoints) is supported now!! Checkout [here](docs/batch.md) for more details! Kudos to [Junchen Liu](https://junchenliu77.github.io/).

[May 2025] [Jonathan Stephens](https://x.com/jonstephens85) makes a great [tutorial video](https://www.youtube.com/watch?v=ACPTiP98Pf8) for Windows users on how to install gsplat and get start with 3DGUT.

[April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) is now integrated in gsplat! Checkout [here](docs/3dgut.md) for more details. [[NVIDIA Tech Blog]](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) [[NVIDIA Sweepstakes]](https://www.nvidia.com/en-us/research/3dgut-sweepstakes/)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)


## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
