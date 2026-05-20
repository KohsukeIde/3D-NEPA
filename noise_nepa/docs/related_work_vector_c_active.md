# Vector C Related-Work Risk Notes

Status: active pre-mortem notes for Noise-NEPA / Denoising-Time NEPA.

## Narrow Claim To Preserve

The defendable claim is not "first diffusion 3D SSL method".

The narrow claim is:

> Learn 3D point-cloud representations by predicting denoising-time transitions in encoder latent space, with no point-coordinate reconstruction decoder, no masked-patch reconstruction decoder, and no external diffusion teacher. Semigroup consistency is imposed on latent denoising flow maps.

## Closest Conflicting Work

| Work | Primary source | Why it is close | Difference Noise-NEPA must preserve |
|---|---|---|---|
| PointDif | https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Point_Cloud_Pre-training_with_Diffusion_Models_CVPR_2024_paper.html | Diffusion-based point-cloud pretraining; conditional point generator recovers noisy point clouds. | Noise-NEPA should avoid coordinate reconstruction/generator claims and focus on latent transition/semigroup objectives. |
| Point-MaDi | https://papers.neurips.cc/paper_files/paper/2025/hash/4809dd4b628b6253d0aad0154014f7a3-Abstract-Conference.html | MAE plus diffusion for point-cloud pretraining; includes center diffusion and conditional patch diffusion. | Noise-NEPA should avoid MAE framing and decoder-side patch reconstruction. |
| PointDico | https://arxiv.org/abs/2512.08330 | Diffusion-guided contrastive 3D representation learning with denoising generative modeling and distillation. | Noise-NEPA should avoid contrastive distillation and diffusion-as-teacher framing. |
| PointSD | https://arxiv.org/abs/2507.09102 | Uses Stable Diffusion / point-to-image diffusion features for 3D SSL. | Noise-NEPA should avoid external image diffusion teachers and rendered-image feature alignment. |
| Consistency Models | https://proceedings.mlr.press/v202/song23a.html | Cross-time consistency / direct noisy-to-data mapping is established in generative modeling. | Noise-NEPA can cite this as conceptual support, but must not claim first denoising-time consistency. |
| Flow Map Matching | https://arxiv.org/abs/2406.07507 | Two-time flow maps and semigroup-like consistency are established for generative flows. | Noise-NEPA must position semigroup as applied to representation learning over point-cloud encoder latents. |
| ConTiCoM-3D | https://arxiv.org/abs/2509.01492 | Continuous-time consistency model for point-cloud generation. | Noise-NEPA is not a point generator and should not compete on generation. |
| MLPCM | https://arxiv.org/abs/2412.19413 | Latent point consistency model for 3D shape generation. | Noise-NEPA is SSL representation learning, not latent generative sampling. |

## Unsafe Claims

- First diffusion method for point-cloud SSL.
- First consistency model for point clouds.
- First latent consistency model for 3D.
- First denoising-time representation learner in general.
- Diffusion pretraining without specifying the decoder-free latent objective.

## Safe Claims

- Decoder-free latent denoising-time transition objective for 3D point-cloud SSL.
- Semigroup consistency over learned encoder latents rather than point-space generation.
- No coordinate reconstruction, no masked-patch reconstruction decoder, no Stable-Diffusion/image teacher.
- A controlled response to prior PointNEPA/PosetNEPA/ViewAction failures: replace arbitrary spatial/action order with known diffusion time.

## Experiments Needed To Distinguish The Claim

1. Same-shape noise-level retrieval: for each object, choose the correct target noise level among same-object candidates. This prevents cross-shape identity retrieval from carrying the result.
2. Semigroup held-out triples: evaluate direct `t -> r` versus composed `t -> s -> r` on triples not emphasized during training.
3. Decoder-free ablation: latent transition only versus latent transition + reconstruction decoder versus reconstruction-only baseline, with compute/memory and downstream transfer.
