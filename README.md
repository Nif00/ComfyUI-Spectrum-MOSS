# ComfyUI-Spectrum-MOSS

Spectrum feature forecasting for the MOSS-Image DiT. On forecast steps the 42 blocks are skipped and the
Chebyshev-fitted pre-head feature goes straight into the final norm and head.

Add **Spectrum MOSS-Image** (`sampling/spectrum`) between the MOSS-Image loader and the sampler. Parameters match
[ComfyUI-Spectrum-Proper](https://github.com/xmarre/ComfyUI-Spectrum-Proper).

Supported samplers (one model call per step): `euler`, `euler_ancestral`, `er_sde`, `dpmpp_2m`, `dpmpp_2m_sde`,
`dpmpp_3m_sde` (and `_gpu` variants), `res_multistep`, `gradient_estimation`. Other samplers run normally with a warning.
