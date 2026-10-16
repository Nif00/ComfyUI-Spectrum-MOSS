import logging

import torch
from comfy import patcher_extension

LOG = logging.getLogger(__name__)

# Samplers that call the model exactly once per solver step.
SAMPLERS = {
    "sample_euler",
    "sample_euler_ancestral",
    "sample_er_sde",
    "sample_dpmpp_2m",
    "sample_dpmpp_2m_sde",
    "sample_dpmpp_2m_sde_gpu",
    "sample_dpmpp_3m_sde",
    "sample_dpmpp_3m_sde_gpu",
    "sample_res_multistep",
    "sample_gradient_estimation",
}


def chebyshev(t, degree):
    basis = [1.0, t]
    for _ in range(2, degree + 1):
        basis.append(2 * t * basis[-1] - basis[-2])
    return torch.tensor(basis[: degree + 1], dtype=torch.float64)


class Forecaster:
    """Online ridge fit of Chebyshev coefficients to one row's pre-head feature."""

    def __init__(self, degree, ridge):
        self.degree = degree
        self.ridge = ridge
        self.count = 0
        self.gram = torch.zeros(degree + 1, degree + 1, dtype=torch.float64)
        self.rhs = None
        self.last = self.prev = None

    def ready(self):
        return self.count > self.degree

    def update(self, t, feature):
        basis = chebyshev(t, self.degree)
        flat = feature.flatten().float()
        if self.rhs is None:
            self.rhs = torch.zeros(self.degree + 1, flat.numel(), device=flat.device)
        self.rhs.addmm_(basis.to(flat.device, torch.float32)[:, None], flat[None])
        self.gram += torch.outer(basis, basis)
        self.count += 1
        self.prev, self.last = self.last, (t, feature.clone())

    def predict(self, t, blend):
        last_t, last = self.last
        lhs = self.gram + self.ridge * torch.eye(self.degree + 1, dtype=torch.float64)
        weights = torch.linalg.solve(lhs, chebyshev(t, self.degree)).to(self.rhs.device, torch.float32)
        out = (weights @ self.rhs).view(last.shape)
        if blend < 1.0:
            prev_t, prev = self.prev
            last, prev = last.float(), prev.float()
            linear = last + (t - last_t) / (last_t - prev_t) * (last - prev)
            out = blend * out + (1.0 - blend) * linear
        return out.to(self.last[1].dtype)


class Spectrum:
    def __init__(self, blend, degree, ridge, window, flex_window, warmup, tail):
        self.blend = blend
        self.degree = degree
        self.ridge = ridge
        self.window = window
        self.flex_window = flex_window
        self.warmup = warmup
        self.tail = tail
        self.active = False
        self.enabled = False
        self.forecasters = {}

    def run_start(self, sampler, sigmas):
        name = getattr(getattr(sampler, "sampler_function", None), "__name__", type(sampler).__name__)
        self.forecasters = {}
        self.sigmas = sigmas.tolist()
        self.total = len(self.sigmas) - 1
        self.enabled = name in SAMPLERS and self.total > 1
        self.step = 0
        self.recorded = 0
        self.cached_run = 0
        self.curr_window = self.window
        self.forecast_steps = 0
        if not self.enabled:
            LOG.warning("Spectrum: sampler %s is not supported, running normally.", name)

    def run_end(self):
        if self.enabled:
            LOG.info("Spectrum: forecast %d of %d steps.", self.forecast_steps, self.total)
        self.forecasters = {}

    def step_start(self, sigma):
        first, last = self.sigmas[0], self.sigmas[-2]
        self.coord = (sigma - first) / (last - first) * 2.0 - 1.0
        self.forecasting = (
            self.step >= self.warmup
            and self.step < self.total - self.tail
            and self.recorded > self.degree
            and (self.cached_run + 1) % max(1, int(self.curr_window)) != 0
        )
        self.active = True

    def step_end(self):
        self.active = False
        self.step += 1
        if self.forecasting:
            self.cached_run += 1
            self.forecast_steps += 1
            return
        self.cached_run = 0
        self.recorded += 1
        if self.step > self.warmup:
            self.curr_window += self.flex_window

    def model_start(self, uuids, x):
        per_chunk = x.shape[0] // len(uuids)
        self.keys = [(uuid, i) for uuid in uuids for i in range(per_chunk)]
        self.tokens = x.shape[2] * x.shape[3]
        self.row = 0

    def block(self, index, last_index):
        def run(args, extra):
            if not self.active:
                return extra["original_block"](args)
            if index == 0:
                self.key = self.keys[self.row]
                self.row += 1
                forecaster = self.forecasters.get(self.key)
                self.cached = self.forecasting and forecaster is not None and forecaster.ready()
                if self.cached:
                    return {"img": forecaster.predict(self.coord, self.blend)}
            elif self.cached:
                return {"img": args["img"]}
            out = extra["original_block"](args)
            if index == last_index:
                forecaster = self.forecasters.setdefault(self.key, Forecaster(self.degree, self.ridge))
                forecaster.update(self.coord, out["img"][:, : self.tokens])
            return out

        return run


def patch(model, **settings):
    diffusion_model = model.model.diffusion_model
    if type(diffusion_model).__name__ != "MOSSImageTransformer":
        raise ValueError("Spectrum MOSS requires a MOSS-Image model.")
    model = model.clone()
    spectrum = Spectrum(**settings)

    # Wrappers are plain closures so model option copies keep sharing one Spectrum.
    def sampler_sample(executor, guider, sigmas, *args, **kwargs):
        spectrum.run_start(executor.class_obj, sigmas)
        try:
            return executor(guider, sigmas, *args, **kwargs)
        finally:
            spectrum.run_end()

    def predict_noise(executor, x, timestep, *args, **kwargs):
        if not spectrum.enabled:
            return executor(x, timestep, *args, **kwargs)
        spectrum.step_start(timestep.flatten()[0].item())
        try:
            return executor(x, timestep, *args, **kwargs)
        finally:
            spectrum.step_end()

    def apply_model(executor, x, t, c_concat, c_crossattn, control, transformer_options, **kwargs):
        if spectrum.active:
            spectrum.model_start(transformer_options["uuids"], x)
        return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    for wrapper_type, wrapper in (
        (patcher_extension.WrappersMP.SAMPLER_SAMPLE, sampler_sample),
        (patcher_extension.WrappersMP.PREDICT_NOISE, predict_noise),
        (patcher_extension.WrappersMP.APPLY_MODEL, apply_model),
    ):
        patcher_extension.add_wrapper_with_key(wrapper_type, "spectrum", wrapper, model.model_options, is_model_options=True)

    last_index = len(diffusion_model.layers) - 1
    for index in range(last_index + 1):
        model.set_model_patch_replace(spectrum.block(index, last_index), "dit", "single_block", index)
    return model
