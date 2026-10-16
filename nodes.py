from comfy_api.latest import ComfyExtension, io

from .spectrum import patch


class SpectrumMOSSImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SpectrumMOSSImage",
            display_name="Spectrum MOSS-Image",
            category="sampling/spectrum",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("blend_weight", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("degree", default=4, min=1, max=16),
                io.Float.Input("ridge_lambda", default=0.1, min=0.0, max=10.0, step=0.01),
                io.Float.Input("window_size", default=2.0, min=1.0, max=16.0, step=0.05),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=8.0, step=0.05),
                io.Int.Input("warmup_steps", default=5, min=0, max=32),
                io.Int.Input("tail_actual_steps", default=3, min=0, max=32),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, blend_weight, degree, ridge_lambda, window_size, flex_window, warmup_steps, tail_actual_steps):
        return io.NodeOutput(
            patch(
                model,
                blend=blend_weight,
                degree=degree,
                ridge=ridge_lambda,
                window=window_size,
                flex_window=flex_window,
                warmup=warmup_steps,
                tail=tail_actual_steps,
            )
        )


class SpectrumMOSSExtension(ComfyExtension):
    async def get_node_list(self):
        return [SpectrumMOSSImage]


async def comfy_entrypoint():
    return SpectrumMOSSExtension()
