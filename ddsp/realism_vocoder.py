from .realism import VocalRealismAdapter
from .vocoder import CombSubSuperFast


class CombSubSuperFastRealism(CombSubSuperFast):
    """Compatibility wrapper that injects the R0 realism residuals.

    Keeping the upstream synthesizer untouched makes it easy to rebase future
    DDSP-SVC changes. With realism disabled, forward delegates directly to the
    original implementation with no additional numerical operations.
    """

    def __init__(self, *args, realism_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        realism_config = realism_config or {}
        self.realism_enabled = bool(realism_config.get("enabled", False))
        self.realism_strength = float(realism_config.get("strength", 1.0))
        self.realism = None
        if self.realism_enabled:
            n_unit = kwargs.get("n_unit")
            if n_unit is None and len(args) >= 4:
                n_unit = args[3]
            if n_unit is None:
                n_unit = 256
            self.realism = VocalRealismAdapter(
                n_unit=int(n_unit),
                hidden_channels=int(realism_config.get("hidden_channels", 128)),
                num_layers=int(realism_config.get("num_layers", 2)),
                max_f0_cents=float(realism_config.get("max_f0_cents", 35.0)),
                max_volume_db=float(realism_config.get("max_volume_db", 3.0)),
            )

    def forward(
        self,
        units_frames,
        f0_frames,
        volume_frames,
        spk_id=None,
        spk_mix_dict=None,
        aug_shift=None,
        initial_phase=None,
        infer=True,
        **kwargs,
    ):
        if self.realism is not None:
            f0_frames, volume_frames, _ = self.realism(
                units_frames,
                f0_frames,
                volume_frames,
                strength=self.realism_strength,
            )
        return super().forward(
            units_frames,
            f0_frames,
            volume_frames,
            spk_id=spk_id,
            spk_mix_dict=spk_mix_dict,
            aug_shift=aug_shift,
            initial_phase=initial_phase,
            infer=infer,
            **kwargs,
        )
