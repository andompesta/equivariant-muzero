from dataclasses import dataclass

@dataclass
class ReanalyzerConfig:
    # Reanalyze (See paper appendix Reanalyse)
    use_last_model_value: bool
    reanalyse_on_gpu: bool
