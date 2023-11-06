from typing import Dict, Set

import tvm
import tvm.dlight as dl

from dlight_bench import DlightBench
from models import stable_diffusion_xl


def factorized(factor: int, minimum: int):
    """Factorized dynamic shape variable sample function factory."""

    def sample_dym_var_sequential(
        dym_vars: Set[str], sample_idx: int, _: int
    ) -> Dict[str, int]:
        """
        Sequential dynamic shape variable sample function.
        Sample a sequential value for each dynamic shape variable.

        Parameters
        ----------
        dym_vars : Set[str]
            Dynamic shape variable set, e.g., {"n", "m"}
        sample_idx : int
            Sample index denotes the index the function is called for the same
            dynamic shape variable dictionary & function.
        sample_num : int
            Sample number denotes the total number of samples.

        Returns
        -------
        result : Dict[str, int]
            Dynamic shape variable sample, e.g., {"n": 64, "m": 128}
        """
        results = {}
        cnt = 1
        for var in dym_vars:
            results[var] = 2 ** (sample_idx // cnt % factor + minimum)
            cnt *= factor
        return results

    return sample_dym_var_sequential


with tvm.target.Target("apple/m1-gpu"):
    DlightBench.benchmark(
        "llama_2_batch",
        category="Matmul",
        passes=[dl.ApplyDefaultSchedule(dl.gpu.Matmul())],
        sample_func=factorized(5, 5),
        sample_num_per_func=10,
    )
    DlightBench.benchmark(
        "llama_2_batch",
        category="GEMV",
        passes=[dl.ApplyDefaultSchedule(dl.gpu.GEMV())],
        sample_func=factorized(5, 5),
        sample_num_per_func=10,
    )
    DlightBench.benchmark(
        "llama_2_batch",
        category="Reduction",
        passes=[dl.ApplyDefaultSchedule(dl.gpu.Reduction())],
        sample_func=factorized(5, 5),
        sample_num_per_func=10,
    )
