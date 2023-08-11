from typing import Dict, Set

import tvm
import tvm.dlight as dl

from dlight_bench import DlightBench


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


with tvm.target.Target("nvidia/geforce-rtx-3070"):
    DlightBench.benchmark(
        "vicuna_v1_7b_fp16",
        func_names=["matmul"],
        passes=[tvm.tir.transform.DefaultGPUSchedule()],
        sample_func=factorized(5, 5),
        sample_num_per_func=10,
    )
    DlightBench.benchmark(
        "vicuna_v1_7b_fp16",
        func_names=["matmul"],
        passes=[dl.ApplyDefaultSchedule(dl.gpu.Fallback())],
        sample_func=factorized(5, 5),
        sample_num_per_func=10,
    )
    # DlightBench.benchmark(
    #     "vicuna_v1_7b_fp16",
    #     category="Fallback",
    #     passes=[dl.ApplyDefaultSchedule(dl.gpu.Fallback())],
    #     sample_func=factorized(5, 5),
    #     sample_num_per_func=10,
    # )
