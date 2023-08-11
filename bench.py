"""The model benchmark for dlight."""
from typing import TYPE_CHECKING, Dict, List, Callable, Union, Optional, Literal

import tvm
from tvm.tir import PrimFunc
from tvm.ir import IRModule
from tvm.ir.transform import ModulePass

from tvm import dlight as dl
from tvm.dlight.benchmark import benchmark_prim_func
from tvm.dlight.benchmark.utils import random_dym_var_sample_func

CATEGORY = Literal[
    "Reduction", "GEMV", "Fallback", "Matmul", "Transpose", "GeneralReduction"
]

if TYPE_CHECKING:
    from tvm.meta_schedule.runner import RPCConfig, EvaluatorConfig


class DlightBench:
    """The class to register benchmark workloads and run benchmark.

    Parameters
    ----------
    workloads : Dict[str, Dict[str, PrimFunc]]
        The dictionary of benchmark workloads.
    """

    workloads: Dict[str, Dict[str, PrimFunc]] = {}

    @staticmethod
    def register_bench_workload(
        mod_or_func: Union[IRModule, PrimFunc], model_name: str, func_name: str
    ):
        """Register a benchmark workload.

        Parameters
        ----------
        mod_or_func : Union[IRModule, PrimFunc]
            The IRModule or PrimFunc to be registered.
        workload_name : str
            The workload name.
        func_name : str
            The function name.
        """
        if isinstance(mod_or_func, IRModule):
            func = mod_or_func[func_name]
        elif isinstance(mod_or_func, PrimFunc):
            func = mod_or_func
        else:
            raise TypeError("Unsupported type: " + str(type(mod_or_func)))

        if model_name not in DlightBench.workloads:
            DlightBench.workloads[model_name] = {}
        # if func_name in DlightBench.workloads[model_name]:
        #     raise ValueError("Workload already registered: " + func_name)

        DlightBench.workloads[model_name][func_name] = func

    @staticmethod
    def benchmark(
        model_name: str,
        *,
        passes: List[
            Union[ModulePass, Callable]  # pylint: disable=used-before-assignment
        ],
        func_names: Optional[List[str]] = None,
        sample_func: Optional[
            Callable[[Dict[str, str], int, int], Dict[str, int]]
        ] = None,
        category: Optional[CATEGORY] = None,
        target: Optional[tvm.target.Target] = None,
        sample_num_per_func: int = 5,
        evaluator_config: Optional["EvaluatorConfig"] = None,
        rpc_config: Optional["RPCConfig"] = None,
    ) -> None:
        """Run benchmark.

        Parameters
        ----------
        model_name : str
            The model name.
        passes : List[Union[ModulePass, Callable]]
            The passes to be applied to the PrimFuncs.
        func_names : Optional[List[str]]
            The list of function names to be benchmarked, if None, benchmark
            all functions in the model.
        sample_func : Optional[Callable[[Dict[str, str], int, int], Dict[str, int]]]
            The function to sample dynamic shape variables, if None, use the
            default function.
        category : Optional[CATEGORY]
            Filter out the workloads with the specified category, if None, do
            not filter.
        target : Optional[tvm.target.Target]
            The target to run benchmark, if None, use the current target.
        sample_num_per_func : int
            The number of samples per function, default is 5.
        evaluator_config : Optional["EvaluatorConfig"]
            The evaluator config, if None, use the default config.
        rpc_config : Optional["RPCConfig"]
            The RPC config, if None, use the default config.
        """

        def is_category(
            func: PrimFunc, category: CATEGORY, target: tvm.target.Target
        ) -> bool:
            rules = {
                "Reduction": dl.gpu.Reduction(),
                "GEMV": dl.gpu.GEMV(),
                "Fallback": dl.gpu.Fallback(),
                "Matmul": dl.gpu.Matmul(),
                "Transpose": dl.gpu.Transpose(),
                "GeneralReduction": dl.gpu.GeneralReduction(),
            }
            if category in rules:
                try:
                    mod = IRModule.from_expr(func)
                    with tvm.transform.PassContext(opt_level=3):
                        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                            rules[category]
                        )(mod)
                    tvm.build(mod, target=target)
                except Exception:  # pylint: disable=broad-except
                    return False
                return True
            else:
                raise ValueError("Unsupported category: " + category)

        if sample_func is None:
            sample_func = random_dym_var_sample_func
        if target is None:
            target = tvm.target.Target.current()
            assert target is not None, "No target specified."
        if model_name not in DlightBench.workloads:
            raise ValueError("Model not registered: " + model_name)
        if func_names is None:
            func_names = [func_name for func_name in DlightBench.workloads[model_name]]

        # Run benchmark
        print("Model:", model_name)
        print("Target:", target)
        print("Category:", category if category is not None else "All", end="\n\n")

        for func_name in func_names:
            func = DlightBench.workloads[model_name][func_name]
            if category is not None:
                if not is_category(func, category, target):
                    continue
            print(f"Benchmarking {func_name}:")
            mod = IRModule.from_expr(func)
            for pass_ in passes:
                print("Applying pass:", pass_)
            mod = tvm.transform.Sequential(passes, opt_level=3)(mod)
            benchmark_prim_func(
                mod,
                dym_var_sample_func=sample_func,
                sample_num=sample_num_per_func,
                target=target,
                prim_func_name=func_name,
                evaluator_config=evaluator_config,
                rpc_config=rpc_config,
                drop_cols=["Weight", "WxTime(ms)", "Std(us)", "PrimFunc"],
            )
            print()
