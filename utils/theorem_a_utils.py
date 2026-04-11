from typing import Any


def summarize_theorem_a_trace(trace: dict[str, Any]) -> dict[str, float]:
    layer_metrics = trace["layer_metrics"]
    return {
        "E_kernel": trace["kernel_err"],
        "E_roll_all": trace["roll_err_all"],
        "E_roll_train": trace["roll_err_train"],
        "E_roll_test": trace["roll_err_test"],
        "E_exact_max": max(m["exact_err"] for m in layer_metrics),
        "E_local_max": max(m["local_err"] for m in layer_metrics),
        "E_span_max": max(m["span_err"] for m in layer_metrics),
        "E_value_align_max": max(m["value_align_err"] for m in layer_metrics),
        "chi_v": trace["chi_v"],
        "alpha_v": trace["alpha_v"],
    }