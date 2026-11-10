"""Re-run LanceOtron (the peak caller) on a bigWig and locate its output.

LanceOtron needs its own module environment - a different Python + TensorFlow
stack from the rest of this package (see `module load lanceotron` on this
cluster, which pulls in its own python-base, cuda, cuDNN and tensorflow2-gpu).
It is always invoked as an isolated subprocess rather than imported, so this
module only wraps that subprocess call.
"""
import subprocess
from pathlib import Path

LANCEOTRON_MODULE = "lanceotron"


def call_peaks(bigwig_path: Path, output_dir: Path, peak_score_cutoff: float = 0.0) -> Path:
    """Run `lanceotron callPeaks` on bigwig_path, returning the resulting peak-call bed file.

    Runs in a clean shell (module purge + module load lanceotron, PYTHONPATH
    unset) so LanceOtron's module environment never mixes with the
    python-cbrg environment the rest of this package runs under.

    LanceOtron derives its own output filename from the input bigWig's
    basename split on the FIRST "." (not Python's Path.stem, which only
    strips the final suffix) - see `find_and_score_peaks` in LanceOtron
    1.2.7's modules.py: `bigwig_file.split("/")[-1].split(".")[0] +
    "_L-tron.bed"`. For a single-dot filename like "Mast.bw" this is the
    same as the stem ("Mast"), but for a multi-dot filename like
    "Mast.down_sample.1.0.0.bw" (as produced by this repo's degradation
    benchmark) it truncates to just "Mast" - confirmed by both reading
    LanceOtron's source and comparing real bigwig/bed filename pairs in
    the pipeline's existing collections (which are all single-dot, so this
    was previously latent). We therefore predict the exact expected
    filename using LanceOtron's own convention rather than Path(...).stem,
    falling back to a directory-wide glob only as a safety net against
    LanceOtron version drift.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = (
        "unset PYTHONPATH; "
        "module purge >/dev/null 2>&1 || true; "
        f"module load {LANCEOTRON_MODULE} >/dev/null 2>&1; "
        # Two env vars, two separate confirmed bugs in this environment:
        # 1. ptxas 12.3.103 crashes XLA with SIGABRT on every GPU tried (TITAN
        #    RTX/gpu-turing, L40S/gpu-ada). TF_XLA_FLAGS=--tf_xla_auto_jit=0
        #    alone did NOT fix it (confirmed: identical crash - the bug is a
        #    hardcoded ptxas-version check inside XLA itself, not limited to
        #    auto-clustered ops). XLA_FLAGS=--xla_gpu_autotune_level=0 (avoids
        #    triggering ptxas via kernel autotuning) CONFIRMED FIXED this -
        #    the crash is gone entirely with it set.
        # 2. With (1) fixed, hit a second, previously-masked error: "libdevice
        #    not found at ./libdevice.10.bc" - this cluster's CUDA 12.3 (what
        #    tensorflow2-gpu/2.16.1 + ptxas 12.3.103 actually targets) has no
        #    standalone install under /package/cuda/, and no pip-based
        #    nvidia-cuda-nvcc wheel either. Pointed --xla_gpu_cuda_data_dir at
        #    /package/cuda/11.8 instead (the newest standalone CUDA actually
        #    present, confirmed to contain nvvm/libdevice/libdevice.10.bc) -
        #    libdevice.10.bc is typically stable enough across CUDA versions
        #    for this to work as a substitute, but this exact combination is
        #    UNVERIFIED end-to-end - next thing to test.
        "export TF_XLA_FLAGS=--tf_xla_auto_jit=0; "
        "export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=/package/cuda/11.8'; "
        f"lanceotron callPeaks '{bigwig_path}' --format Web -c {peak_score_cutoff} -f '{output_dir}'"
    )
    subprocess.run(["bash", "-lc", command], check=True)

    expected_name = Path(bigwig_path).name.split(".")[0] + "_L-tron.bed"
    expected_path = output_dir / expected_name
    if expected_path.exists():
        return expected_path

    candidates = sorted(output_dir.glob("*_L-tron.bed"))
    if not candidates:
        raise FileNotFoundError(
            f"LanceOtron produced no peak-call bed file in {output_dir} for {Path(bigwig_path).name}"
        )
    return candidates[0]
