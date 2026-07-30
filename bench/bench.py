"""mojo-healpy against healpy on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import healpy as hp  # noqa: E402
import mojo_healpy as mh  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def gpu_memory_free_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return min(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def main():
    rng = np.random.default_rng(2026)
    cases = []

    nside = 512
    count = 2_000_000
    theta = np.ascontiguousarray(np.arccos(rng.uniform(-1.0, 1.0, count)))
    phi = np.ascontiguousarray(rng.uniform(-8.0, 8.0, count))
    pixels = np.ascontiguousarray(rng.integers(0, hp.nside2npix(nside), count))
    cases.append(
        (
            "ang2pix, 2M directions",
            lambda: mh.ang2pix(nside, theta, phi),
            lambda: hp.ang2pix(nside, theta, phi),
            5,
        )
    )
    cases.append(
        (
            "pix2ang, 2M pixels",
            lambda: mh.pix2ang(nside, pixels),
            lambda: hp.pix2ang(nside, pixels),
            5,
        )
    )
    cases.append(
        (
            "ring2nest, 2M pixels",
            lambda: mh.ring2nest(nside, pixels),
            lambda: hp.ring2nest(nside, pixels),
            5,
        )
    )

    lmax = 2048
    alm = np.ascontiguousarray(
        rng.normal(size=mh.Alm.getsize(lmax))
        + 1j * rng.normal(size=mh.Alm.getsize(lmax))
    )
    transfer = np.ascontiguousarray(np.exp(-np.arange(lmax + 1) / 1000.0))
    cases.append(
        (
            "almxfl, lmax=2048",
            lambda: mh.almxfl(alm, transfer),
            lambda: hp.almxfl(alm, transfer),
            5,
        )
    )
    cases.append(
        (
            "alm2cl, lmax=2048",
            lambda: mh.alm2cl(alm),
            lambda: hp.alm2cl(alm),
            5,
        )
    )

    transform_nside, transform_lmax = 32, 32
    small_alm = np.ascontiguousarray(
        rng.normal(size=mh.Alm.getsize(transform_lmax))
        + 1j * rng.normal(size=mh.Alm.getsize(transform_lmax))
    )
    sky = hp.alm2map(small_alm, transform_nside, lmax=transform_lmax)
    cases.append(
        (
            "alm2map, nside=32 lmax=32",
            lambda: mh.alm2map(
                small_alm, transform_nside, lmax=transform_lmax
            ),
            lambda: hp.alm2map(
                small_alm, transform_nside, lmax=transform_lmax
            ),
            3,
        )
    )
    if os.environ.get("MOJO_HEALPY_BENCH_GPU") == "1":
        free_mib = gpu_memory_free_mib()
        if free_mib >= 4000:
            cases.append(
                (
                    "alm2map GPU, nside=32 lmax=32",
                    lambda: mh.alm2map(
                        small_alm,
                        transform_nside,
                        lmax=transform_lmax,
                        device="gpu",
                    ),
                    lambda: hp.alm2map(
                        small_alm, transform_nside, lmax=transform_lmax
                    ),
                    3,
                )
            )
        else:
            print(
                "GPU benchmark skipped: "
                f"{free_mib} MiB free is below the 4000 MiB safety threshold."
            )
    cases.append(
        (
            "map2alm, nside=32 lmax=32 iter=0",
            lambda: mh.map2alm(sky, lmax=transform_lmax, iter=0),
            lambda: hp.map2alm(sky, lmax=transform_lmax, iter=0),
            3,
        )
    )

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print()
    print("| kernel | mojo-healpy | healpy | healpy / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream, repeat in cases:
        ours()
        upstream()
        mojo_time = timeit(ours, repeat)
        healpy_time = timeit(upstream, repeat)
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{healpy_time * 1e3:.3f} ms | {healpy_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
