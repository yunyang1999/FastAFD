"""JIT loader for CUDA kernels vendored from sglang 0.5.18 (``sglang/kernels/jit``).

The ``include/sgl_kernel`` headers and the kernel sources under ``csrc/`` are
copied unchanged from sglang (Apache-2.0, see NOTICE) so FastAFD runs the same
decode kernels as the serving stack. A module is one generated translation unit
that includes the requested sources and exports the requested entry points via
tvm-ffi (the same wrapper sglang generates); it is compiled with nvcc for the
local GPU on first use and cached under ``$MINISGL_JIT_CACHE_DIR`` (default
``~/.cache/minisgl/sgl_jit``), keyed by a hash of the sources, flags and toolkit.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import torch

_ROOT = Path(__file__).resolve().parent
INCLUDE_DIR = _ROOT / "include"
CSRC_DIR = _ROOT / "csrc"

_FFI_INCLUDES = (
    "#include <tvm/ffi/container/tensor.h>",
    "#include <tvm/ffi/dtype.h>",
    "#include <tvm/ffi/error.h>",
    "#include <tvm/ffi/extra/c_env_api.h>",
    "#include <tvm/ffi/function.h>",
)

CPP_DTYPE = {
    torch.float32: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
}


def cpp_args(*args) -> str:
    """Python values -> C++ template argument list."""
    out = []
    for a in args:
        if isinstance(a, bool):
            out.append("true" if a else "false")
        elif isinstance(a, torch.dtype):
            out.append(CPP_DTYPE[a])
        else:
            out.append(str(a))
    return ", ".join(out)


@functools.cache
def _cuda_home() -> Path:
    for key in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(key):
            return Path(os.environ[key])
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return Path("/usr/local/cuda")


@functools.cache
def _arch() -> tuple[int, int]:
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def supports_pdl() -> bool:
    return _arch()[0] >= 9


def _target_flags() -> list[str]:
    major, minor = _arch()
    suffix = "a" if major >= 9 else ""
    cc = f"{major}{minor}{suffix}"
    return [
        f"-DSGL_CUDA_ARCH={major * 100 + minor * 10}",
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
    ]


@functools.cache
def _tree_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(list(INCLUDE_DIR.rglob("*")) + list(CSRC_DIR.rglob("*"))):
        if p.is_file():
            h.update(str(p.relative_to(_ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@functools.cache
def _toolchain_id() -> str:
    nvcc = _cuda_home() / "bin" / "nvcc"
    try:
        ver = subprocess.check_output([str(nvcc), "--version"], text=True)
    except Exception:  # pragma: no cover - surfaced at build time
        ver = "unknown"
    import tvm_ffi

    return f"{ver}|tvm_ffi={getattr(tvm_ffi, '__version__', '?')}"


def _cache_dir() -> Path:
    base = os.environ.get("MINISGL_JIT_CACHE_DIR")
    return Path(base) if base else Path.home() / ".cache" / "minisgl" / "sgl_jit"


def _wrapper_source(files: Sequence[str], wrappers: Sequence[tuple[str, str]]) -> str:
    lines = list(_FFI_INCLUDES)
    lines += [f'#include "{(CSRC_DIR / f).as_posix()}"' for f in files]
    lines.append("namespace sglang {")
    lines += [f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({name}, ({kernel}));" for name, kernel in wrappers]
    lines.append("}  // namespace sglang")
    return "\n".join(lines) + "\n"


def _build(name: str, source: str, cuda_flags: list[str], ld_flags: list[str], out_dir: Path) -> Path:
    from tvm_ffi import libinfo

    cuda_home = _cuda_home()
    nvcc = cuda_home / "bin" / "nvcc"
    lib = Path(libinfo.find_libtvm_ffi())
    staging = out_dir.parent / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        src = staging / "cuda.cu"
        src.write_text(source)
        obj = staging / "cuda.o"
        so = staging / f"{name}.so"
        includes = [libinfo.find_include_path(), libinfo.find_dlpack_include_path(), str(INCLUDE_DIR)]
        cmd = [str(nvcc), "-Xcompiler", "-fPIC", *cuda_flags, *[f"-I{p}" for p in includes], "-c", str(src), "-o", str(obj)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed for {name}:\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        cxx = os.environ.get("CXX", "c++")
        link = [cxx, str(obj), "-shared", f"-L{lib.parent}", "-ltvm_ffi", f"-L{cuda_home / 'lib64'}", "-lcudart", *ld_flags, "-o", str(so)]
        r = subprocess.run(link, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"link failed for {name}:\n{' '.join(link)}\n{r.stdout}\n{r.stderr}")
        try:
            os.rename(staging, out_dir)  # atomic publish
        except OSError:
            if not (out_dir / f"{name}.so").exists():
                raise
        return out_dir / f"{name}.so"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_jit(
    *args,
    cuda_files: Iterable[str],
    cuda_wrappers: Iterable[tuple[str, str]],
    extra_cuda_cflags: Iterable[str] = (),
    extra_ldflags: Iterable[str] = (),
):
    """Compile (once) and load a tvm-ffi module exporting ``cuda_wrappers``.

    ``args`` name the module variant (e.g. template arguments); ``cuda_files``
    are paths relative to ``csrc/``; ``cuda_wrappers`` are (export, C++ entry)
    pairs as in sglang's ``load_jit``.
    """
    from tvm_ffi import load_module

    name = "sgl_" + "_".join(str(a).replace(", ", "_").replace(" ", "") for a in args)
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    files = list(cuda_files)
    wrappers = list(cuda_wrappers)
    flags = ["-std=c++20", "-O3", "--expt-relaxed-constexpr", *_target_flags(), *extra_cuda_cflags]
    source = _wrapper_source(files, wrappers)
    key = hashlib.sha256(
        "\n".join([source, " ".join(flags), " ".join(extra_ldflags), _tree_digest(), _toolchain_id()]).encode()
    ).hexdigest()[:16]
    out_dir = _cache_dir() / f"{name}-{key}"
    so = out_dir / f"{name}.so"
    if not so.exists():
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(out_dir.parent / f".{name}.lock", "w") as lk:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            if not so.exists():
                so = _build(name, source, flags, list(extra_ldflags), out_dir)
    return load_module(str(so))


__all__ = ["load_jit", "cpp_args", "supports_pdl", "INCLUDE_DIR", "CSRC_DIR"]
