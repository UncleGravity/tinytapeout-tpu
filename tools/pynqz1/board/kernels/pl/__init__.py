"""PL-resident kernels. Empty unless a matching bitstream is loaded.

Use ``register_all(registry, overlay)`` after the PS kernels to install
PL replacements. Any name collision (e.g. ``GOP_COPY``) overrides the
PS kernel — that's the intended swap-point.
"""

from board.kernels.pl import loopback, matmul_q1a8
from board.kernels.registry import KernelRegistry


def register_all(registry: KernelRegistry, overlay) -> None:
    """Register PL kernels exposed by ``overlay``.

    Different bring-up bitstreams expose different modules. Registering by
    presence keeps a matmul-only overlay from accidentally replacing COPY with
    the loopback driver.
    """
    if _overlay_has(overlay, "axis_loopback_0"):
        registry.register(loopback.PLLoopback(overlay))
    if _overlay_has(overlay, "q1a8_kernel_top_0"):
        registry.register(matmul_q1a8.PLMatmulQ1A8(overlay))


def _overlay_has(overlay, name: str) -> bool:
    ip_dict = getattr(overlay, "ip_dict", None)
    if isinstance(ip_dict, dict) and name in ip_dict:
        return True
    return hasattr(overlay, name)
