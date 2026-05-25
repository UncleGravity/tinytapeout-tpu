"""PL-resident kernels. Empty unless a bitstream is loaded.

Use ``register_all(registry, overlay)`` after the PS kernels to install
PL replacements. Any name collision (e.g. ``GOP_COPY``) overrides the
PS kernel — that's the intended swap-point.
"""

from board.kernels.pl.loopback import register_all  # noqa: F401
