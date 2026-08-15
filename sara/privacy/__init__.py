"""ASTRO-shaped permission kernel: grants, vault, audit, on-device learn.

Default-deny. Preview+approve before camera/mic/files/mesh/cloud/learn.
High-risk tools (shell, file_write, web, cloud, mesh share_raw) need a grant.
"""

from .grants import GrantLedger, CAPABILITIES, HIGH_RISK_TOOLS, GrantError
from .vault import Vault
from .audit import AuditLog
from .learn import LocalLearner

__all__ = [
    "GrantLedger",
    "CAPABILITIES",
    "HIGH_RISK_TOOLS",
    "GrantError",
    "Vault",
    "AuditLog",
    "LocalLearner",
]
