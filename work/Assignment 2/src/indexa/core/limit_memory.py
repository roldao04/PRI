"""Memory monitor - DISABLED for Assignment 2.

This module has been gutted to remove the 2GB memory constraint from Assignment 1.
All functions are now no-ops to maintain compatibility with existing code.
"""

import logging

logger = logging.getLogger(__name__)


def start_memory_monitor(show_memory_updates: bool = False):
    """No-op: Memory monitoring disabled for Assignment 2.

    This function previously monitored memory usage and crashed the program
    if it exceeded 2GB. For Assignment 2, we don't need this constraint.
    """
    logger.info("[MemoryGuard] Memory monitoring DISABLED (Assignment 2 - no memory constraints)")


def get_current_memory_usage_in_mb() -> int:
    """No-op: Returns 0 (monitoring disabled)."""
    return 0
