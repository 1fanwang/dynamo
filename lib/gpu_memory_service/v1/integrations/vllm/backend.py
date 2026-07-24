# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-engine snapshot lifecycle for the experimental GMS V1 worker."""

from __future__ import annotations

import logging

import torch
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackendFactory,
)

from ...client.memory_manager import SnapshotMemoryManager
from ...client.rpc import AllocationClient
from ...client.torch import SnapshotTorchPool
from .patches import install_model_loader_patch

BACKEND_NAME = "gms-v1-snapshot"
logger = logging.getLogger(__name__)


class GMSV1SleepModeBackend(CuMemBackend):
    """Compose native KV-cache sleep with GMS V1 parameter sleep."""

    def __init__(self) -> None:
        super().__init__()
        device = torch.cuda.current_device()
        client = AllocationClient(get_socket_path(device, "snapshot-v1"))
        try:
            manager = SnapshotMemoryManager(client, get_vmm(), device)
            pool = SnapshotTorchPool(manager)
            install_model_loader_patch(pool)
        except BaseException:
            client.close()
            raise
        self._client = client
        self._manager = manager
        self._pool = pool

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")

        try:
            super().suspend(level)
            self._pool.prepare_snapshot()
        except Exception as cause:
            logger.exception("GMS V1 suspend failed; terminating the worker process")
            raise SystemExit(1) from cause

    def resume(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
        if self._state != "SUSPENDED":
            raise RuntimeError(f"cannot resume GMS V1 from {self._state}")

        try:
            self._state = "RESUMING"
            self._manager.wake()
            super().resume(tags)
        except Exception as cause:
            logger.exception("GMS V1 resume failed; terminating the worker process")
            raise SystemExit(1) from cause


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
