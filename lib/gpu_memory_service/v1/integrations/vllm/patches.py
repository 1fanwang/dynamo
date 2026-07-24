# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confine GMS V1 to vLLM's normal model loader."""

from __future__ import annotations

from typing import Any

from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from ...client.torch import SnapshotTorchPool


def install_model_loader_patch(pool: SnapshotTorchPool) -> None:
    """Run the normal vLLM model loader in the GMS model-load pool."""
    original_load_model = BaseModelLoader.load_model

    def load_model(loader: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            with pool.model_load_pool():
                model = original_load_model(loader, *args, **kwargs)
        except Exception as cause:
            pool.abort_model_load(cause)
            raise
        pool.finalize_model_load(model)
        return model

    BaseModelLoader.load_model = load_model
