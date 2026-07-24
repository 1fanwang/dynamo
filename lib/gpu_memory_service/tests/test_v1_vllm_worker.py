# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


@pytest.fixture(scope="module")
def vllm_modules():
    pytest.importorskip("vllm.device_allocator.sleep_mode_backend")
    loader_module = pytest.importorskip("vllm.model_executor.model_loader.base_loader")
    pytest.importorskip("vllm.v1.worker.gpu_worker")
    backend = importlib.import_module("gpu_memory_service.v1.integrations.vllm.backend")
    patches = importlib.import_module("gpu_memory_service.v1.integrations.vllm.patches")
    worker = importlib.import_module("gpu_memory_service.v1.integrations.vllm.worker")
    return backend, loader_module.BaseModelLoader, patches, worker


def test_worker_selects_and_eagerly_constructs_backend_after_device_init(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, _base_loader, _patches, worker_module = vllm_modules
    events = []
    backend_instance = object()

    def upstream_init(instance) -> None:
        events.append("upstream_init")
        instance.device = torch.device("cuda:3")

    def get_backend(instance):
        events.append(("get_backend", instance.device))
        return backend_instance

    @contextmanager
    def native_pool(tag):
        events.append(("native_pool", tag))
        yield

    monkeypatch.setattr(worker_module.Worker, "init_device", upstream_init)
    monkeypatch.setattr(worker_module.Worker, "_get_sleep_mode_backend", get_backend)
    monkeypatch.setattr(
        worker_module.Worker,
        "_maybe_get_memory_pool_context",
        lambda _instance, tag: native_pool(tag),
    )

    worker = object.__new__(worker_module.GMSV1Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            sleep_mode_backend="cumem",
        )
    )
    worker.init_device()

    assert worker.vllm_config.model_config.sleep_mode_backend == backend.BACKEND_NAME
    with worker._maybe_get_memory_pool_context("weights"):
        events.append("weights_scope")
    with worker._maybe_get_memory_pool_context("kv_cache"):
        events.append("kv_cache_scope")
    assert events == [
        "upstream_init",
        ("get_backend", torch.device("cuda:3")),
        "weights_scope",
        ("native_pool", "kv_cache"),
        "kv_cache_scope",
    ]


def test_worker_requires_sleep_mode_before_device_init(
    vllm_modules,
    monkeypatch,
) -> None:
    _backend, _base_loader, _patches, worker_module = vllm_modules
    monkeypatch.setattr(
        worker_module.Worker,
        "init_device",
        lambda _instance: pytest.fail("device initialization must not run"),
    )
    worker = object.__new__(worker_module.GMSV1Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            sleep_mode_backend="cumem",
        )
    )

    with pytest.raises(RuntimeError, match="requires vLLM sleep mode"):
        worker.init_device()


def test_backend_owns_gms_resources_and_composes_native_kv_lifecycle(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, _base_loader, _patches, _worker = vllm_modules
    events = []
    client = SimpleNamespace(close=lambda: events.append("client_close"))
    manager = SimpleNamespace(wake=lambda: events.append("gms_wake"))
    pool = SimpleNamespace(prepare_snapshot=lambda: events.append("gms_sleep"))
    allocator = SimpleNamespace(
        sleep=lambda offload_tags: events.append(("native_sleep", offload_tags)),
        wake_up=lambda tags: events.append(("native_wake", tags)),
    )

    monkeypatch.setattr(backend.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        backend,
        "get_socket_path",
        lambda device, tag: events.append(("socket", device, tag)) or "/gms.sock",
    )
    monkeypatch.setattr(
        backend,
        "AllocationClient",
        lambda path: events.append(("client", path)) or client,
    )
    monkeypatch.setattr(backend, "get_vmm", lambda: "vmm")
    monkeypatch.setattr(
        backend,
        "SnapshotMemoryManager",
        lambda received_client, vmm, device: (
            events.append(("manager", received_client, vmm, device)) or manager
        ),
    )
    monkeypatch.setattr(
        backend,
        "SnapshotTorchPool",
        lambda received_manager: events.append(("pool", received_manager)) or pool,
    )
    monkeypatch.setattr(
        backend,
        "install_model_loader_patch",
        lambda received_pool: events.append(("install_loader", received_pool)),
    )
    monkeypatch.setattr(
        "vllm.device_allocator.get_mem_allocator_instance", lambda: allocator
    )

    instance = backend.GMSV1SleepModeBackend()

    assert instance._client is client
    assert instance._manager is manager
    assert instance._pool is pool
    with pytest.raises(ValueError, match="level 1"):
        instance.suspend(2)
    instance.suspend()
    with pytest.raises(ValueError, match="partial-tag"):
        instance.resume(["weights"])
    instance.resume()

    assert events == [
        ("socket", 3, "snapshot-v1"),
        ("client", "/gms.sock"),
        ("manager", client, "vmm", 3),
        ("pool", manager),
        ("install_loader", pool),
        ("native_sleep", ("weights",)),
        "gms_sleep",
        "gms_wake",
        ("native_wake", None),
    ]
    assert instance.state() == "RUNNING"


def test_backend_exits_on_partial_suspend_and_resume_failures(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, _base_loader, _patches, _worker = vllm_modules
    events = []
    suspend_failure = RuntimeError("GMS suspend failed")
    resume_failure = RuntimeError("native resume failed")

    def fail_gms_suspend():
        events.append("gms_sleep")
        raise suspend_failure

    def fail_native_resume(tags):
        events.append(("native_wake", tags))
        raise resume_failure

    allocator = SimpleNamespace(
        sleep=lambda offload_tags: events.append(("native_sleep", offload_tags)),
        wake_up=fail_native_resume,
    )
    monkeypatch.setattr(
        "vllm.device_allocator.get_mem_allocator_instance", lambda: allocator
    )

    instance = object.__new__(backend.GMSV1SleepModeBackend)
    backend.CuMemBackend.__init__(instance)
    instance._pool = SimpleNamespace(prepare_snapshot=fail_gms_suspend)
    instance._manager = SimpleNamespace(wake=lambda: events.append("gms_wake"))

    with pytest.raises(SystemExit) as suspend_exit:
        instance.suspend()
    assert suspend_exit.value.code == 1
    assert suspend_exit.value.__cause__ is suspend_failure

    with pytest.raises(SystemExit) as resume_exit:
        instance.resume()
    assert resume_exit.value.code == 1
    assert resume_exit.value.__cause__ is resume_failure
    assert events == [
        ("native_sleep", ("weights",)),
        "gms_sleep",
        "gms_wake",
        ("native_wake", None),
    ]


def test_backend_closes_client_on_partial_construction_failure(
    vllm_modules,
    monkeypatch,
) -> None:
    backend, _base_loader, _patches, _worker = vllm_modules
    events = []
    client = SimpleNamespace(close=lambda: events.append("client_close"))

    monkeypatch.setattr(backend.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(backend, "get_socket_path", lambda _device, _tag: "/gms.sock")
    monkeypatch.setattr(backend, "AllocationClient", lambda _path: client)
    monkeypatch.setattr(backend, "get_vmm", lambda: "vmm")

    def fail_manager(_client, _vmm, _device):
        raise RuntimeError("manager failed")

    monkeypatch.setattr(backend, "SnapshotMemoryManager", fail_manager)

    with pytest.raises(RuntimeError, match="manager failed"):
        backend.GMSV1SleepModeBackend()

    assert events == ["client_close"]


def test_model_loader_patch_finalizes_after_leaving_gms_pool(
    vllm_modules,
    monkeypatch,
) -> None:
    _backend, base_loader, patches, _worker = vllm_modules

    events = []
    model = torch.nn.Module()

    @contextmanager
    def model_load_pool():
        events.append("pool_enter")
        try:
            yield
        finally:
            events.append("pool_exit")

    pool = SimpleNamespace(
        model_load_pool=model_load_pool,
        finalize_model_load=lambda received: events.append(("finalize", received)),
        abort_model_load=lambda cause: events.append(("abort", cause)),
    )

    def normal_loader(_loader, *args, **kwargs):
        events.append(("load", args, kwargs))
        return model

    monkeypatch.setattr(base_loader, "load_model", normal_loader)
    patches.install_model_loader_patch(pool)

    assert base_loader.load_model(object(), "config", prefix="model") is model
    assert events == [
        "pool_enter",
        ("load", ("config",), {"prefix": "model"}),
        "pool_exit",
        ("finalize", model),
    ]


def test_model_loader_patch_aborts_and_propagates_load_failure(
    vllm_modules,
    monkeypatch,
) -> None:
    _backend, base_loader, patches, _worker = vllm_modules

    events = []
    failure = RuntimeError("load failed")

    @contextmanager
    def model_load_pool():
        events.append("pool_enter")
        try:
            yield
        finally:
            events.append("pool_exit")

    pool = SimpleNamespace(
        model_load_pool=model_load_pool,
        finalize_model_load=lambda _model: pytest.fail("must not finalize"),
        abort_model_load=lambda cause: events.append(("abort", cause)),
    )

    def failing_loader(_loader, *args, **kwargs):
        events.append("load")
        raise failure

    monkeypatch.setattr(base_loader, "load_model", failing_loader)
    patches.install_model_loader_patch(pool)

    with pytest.raises(RuntimeError, match="load failed") as raised:
        base_loader.load_model(object())

    assert raised.value is failure
    assert events == [
        "pool_enter",
        "load",
        "pool_exit",
        ("abort", failure),
    ]
