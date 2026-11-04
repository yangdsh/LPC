# SPDX-License-Identifier: Apache-2.0
"""Tests for loading the LPC predictor checkpoint named in a cache hint."""

import os
import pickle
import queue
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.core import learn_conversation
from vllm.core.block import prefix_caching_block
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.learn_conversation import MLModel, MLPClassifier

SHIPPED_CHECKPOINTS = sorted(
    Path(__file__).parents[3].glob("benchmarks/checkpoints_*/*.pt"))


class DummySentenceTransformer:
    """Stands in for the embedding model, so load_model downloads nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def eval(self):
        return self


@pytest.fixture(autouse=True)
def no_embedding_model(monkeypatch):
    monkeypatch.setattr(learn_conversation, "SentenceTransformer",
                        DummySentenceTransformer)


def make_model() -> MLModel:
    model = MLModel(device="cpu")
    model.input_dim, model.output_dim = 385, 2
    model.classifier = MLPClassifier(input_dim=385,
                                     hidden_dim=128,
                                     output_dim=2,
                                     num_layers=3,
                                     dropout=0.3)
    return model


def assert_same_weights(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def write_marker(path):
    with open(path, "w") as f:
        f.write("ran during torch.load")


class RunsCodeOnLoad:
    """Unpickling this object calls write_marker."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return write_marker, (self.marker, )


class PrintsOnLoad(torch.nn.Module):
    """torch.jit.load of this module's archive runs __setstate__."""

    def __init__(self):
        super().__init__()
        self.weight = torch.zeros(1)

    @torch.jit.export
    def __getstate__(self) -> tuple[torch.Tensor, bool]:
        return (self.weight, self.training)

    @torch.jit.export
    def __setstate__(self, state: tuple[torch.Tensor, bool]) -> None:
        print("ran during torch.load")
        self.weight = state[0]
        self.training = state[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_save_model_round_trip(tmp_path):
    model = make_model()
    path = str(tmp_path / "predictor.pt")
    model.save_model(path)

    loaded = MLModel(device="cpu")
    loaded.load_model(path)

    assert_same_weights(loaded.classifier.state_dict(),
                        model.classifier.state_dict())


@pytest.mark.parametrize("path",
                         SHIPPED_CHECKPOINTS,
                         ids=[p.parent.name for p in SHIPPED_CHECKPOINTS])
def test_shipped_checkpoints_load(path):
    model = MLModel(device="cpu")
    model.load_model(str(path))

    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert_same_weights(model.classifier.state_dict(),
                        saved["model_state_dict"])


def test_load_model_rejects_pickled_code(tmp_path):
    path = str(tmp_path / "predictor.pt")
    make_model().save_model(path)
    checkpoint = torch.load(path, weights_only=True)
    marker = tmp_path / "marker"
    checkpoint["extra"] = RunsCodeOnLoad(str(marker))
    torch.save(checkpoint, path)

    with pytest.raises(pickle.UnpicklingError):
        MLModel(device="cpu").load_model(path)
    assert not marker.exists()


def test_load_model_rejects_torchscript_archive(tmp_path, capsys):
    path = str(tmp_path / "predictor.pt")
    torch.jit.save(torch.jit.script(PrintsOnLoad()), path)

    with pytest.raises(RuntimeError, match="TorchScript archive"):
        MLModel(device="cpu").load_model(path)
    assert "ran during torch.load" not in capsys.readouterr().out


def test_load_model_accepts_numpy_best_threshold(tmp_path):
    # vllm_cache_bench/get_predictor_accuracy.py saves best_threshold, taken
    # from sklearn's precision_recall_curve, so it is a numpy.float64.
    model = make_model()
    path = str(tmp_path / "predictor.pt")
    model.save_model(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["config"]["best_threshold"] = np.float64(0.47)
    torch.save(checkpoint, path)

    loaded = MLModel(device="cpu")
    loaded.load_model(path)

    assert_same_weights(loaded.classifier.state_dict(),
                        model.classifier.state_dict())


@pytest.fixture
def predictor_dir_and_paths(tmp_path):
    allowed = tmp_path / "predictors"
    outside = tmp_path / "outside"
    sibling = tmp_path / "predictors_evil"
    for directory in (allowed, outside, sibling):
        directory.mkdir()
    for file in (allowed / "good.pt", outside / "evil.pt",
                 sibling / "evil.pt"):
        file.touch()
    (allowed / "link.pt").symlink_to(outside / "evil.pt")
    return allowed, {
        "inside": str(allowed / "good.pt"),
        "outside": str(outside / "evil.pt"),
        "dotdot": str(allowed / ".." / "outside" / "evil.pt"),
        "symlink_out": str(allowed / "link.pt"),
        "sibling_prefix": str(sibling / "evil.pt"),
    }


def run_predictor_worker(monkeypatch, paths, on_init=None):
    """Runs predictor_worker on one hint per path with a fake predictor.

    on_init, if given, runs when the worker creates a predictor. Returns the
    paths the worker loaded and each hint's prob_has_next.
    """
    loaded = []

    class FakePredictor:

        def __init__(self):
            if on_init is not None:
                on_init()

        def load_model(self, path):
            loaded.append(path)

        def predict_single_processed(self, conversation_input, turns):
            return 0.7

    hint_queue = queue.Queue()
    monkeypatch.setattr(prefix_caching_block, "MLModel", FakePredictor)
    monkeypatch.setattr(prefix_caching_block, "to_predict_queue", hint_queue)
    hints = {
        name: {
            "checkpoint": path,
            "conversation_input": "query: hello",
            "turns": 1
        }
        for name, path in paths.items()
    }
    for hint in hints.values():
        hint_queue.put(hint)
    hint_queue.put(None)

    # The default 'lru' eviction starts no predictor thread, so run the worker
    # here. It returns when it reads the None.
    PrefixCachingBlockAllocator(num_blocks=4, block_size=16).predictor_worker()

    # Every hint was marked done. Only the None is left.
    assert hint_queue.unfinished_tasks == 1
    return loaded, {
        name: hint["prob_has_next"]
        for name, hint in hints.items()
    }


def test_predictor_worker_only_loads_from_lpc_predictor_dir(
        monkeypatch, predictor_dir_and_paths):
    allowed, paths = predictor_dir_and_paths
    monkeypatch.setenv("LPC_PREDICTOR_DIR", str(allowed))

    loaded, probs = run_predictor_worker(monkeypatch, paths)

    assert loaded == [os.path.realpath(paths["inside"])]
    assert probs == {
        "inside": 0.7,
        "outside": 0,
        "dotdot": 0,
        "symlink_out": 0,
        "sibling_prefix": 0,
    }


def test_predictor_worker_loads_the_path_it_checked(monkeypatch,
                                                    predictor_dir_and_paths):
    allowed, paths = predictor_dir_and_paths
    link = Path(paths["outside"]).with_name("swap.pt")
    link.symlink_to(paths["inside"])

    def repoint_link():
        # predictor_worker creates the predictor after the LPC_PREDICTOR_DIR
        # check and before load_model.
        new_link = link.with_name("swap.pt.new")
        new_link.symlink_to(paths["outside"])
        os.replace(new_link, link)

    monkeypatch.setenv("LPC_PREDICTOR_DIR", str(allowed))
    loaded, probs = run_predictor_worker(monkeypatch, {"swap": str(link)},
                                         on_init=repoint_link)

    assert loaded == [os.path.realpath(paths["inside"])]
    assert probs == {"swap": 0.7}


def test_predictor_worker_loads_any_path_without_lpc_predictor_dir(
        monkeypatch, predictor_dir_and_paths):
    _, paths = predictor_dir_and_paths
    monkeypatch.delenv("LPC_PREDICTOR_DIR", raising=False)

    loaded, probs = run_predictor_worker(monkeypatch, paths)

    assert loaded == list(paths.values())
    assert set(probs.values()) == {0.7}
