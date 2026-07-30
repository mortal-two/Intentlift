from __future__ import annotations

import numpy as np
import pytest

from intentir.backend.cuda.qd import _logical_runtime_array


def test_qd_preserves_rank_zero_tensor_abi_for_scalar_carriers() -> None:
    scalar = _logical_runtime_array(
        np.asarray([3.25], dtype=np.float32), rank=0, name="scale"
    )

    assert scalar.shape == ()
    assert scalar.dtype == np.float32
    assert float(scalar) == pytest.approx(3.25)


def test_qd_rejects_multiple_values_for_rank_zero_input() -> None:
    with pytest.raises(ValueError, match="rank-0 logical input"):
        _logical_runtime_array(
            np.asarray([1, 2], dtype=np.int32), rank=0, name="extent"
        )
