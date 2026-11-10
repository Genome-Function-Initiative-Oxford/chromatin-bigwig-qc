import numpy as np
import pyBigWig
import pytest

from bigwig_qc.bigwig_io import degrade_bigwig_file


@pytest.fixture
def toy_bigwig(tmp_path):
    path = tmp_path / "toy.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries(["chr1"], [0], ends=[1000], values=[4.0])
    bw.close()
    return path


def test_degrade_bigwig_file_identity_down_sample_preserves_signal(toy_bigwig, tmp_path):
    output = tmp_path / "out.bw"
    degrade_bigwig_file(toy_bigwig, mode="down_sample", level=1.0, seed=0, output_path=output, bin_size=100)

    bw = pyBigWig.open(str(output))
    values = np.nan_to_num(np.array(bw.values("chr1", 0, 1000)), nan=0.0)
    bw.close()
    assert values.mean() == pytest.approx(4.0, rel=0.05)


def test_degrade_bigwig_file_down_sample_reduces_signal(toy_bigwig, tmp_path):
    output = tmp_path / "out.bw"
    degrade_bigwig_file(toy_bigwig, mode="down_sample", level=0.1, seed=0, output_path=output, bin_size=100)

    bw = pyBigWig.open(str(output))
    values = np.nan_to_num(np.array(bw.values("chr1", 0, 1000)), nan=0.0)
    bw.close()
    assert values.mean() < 1.0
