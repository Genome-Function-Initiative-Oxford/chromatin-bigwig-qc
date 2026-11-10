import pytest

from bigwig_qc import lanceotron


def test_call_peaks_locates_output_file(tmp_path, monkeypatch):
    bigwig_path = tmp_path / "sample.bw"
    bigwig_path.touch()
    output_dir = tmp_path / "peaks"

    def fake_run(cmd, check):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample_L-tron.bed").write_text("chrom\tstart\tend\n")

    monkeypatch.setattr(lanceotron.subprocess, "run", fake_run)

    result = lanceotron.call_peaks(bigwig_path, output_dir)
    assert result == output_dir / "sample_L-tron.bed"


def test_call_peaks_locates_output_file_for_multi_dot_bigwig_name(tmp_path, monkeypatch):
    # LanceOtron derives its output filename from the basename split on the
    # FIRST "." (not Path.stem, which only strips the final suffix) - for a
    # multi-dot name like the degradation benchmark's "<track>.<mode>.<level>.
    # <seed>.bw", LanceOtron truncates to just "<track>_L-tron.bed". Confirmed
    # against a real smoke-test job's traceback (see lanceotron.py's docstring).
    bigwig_path = tmp_path / "Mast.down_sample.1.0.0.bw"
    bigwig_path.touch()
    output_dir = tmp_path / "peaks"

    def fake_run(cmd, check):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "Mast_L-tron.bed").write_text("chrom\tstart\tend\n")

    monkeypatch.setattr(lanceotron.subprocess, "run", fake_run)

    result = lanceotron.call_peaks(bigwig_path, output_dir)
    assert result == output_dir / "Mast_L-tron.bed"


def test_call_peaks_raises_if_no_output_produced(tmp_path, monkeypatch):
    bigwig_path = tmp_path / "sample.bw"
    bigwig_path.touch()
    output_dir = tmp_path / "peaks"

    monkeypatch.setattr(lanceotron.subprocess, "run", lambda cmd, check: None)

    with pytest.raises(FileNotFoundError):
        lanceotron.call_peaks(bigwig_path, output_dir)
