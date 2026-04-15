from pathlib import Path
from subprocess import run


def test_xai_run_creates_output(tmp_path):
    out = tmp_path / "xai_out"
    out.mkdir()
    cmd = [
        "evrp-xai",
        "--env-config", "configs/agents_examples.yaml",
        "--example", "a2c_example",
        "--out", str(out),
    ]
    res = run(cmd)
    # script should exit 0
    assert res.returncode == 0
    # expect an output file
    assert (out / "xai_route.png").exists()
