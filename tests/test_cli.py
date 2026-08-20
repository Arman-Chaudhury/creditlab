import json

import pytest

from creditlab.cli import main


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run fixtures -> ingest -> train once; commands under test reuse it."""
    root = tmp_path_factory.mktemp("cli")
    fix = root / "fix"
    work = root / "work"
    assert main(["fixtures", "--out", str(fix), "--loans", "300", "--seed", "5"]) == 0
    acq, perf = str(fix / "acquisition.txt"), str(fix / "performance.txt")
    assert main(["ingest", acq, perf, "--workdir", str(work)]) == 0
    assert main(["train", "--workdir", str(work)]) == 0
    return work


class TestPipeline:
    def test_artifacts_exist(self, pipeline):
        assert (pipeline / "dataset.json.gz").exists()
        assert (pipeline / "model.json").exists()
        artifact = json.loads((pipeline / "model.json").read_text())
        assert artifact["kind"] == "logistic"

    def test_validate_writes_report(self, pipeline, capsys):
        assert main(["validate", "--workdir", str(pipeline)]) == 0
        out = capsys.readouterr().out
        assert "OOT" in out and "realized / predicted" in out
        report = json.loads((pipeline / "validation.json").read_text())
        assert report["train_max_vintage"] == 2006

    def test_explain_prints_reasons(self, pipeline, capsys):
        dataset = json.loads(
            __import__("gzip").open(pipeline / "dataset.json.gz", "rt").read()
        )
        loan_id = dataset["loan_ids"][0]
        assert main(["explain", loan_id, "--workdir", str(pipeline)]) == 0
        out = capsys.readouterr().out
        assert f"loan {loan_id}" in out and "PD=" in out

    def test_explain_missing_loan_exits(self, pipeline):
        with pytest.raises(SystemExit):
            main(["explain", "NOPE", "--workdir", str(pipeline)])

    def test_tearsheet_renders(self, pipeline):
        assert main(["tearsheet", "--workdir", str(pipeline),
                     "--data-note", "unit-test synthetic data"]) == 0
        html = (pipeline / "tearsheet.html").read_text()
        assert html.startswith("<!DOCTYPE html>")
        for section in ("Dataset accounting", "Crisis out-of-time validation",
                        "Proxy-fairness audit", "risk decile"):
            assert section in html
        assert "unit-test synthetic data" in html
        # self-contained: no external scripts, stylesheets, or images
        for forbidden in ("<script", "src=", "<link", "http://", "https://"):
            assert forbidden not in html, f"external reference: {forbidden}"

    def test_gbm_training_runs(self, pipeline, capsys):
        assert main(["train", "--model", "gbm", "--workdir", str(pipeline)]) == 0
        assert "gbm in-sample" in capsys.readouterr().out
        # restore the logistic artifact for any later test using model.json
        assert main(["train", "--workdir", str(pipeline)]) == 0

    def test_missing_dataset_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="ingest"):
            main(["train", "--workdir", str(tmp_path / "empty")])


class TestDemo:
    def test_demo_reproduces_readme_numbers(self, tmp_path, capsys):
        """The README guarantee, enforced: `creditlab demo` recomputes the
        headline table and must match the published reference values."""
        code = main(["demo", "--out", str(tmp_path / "demo")])
        out = capsys.readouterr().out
        assert code == 0, f"demo drifted from README:\n{out}"
        assert "demo reproduces the README reference numbers" in out
        assert "DRIFTED" not in out
        assert (tmp_path / "demo" / "tearsheet.html").exists()
        assert "SYNTHETIC" in (tmp_path / "demo" / "tearsheet.html").read_text()
