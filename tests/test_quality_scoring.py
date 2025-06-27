from testgen_copilot.quality import TestQualityScorer


def _make_tests(tmp_path, with_assert: bool = True):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    if with_assert:
        (tests_dir / "test_sample.py").write_text(
            """

def test_ok():
    assert True
"""
        )
    else:
        (tests_dir / "test_sample.py").write_text(
            """

def test_bad():
    pass
"""
        )
    return tests_dir


def test_quality_full_score(tmp_path):
    tests_dir = _make_tests(tmp_path, with_assert=True)
    scorer = TestQualityScorer()
    assert scorer.score(tests_dir) == 100.0


def test_quality_partial_score(tmp_path):
    tests_dir = _make_tests(tmp_path, with_assert=False)
    scorer = TestQualityScorer()
    assert scorer.score(tests_dir) == 0.0
    assert scorer.low_quality_tests(tests_dir) == {"test_bad"}


def test_no_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    scorer = TestQualityScorer()
    assert scorer.score(tests_dir) == 100.0
    assert scorer.low_quality_tests(tests_dir) == set()
