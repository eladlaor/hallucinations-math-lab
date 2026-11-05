"""TriviaQA slicing: repeated questions must not leak across or within slices.

TriviaQA's rc.nocontext validation split lists most questions more than once
(one row per evidence source), so slicing rows instead of questions puts the
same question in two "disjoint" slices.
"""

import pytest

from eval.datasets.load_triviaqa import slice_bounds, unique_question_indices


def test_keeps_the_first_row_of_each_question_in_order() -> None:
    assert unique_question_indices(["a", "b", "a", "c", "b"]) == [0, 1, 3]


def test_no_repeats_keeps_every_row() -> None:
    assert unique_question_indices(["x", "y", "z"]) == [0, 1, 2]


def test_slice_beyond_the_unique_questions_fails_fast() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        slice_bounds(offset=2, n=2, dataset_size=3)
