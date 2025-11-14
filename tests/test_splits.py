import pytest
import warnings
import numpy as np
from temporaldata import (
    ArrayDict,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
    Data,
)


@pytest.fixture
def simple_data():
    """Fixture providing a simple Data object with RegularTimeSeries."""
    return Data(
        lfp=RegularTimeSeries(
            lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
        ),
        domain="auto",
    )


@pytest.fixture
def nested_data():
    """Fixture providing a Data object with nested RegularTimeSeries, IrregularTimeSeries, and Interval."""
    return Data(
        lfp=RegularTimeSeries(
            lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
        ),
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3, 7.1, 7.2, 8.1, 8.2]),
            unit_index=np.array([0, 0, 1, 0, 1, 2, 0, 1, 0, 1]),
            waveforms=np.zeros((10, 48)),
            domain="auto",
        ),
        trials=Interval(
            start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ),
        domain="auto",
    )


@pytest.fixture
def data_with_trials():
    """Fixture providing a Data object with RegularTimeSeries and Interval."""
    return Data(
        lfp=RegularTimeSeries(
            lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
        ),
        trials=Interval(
            start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            end=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ),
        domain="auto",
    )


@pytest.mark.parametrize(
    "train_interval,test_interval,valid_interval",
    [
        # Basic case
        (Interval(0.0, 6.0), Interval(6.0, 8.0), Interval(8.0, 10.0)),
        # Disjoint intervals with gaps
        (Interval(0.0, 3.0), Interval(5.0, 7.0), Interval(9.0, 10.0)),
        # Multiple intervals in train
        (
            Interval(start=np.array([0.0, 2.0]), end=np.array([1.0, 3.0])),
            Interval(5.0, 7.0),
            Interval(8.0, 10.0),
        ),
        # All have multiple intervals
        (
            Interval(start=np.array([0.0, 2.0]), end=np.array([1.0, 3.0])),
            Interval(start=np.array([5.0, 7.0]), end=np.array([6.0, 8.0])),
            Interval(start=np.array([9.0, 9.5]), end=np.array([9.2, 10.0])),
        ),
        # Lots of intervals
        (
            Interval(
                start=np.array([0.0, 3.0, 6.0, 9.0]),
                end=np.array([1.0, 4.0, 7.0, 10.0]),
            ),
            Interval(start=np.array([1.0, 4.0, 7.0]), end=np.array([2.0, 5.0, 8.0])),
            Interval(start=np.array([2.0, 5.0, 8.0]), end=np.array([3.0, 6.0, 9.0])),
        ),
    ],
)
def test_add_splits_single_with_valid(
    data_with_trials, train_interval, test_interval, valid_interval
):
    """Test adding a single split with train, test, and valid intervals."""
    data_with_trials.add_splits(
        name="split_0", train=train_interval, test=test_interval, valid=valid_interval
    )

    # Check that splits object exists and is a Data object
    assert hasattr(data_with_trials, "splits")
    assert isinstance(data_with_trials.splits, Data)

    # Check that split_0 exists and is an ArrayDict
    assert hasattr(data_with_trials.splits, "split_0")
    assert isinstance(data_with_trials.splits.split_0, ArrayDict)

    # Check that split_0 contains train, test, and valid intervals
    assert hasattr(data_with_trials.splits.split_0, "train")
    assert hasattr(data_with_trials.splits.split_0, "test")
    assert hasattr(data_with_trials.splits.split_0, "valid")
    assert isinstance(data_with_trials.splits.split_0.train, Interval)
    assert isinstance(data_with_trials.splits.split_0.test, Interval)
    assert isinstance(data_with_trials.splits.split_0.valid, Interval)

    # Check that intervals are stored correctly
    assert np.allclose(
        data_with_trials.splits.split_0.train.start, train_interval.start
    )
    assert np.allclose(data_with_trials.splits.split_0.train.end, train_interval.end)
    assert np.allclose(data_with_trials.splits.split_0.test.start, test_interval.start)
    assert np.allclose(data_with_trials.splits.split_0.test.end, test_interval.end)
    assert np.allclose(
        data_with_trials.splits.split_0.valid.start, valid_interval.start
    )
    assert np.allclose(data_with_trials.splits.split_0.valid.end, valid_interval.end)


def test_add_splits_single_without_valid(simple_data):
    """Test adding a single split with only train and test intervals (no valid)."""
    train_interval = Interval(0.0, 7.0)
    test_interval = Interval(7.0, 10.0)

    simple_data.add_splits(name="split_0", train=train_interval, test=test_interval)

    # Check that splits object exists
    assert hasattr(simple_data, "splits")
    assert isinstance(simple_data.splits, Data)

    # Check that split_0 exists and contains train and test
    assert hasattr(simple_data.splits, "split_0")
    assert isinstance(simple_data.splits.split_0, ArrayDict)
    assert hasattr(simple_data.splits.split_0, "train")
    assert hasattr(simple_data.splits.split_0, "test")
    assert not hasattr(simple_data.splits.split_0, "valid")


def test_add_splits_multiple(simple_data):
    """Test adding multiple splits at once."""
    splits = [
        {
            "name": "split_0",
            "train": Interval(0.0, 5.0),
            "test": Interval(5.0, 7.0),
            "valid": Interval(7.0, 10.0),
        },
        {
            "name": "split_1",
            "train": Interval(0.0, 6.0),
            "test": Interval(6.0, 10.0),
        },
        {
            "name": "split_2",
            "train": Interval(0.0, 4.0),
            "test": Interval(4.0, 8.0),
            "valid": Interval(8.0, 10.0),
        },
    ]

    simple_data.add_splits(splits=splits)

    # Check that all splits exist
    assert hasattr(simple_data.splits, "split_0")
    assert hasattr(simple_data.splits, "split_1")
    assert hasattr(simple_data.splits, "split_2")

    # Check split_0 has train, test, valid
    assert hasattr(simple_data.splits.split_0, "train")
    assert hasattr(simple_data.splits.split_0, "test")
    assert hasattr(simple_data.splits.split_0, "valid")

    # Check split_1 has train, test but no valid
    assert hasattr(simple_data.splits.split_1, "train")
    assert hasattr(simple_data.splits.split_1, "test")
    assert not hasattr(simple_data.splits.split_1, "valid")

    # Check split_2 has train, test, valid
    assert hasattr(simple_data.splits.split_2, "train")
    assert hasattr(simple_data.splits.split_2, "test")
    assert hasattr(simple_data.splits.split_2, "valid")


def test_add_splits_validation_sorted(simple_data):
    """Test that validation checks for sorted intervals."""
    # Create unsorted train interval
    unsorted_train = Interval(
        start=np.array([3.0, 1.0, 5.0]), end=np.array([4.0, 2.0, 6.0])
    )
    test_interval = Interval(6.0, 10.0)

    with pytest.raises(ValueError, match="sorted"):
        simple_data.add_splits(name="split_0", train=unsorted_train, test=test_interval)


def test_add_splits_validation_disjoint(simple_data):
    """Test that validation checks for disjoint intervals within a split."""
    # Create overlapping train and test intervals
    train_interval = Interval(0.0, 7.0)
    test_interval = Interval(6.0, 10.0)  # Overlaps with train

    with pytest.raises(ValueError, match="disjoint|overlap"):
        simple_data.add_splits(name="split_0", train=train_interval, test=test_interval)


def test_add_splits_validation_disjoint_with_valid(simple_data):
    """Test that validation checks for disjoint intervals when valid is present."""
    # Create overlapping intervals
    train_interval = Interval(0.0, 6.0)
    test_interval = Interval(5.0, 8.0)  # Overlaps with train
    valid_interval = Interval(8.0, 10.0)

    with pytest.raises(ValueError, match="disjoint|overlap"):
        simple_data.add_splits(
            name="split_0",
            train=train_interval,
            test=test_interval,
            valid=valid_interval,
        )

    # Test valid overlapping with test
    train_interval = Interval(0.0, 5.0)
    test_interval = Interval(5.0, 8.0)
    valid_interval = Interval(7.0, 10.0)  # Overlaps with test

    with pytest.raises(ValueError, match="disjoint|overlap"):
        simple_data.add_splits(
            name="split_1",
            train=train_interval,
            test=test_interval,
            valid=valid_interval,
        )


def test_add_splits_backward_compatibility_deprecation(simple_data):
    """Test that old methods still work but emit deprecation warnings."""
    train_interval = Interval(0.0, 6.0)
    valid_interval = Interval(6.0, 8.0)
    test_interval = Interval(8.0, 10.0)

    # Test set_train_domain emits deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simple_data.set_train_domain(train_interval)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower() or "set_train_domain" in str(
            w[0].message
        )

    # Test set_valid_domain emits deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simple_data.set_valid_domain(valid_interval)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Test set_test_domain emits deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simple_data.set_test_domain(test_interval)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Verify old functionality still works
    assert hasattr(simple_data.lfp, "train_mask")
    assert hasattr(simple_data.lfp, "valid_mask")
    assert hasattr(simple_data.lfp, "test_mask")


def test_add_splits_error_duplicate_name(simple_data):
    """Test that adding a split with duplicate name raises an error."""
    simple_data.add_splits(
        name="split_0",
        train=Interval(0.0, 5.0),
        test=Interval(5.0, 10.0),
    )

    # Try to add another split with the same name
    with pytest.raises(ValueError, match="already exists|duplicate"):
        simple_data.add_splits(
            name="split_0",
            train=Interval(0.0, 6.0),
            test=Interval(6.0, 10.0),
        )


def test_add_splits_error_overlapping_intervals(simple_data):
    """Test error cases with overlapping intervals."""
    # Adjacent intervals should be fine (end == start)
    train_interval = Interval(0.0, 5.0)
    test_interval = Interval(5.0, 10.0)  # Adjacent, not overlapping
    simple_data.add_splits(name="split_0", train=train_interval, test=test_interval)

    # But overlapping should fail
    train_interval = Interval(0.0, 5.5)
    test_interval = Interval(5.0, 10.0)  # Overlaps

    with pytest.raises(ValueError, match="disjoint|overlap"):
        simple_data.add_splits(name="split_1", train=train_interval, test=test_interval)


def test_add_splits_error_unsorted_intervals(simple_data):
    """Test error case with unsorted intervals."""
    # Create unsorted test interval
    train_interval = Interval(0.0, 5.0)
    unsorted_test = Interval(start=np.array([7.0, 6.0]), end=np.array([8.0, 7.0]))

    with pytest.raises(ValueError, match="sorted"):
        simple_data.add_splits(name="split_0", train=train_interval, test=unsorted_test)


def test_add_splits_prevent_on_split_info_object(simple_data):
    """Test that calling add_splits on a split information object raises ValueError."""
    # Add a split to create the splits object
    simple_data.add_splits(
        name="split_0",
        train=Interval(0.0, 5.0),
        test=Interval(5.0, 10.0),
    )

    # Try to call add_splits on the splits object itself
    with pytest.raises(ValueError, match="split information|add_splits"):
        simple_data.splits.add_splits(
            name="split_1",
            train=Interval(0.0, 6.0),
            test=Interval(6.0, 10.0),
        )


def test_add_splits_multiple_validation(simple_data):
    """Test that validation is applied to each split when adding multiple splits."""
    # First split is valid
    # Second split has overlapping intervals
    splits = [
        {
            "name": "split_0",
            "train": Interval(0.0, 5.0),
            "test": Interval(5.0, 10.0),
        },
        {
            "name": "split_1",
            "train": Interval(0.0, 6.0),
            "test": Interval(5.5, 10.0),  # Overlaps with train
        },
    ]

    with pytest.raises(ValueError, match="disjoint|overlap"):
        simple_data.add_splits(splits=splits)
