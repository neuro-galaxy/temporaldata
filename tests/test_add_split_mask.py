import numpy as np

from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries


def test_regulartimeseries():
    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
    )

    train_domain = Interval(0.0, 6.0)
    valid_domain = Interval(6.0, 8.0)
    test_domain = Interval(8.0, 10.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    m = data.train_mask
    assert m.sum() == m[:60].sum() == 60
    m = data.valid_mask
    assert m.sum() == m[60:80].sum() == 20
    m = data.test_mask
    assert m.sum() == m[80:].sum() == 20

    # check what happens when things are skewed
    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
    )

    train_domain = Interval(0.0, 6.051)
    valid_domain = Interval(6.051, 7.999)
    test_domain = Interval(7.999, 10.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    m = data.train_mask
    assert m.sum() == m[:61].sum() == len(m[:61]) == 61
    m = data.valid_mask
    assert m.sum() == m[61:80].sum() == len(m[61:80]) == 19
    m = data.test_mask
    assert m.sum() == m[80:].sum() == len(m[80:]) == 20

    data = RegularTimeSeries(
        lfp=np.random.random((300, 48)),
        sampling_rate=10,
        domain=Interval(
            start=np.array([10.0, 30.0, 50.0], dtype=np.float64),
            end=np.array([19.9, 39.9, 59.9], dtype=np.float64),
        ),
    )

    train_domain = Interval(15.0, 34.99)
    valid_domain = Interval(38.001, 55.001)
    test_domain = Interval(55.99, 58.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    m = data.train_mask
    assert m.sum() == m[50:150].sum() == len(m[50:150]) == 100
    m = data.valid_mask
    assert m.sum() == m[181:251].sum() == len(m[181:251]) == 70
    m = data.test_mask
    assert m.sum() == m[260:280].sum() == len(m[260:280]) == 20


def test_irregulartimeseries():
    data = IrregularTimeSeries(
        lfp=np.random.random((100, 48)),
        timestamps=np.arange(0, 10, 0.1, dtype=np.float64),
        domain="auto",
    )

    train_domain = Interval(0.0, 6.0)
    valid_domain = Interval(6.0, 8.0)
    test_domain = Interval(8.0, 10.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    m = data.train_mask
    assert m.sum() == m[:60].sum() == len(m[:60]) == 60
    m = data.valid_mask
    assert m.sum() == m[60:80].sum() == len(m[60:80]) == 20
    m = data.test_mask
    assert m.sum() == m[80:].sum() == len(m[80:]) == 20

    # check what happens when things are skewed
    data = IrregularTimeSeries(
        lfp=np.random.random((100, 48)),
        timestamps=np.arange(0, 10, 0.1, dtype=np.float64),
        domain="auto",
    )

    train_domain = Interval(0.0, 6.051)
    valid_domain = Interval(6.051, 7.999)
    test_domain = Interval(7.999, 10.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    m = data.train_mask
    assert m.sum() == m[:61].sum() == len(m[:61]) == 61
    m = data.valid_mask
    assert m.sum() == m[61:80].sum() == len(m[61:80]) == 19
    m = data.test_mask
    assert m.sum() == m[80:].sum() == len(m[80:]) == 20


# TODO add tests for Interval


def test_set_split_domain():
    data = Data(
        regular=RegularTimeSeries(
            lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
        ),
        irregular=IrregularTimeSeries(
            lfp=np.random.random((100, 48)),
            timestamps=np.arange(0, 10, 0.1, dtype=np.float64),
            domain="auto",
        ),
        domain="auto",
    )

    data.set_train_domain(Interval(0.0, 6.0))
    data.set_valid_domain(Interval(6.0, 8.0))
    data.set_test_domain(Interval(8.0, 10.0))

    regular = data.regular
    assert regular.train_mask.sum() == regular.train_mask[:60].sum() == 60
    assert regular.valid_mask.sum() == regular.valid_mask[60:80].sum() == 20
    assert regular.test_mask.sum() == regular.test_mask[80:].sum() == 20

    irregular = data.irregular
    assert irregular.train_mask.sum() == irregular.train_mask[:60].sum() == 60
    assert irregular.valid_mask.sum() == irregular.valid_mask[60:80].sum() == 20
    assert irregular.test_mask.sum() == irregular.test_mask[80:].sum() == 20
