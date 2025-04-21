# import numpy as np

# from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries


# def test_regulartimeseries():
#     data = RegularTimeSeries(
#         lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
#     )

#     train_domain = Interval(0.0, 6.0)
#     valid_domain = Interval(6.0, 8.0)
#     test_domain = Interval(8.0, 10.0)

#     data.add_split_mask("train", train_domain)
#     data.add_split_mask("valid", valid_domain)
#     data.add_split_mask("test", test_domain)

#     assert data.train_mask.sum() == data.train_mask[:60].sum() == 60
#     assert data.valid_mask.sum() == data.valid_mask[60:80].sum() == 20
#     assert data.test_mask.sum() == data.test_mask[80:].sum() == 20

#     # check what happens when things are skewed
#     data = RegularTimeSeries(
#         lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
#     )

#     train_domain = Interval(0.0, 6.051)
#     valid_domain = Interval(6.051, 7.999)
#     test_domain = Interval(7.999, 10.0)

#     data.add_split_mask("train", train_domain)
#     data.add_split_mask("valid", valid_domain)
#     data.add_split_mask("test", test_domain)

#     assert data.train_mask.sum() == data.train_mask[:61].sum() == 61
#     # assert data.valid_mask.sum() == data.valid_mask[61:80].sum() == 19
#     assert data.test_mask.sum() == data.test_mask[80:].sum() == 20


# def test_irregulartimeseries():
#     data = IrregularTimeSeries(
#         lfp=np.random.random((100, 48)),
#         timestamps=np.arange(0, 10, 0.1, dtype=np.float64),
#         domain="auto",
#     )

#     train_domain = Interval(0.0, 6.0)
#     valid_domain = Interval(6.0, 8.0)
#     test_domain = Interval(8.0, 10.0)

#     data.add_split_mask("train", train_domain)
#     data.add_split_mask("valid", valid_domain)
#     data.add_split_mask("test", test_domain)

#     assert data.train_mask.sum() == data.train_mask[:60].sum() == 60
#     assert data.valid_mask.sum() == data.valid_mask[60:80].sum() == 20
#     assert data.test_mask.sum() == data.test_mask[80:].sum() == 20

#     # check what happens when things are skewed
#     data = IrregularTimeSeries(
#         lfp=np.random.random((100, 48)),
#         timestamps=np.arange(0, 10, 0.1, dtype=np.float64),
#         domain="auto",
#     )

#     train_domain = Interval(0.0, 6.051)
#     valid_domain = Interval(6.051, 7.999)
#     test_domain = Interval(7.999, 10.0)

#     data.add_split_mask("train", train_domain)
#     data.add_split_mask("valid", valid_domain)
#     data.add_split_mask("test", test_domain)

#     assert data.train_mask.sum() == data.train_mask[:61].sum() == 61
#     assert data.valid_mask.sum() == data.valid_mask[61:80].sum() == 19
#     assert data.test_mask.sum() == data.test_mask[80:].sum() == 20


# # TODO add tests for Interval


# def test_set_split_domain():
#     data = Data(
#         regular=RegularTimeSeries(
#             lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
#         ),
#         irregular=IrregularTimeSeries(
#             lfp=np.random.random((100, 48)),
#             timestamps=np.arange(0, 10, 0.1, dtype=np.float64),
#             domain="auto",
#         ),
#         domain="auto",
#     )

#     data.set_train_domain(Interval(0.0, 6.0))
#     data.set_valid_domain(Interval(6.0, 8.0))
#     data.set_test_domain(Interval(8.0, 10.0))

#     regular = data.regular
#     assert regular.train_mask.sum() == regular.train_mask[:60].sum() == 60
#     assert regular.valid_mask.sum() == regular.valid_mask[60:80].sum() == 20
#     assert regular.test_mask.sum() == regular.test_mask[80:].sum() == 20

#     irregular = data.irregular
#     assert irregular.train_mask.sum() == irregular.train_mask[:60].sum() == 60
#     assert irregular.valid_mask.sum() == irregular.valid_mask[60:80].sum() == 20
#     assert irregular.test_mask.sum() == irregular.test_mask[80:].sum() == 20
