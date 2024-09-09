import numpy as np

from .temporaldata import (
    Data,
    ArrayDict,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
)


def concat(objs, domains=None, stack_domains=False, autosort=True, timegap=1.0):
    # if the list is empty, return error
    if len(objs) == 0:
        raise ValueError("need at least one object to concatenate")

    # if there is only one object, return it
    if len(objs) == 1:
        if autosort:
            objs[0].sort()
        return objs[0]

    # type check
    obj_type = type(objs[0])

    if obj_type not in [
        Data,
        ArrayDict,
        IrregularTimeSeries,
        RegularTimeSeries,
        Interval,
    ]:
        raise ValueError("Unsupported object type: {}".format(obj_type))

    if any(not isinstance(obj, obj_type) for obj in objs):
        raise ValueError(
            "All objects must be of the same type, got: {}".format(
                [type(obj) for obj in objs]
            )
        )

    # if domains is not provided, use the domain of each object
    if domains is None:
        if obj_type == Interval:
            domains = objs
        elif obj_type == ArrayDict:
            domains = [None] * len(objs)
        else:
            domains = [obj.domain for obj in objs]

    # check that the length of domains matches the number of objects
    if len(domains) != len(objs):
        raise ValueError(
            f"the lengths of objects and domains must match, got {len(domains)} domains"
            f" and {len(objs)} objects."
        )

    # check that all objects have the same keys
    keys = objs[0].keys
    for obj in objs:
        if set(obj.keys) != set(keys):
            raise ValueError(
                f"All objects must share the same keys, found an object with "
                f"keys:({keys}) and another with keys:({obj.keys})."
            )

    if obj_type == Data:
        obj_concat_dict = {}
        for k in keys:
            obj_concat_dict[k] = concat(
                [getattr(obj, k) for obj in objs],
                domains=domains,
                stack_domains=stack_domains,
                autosort=autosort,
                timegap=timegap,
            )
        obj_concat = Data(**obj_concat_dict)
        return obj_concat

    # check that IrregularTimeSeries or Interval objects have the same timekeys
    if obj_type in [IrregularTimeSeries, Interval]:
        timekeys = objs[0].timekeys
        for obj in objs:
            if set(obj.timekeys) != set(timekeys):
                raise ValueError(
                    f"All objects must share the same timekeys, found an object with "
                    f"timekeys:({timekeys}) and another with timekeys:({obj.timekeys})."
                )

        # build offsets
        start_offsets = [0] + [domain.end[-1] for domain in domains[:-1]]
        start_offsets = np.cumsum(start_offsets)
    else:
        timekeys = []

    # check that RegularTimeSeries objects have the same sampling rate
    if obj_type == RegularTimeSeries:
        sampling_rate = objs[0].sampling_rate
        for obj in objs:
            if set(obj.sampling_rate) != set(sampling_rate):
                raise ValueError(
                    f"All RegularTimeSeries objects must have the same sampling rate, "
                    f"found an object with sampling rate:({sampling_rate}) and another "
                    f"with sampling rate:({obj.sampling_rate})."
                )

    # concatenate the objects
    obj_concat_dict = {}
    for k in keys:
        if k in timekeys and stack_domains:
            obj_concat_dict[k] = np.concatenate(
                [getattr(obj, k) + start_offsets[i] for i, obj in enumerate(objs)]
            )
        else:
            obj_concat_dict[k] = np.concatenate([getattr(obj, k) for obj in objs])

    domain = concat(
        domains, stack_domains=stack_domains, autosort=autosort, timegap=timegap
    )

    if obj_type == IrregularTimeSeries:
        obj_concat = IrregularTimeSeries(
            **obj_concat_dict, timekeys=timekeys, domain=domain
        )
    elif obj_type == Interval:
        obj_concat = Interval(**obj_concat_dict, timekeys=timekeys)
    elif obj_type == RegularTimeSeries:
        obj_concat = RegularTimeSeries(
            **obj_concat_dict, sampling_rate=sampling_rate, domain=domain
        )
    else:
        obj_concat = ArrayDict(obj_concat_dict)

    if autosort and obj_type in [IrregularTimeSeries, Interval]:
        obj_concat.sort()

    return obj_concat
