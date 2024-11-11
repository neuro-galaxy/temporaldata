Getting Started
===============

**temporaldata** is a Python package for working with temporal data. In particular it is
designed for complex, multi-modal, multi-resolution data. 

When to use temporaldata
------------------------

Working with temporal data in real-world applications presents several challenges:

- **Multi-resolution data**: Different sensors often record at different frequencies (e.g., 1000Hz accelerometer data alongside 30Hz video)
- **Missing or irregular data**: Gaps in recordings, manual measurements, or variable sampling rates
- **Complex relationships**: Multiple data streams that need to be aligned and analyzed together
- **Large datasets**: Efficient access and processing of temporal segments

Traditional approaches using simple array operations become cumbersome when dealing with these challenges. For example:

.. code-block:: python
    :linenos:

    # Traditional approach has limitations
    import numpy as np
    
    # Need separate arrays for different sensors
    accelerometer = np.random.randn(1000000)  # 1000Hz
    video = np.random.randn(30000)  # 30Hz
    manual_readings = np.random.randn(10)  # Sporadic
    
    # Difficult to align and slice across different timescales
    # How do we get corresponding data for a 5-second window?

temporaldata provides a solution by:

1. Offering flexible data structures that handle multi-resolution and irregular data naturally
2. Providing efficient temporal indexing across all data streams
3. Maintaining temporal relationships between different data sources
4. Enabling fast, on-the-fly access to temporal segments without preprocessing

The package is particularly useful for researchers and engineers working with sensor data, time series analysis, or any application involving complex temporal data structures.


:obj:`ArrayDict`
----------------

The :obj:`ArrayDict` class is designed to manage a collection of arrays that all share the same
first dimension. Arrays can have a different number of dimensions, and different data
types, as long as they share the same first dimension. This class serves as the 
foundational object for more complex structures that will be defined later. Notably, 
:obj:`ArrayDict` is unique among these structures in that it is non-temporal.

You can think of an :obj:`ArrayDict` as a table where each column is an array. Each array 
(or column) can have a different shape, but they all align along the first dimension. 
In the following example, we will create an :obj:`ArrayDict` object to store information about
countries. We will store the country name, the coordinates of the country, and the population
of the country. Note that the coordinates can be stored as a 2D array, instead of two separate
arrays for longitude and latitude.

.. list-table:: Example Table
   :header-rows: 1

   * - Country
     - Longitude
     - Latitude
     - Population
   * - USA
     - -95.71
     - 37.09
     - 331,002,651
   * - Germany
     - 10.45
     - 51.16
     - 83,783,942
   * - Australia
     - 133.77
     - -25.27
     - 25,499,884
   * - Brazil
     - -51.92
     - -14.23
     - 212,559,417
   * - India
     - 78.96
     - 20.59
     - 1,366,417,754
   * - Nigeria
     - 8.67
     - 9.08
     - 206,139,589
   * - ...
     - ...
     - ...
     - ...
   * - Japan
     - 138.25
     - 36.20
     - 126,476,461

.. raw:: html

   <div style="text-align: center; padding-bottom:1em; font-size:1.5em">
      ↓
   </div>

.. code-block:: python
    :linenos:

    import numpy as np

    country_data = ArrayDict(
        country=np.array(["USA", "Germany", ..., "Japan"]),
        coordinates=np.array([[-95.71, 37.09], [10.45, 51.16], ..., [138.25, 36.20]]),
        population=np.array([331002651, 83783942, ..., 126476461]),
        )

    print(country_data)
    >>> ArrayDict(
            country=[10],
            coordinates=[10, 2],
            population=[10]
        )

In fact, any pandas DataFrame can be converted into an :obj:`ArrayDict` object:

.. code-block:: python

    import pandas as pd
    import seaborn as sns
    from temporaldata import ArrayDict

    # Load the iris dataset from seaborn
    iris_df = sns.load_dataset('iris')

    # Convert the iris dataset DataFrame to an ArrayDict
    data = ArrayDict.from_dataframe(iris_df)

    print(data)
    >>> ArrayDict(
            sepal_length=[150],
            sepal_width=[150],
            petal_length=[150],
            petal_width=[150],
            species=[150]
        )

You can create an :obj:`ArrayDict` directly by providing arrays during initialization. 
Below is an example of how to use it:

.. code-block:: python

    import numpy as np
    from temporaldata import ArrayDict

    users = ArrayDict(
        name=np.array(["Alice", "Bob", "Charlie"]),
        age=np.array([25, 30, 35], dtype=np.int32),
        photo=np.random.random((3, 64, 64, 3)),
        measurements=np.random.random((3, 2)),
    )

    print(users)
    >>> ArrayDict(
            name=[3],
            age=[3],
            photo=[3, 64, 64, 3],
            measurements=[3, 2]
        )

    # get the number of samples (or rows) in the ArrayDict
    print(len(users))
    >>> 3

    # get the keys of the ArrayDict
    print(users.keys)
    >>> ['name', 'age', 'photo', 'measurements']

    # check if a key is in the ArrayDict
    print("name" in users)
    >>> True
    
    # arrays can be accessed by key
    print(users.name)
    >>> array(['Alice', 'Bob', 'Charlie'], dtype='<U7')

    # new arrays can be added to the ArrayDict
    users.score = np.array([0.9, 0.8, 0.7])

    print(users)
    >>> ArrayDict(
            name=[3],
            age=[3],
            photo=[3, 64, 64, 3],
            measurements=[3, 2],
            score=[3]
        )

TL;DR
+++++

:obj:`ArrayDict` has the following features:

	1.	Multiple Attributes: An ArrayDict can contain multiple attributes (arrays) with varying numbers of dimensions. This allows for a rich representation of complex data.
	2.	Shared First Dimension: All arrays within an ArrayDict instance share the same size for their first dimension. This allows for consistent indexing across all arrays.
	3.	Variable Dimensions: While the arrays share the same first dimension, they can each have a different number of dimensions. This flexibility allows for diverse data types and structures to be stored within a single ArrayDict.
	4.	No Type Restrictions: There are no limitations on the types of arrays that can be stored in an ArrayDict. This means you can include arrays of integers, floats, strings, or any other data type supported by the array structure you are using.
	5.	Transformation from DataFrame: Any pandas DataFrame can be converted into an ArrayDict object. This feature makes it easy to integrate ArrayDict into workflows that already use pandas for data manipulation and analysis.

:obj:`Interval`
***************

The Interval object builds upon the ArrayDict structure by incorporating the concept of 
time intervals. Each interval is defined by a start time and an end time, which we encode
using arrays. Note that all time attributes should be defined in seconds.

.. code-block:: python
  
      import numpy as np
      from temporaldata import Interval
  
      meetings = Interval(
          start=np.array([10., 3000., 8120.]),
          end=np.array([1810., 4000, 10234.]),
          title=np.array(["1-on-1", "Team Meeting", "Project Review"]),
          location=np.array(["Office", "Conference Room", "Online"]),
          num_attendees=np.array([2, 10, 5]),
          recorded=np.array([False, False, True]),
      )

The Interval class extends the functionality of ArrayDict, inheriting all its 
capabilities while adding specialized methods for managing time intervals:

- :obj:`Interval.is_disjoint()` returns :obj:`True` if the intervals are disjoint, meaning that no two intervals overlap.
- :obj:`Interval.is_sorted()` returns :obj:`True` if the intervals are sorted in ascending order.
- :obj:`Interval.sort()` sorts the intervals in-place based on their start and end times, but will error out if the intervals are not disjoint.

When adding attributes that are time-based, it is important to specify this so that these
attributes are updated accordingly when manipulating the object. This is done by specifying
the `timekeys` attributes when creating the object. 

.. code-block:: python
  
      import numpy as np
      from temporaldata import Interval
  
      trials = Interval(
          start=np.array([0., 1., 2.]),
          end=np.array([1., 2., 3.]),
          go_cue=np.array([0.1, 1.1, 2.1]),
          movement_onset=np.array([0.2, 1.2, 2.2]),
          result=np.array(["success", "failure", "success"])
          reaction_time=np.array([0.1, 0.1, 0.1]),
          timekeys=["start", "end", "go_cue", "movement_onset"],
      )

Note that in this example, `go_cue` and `movement_onset` are time-based attributes, 
but `reaction_time` is not, because it represent a duration rather than a time. By default,
you do not need to specify `timekeys` if only the start and end times are time-based. 

There are many ways to create an Interval object:

- Using the constructor directly, as shown in the examples above.
- :obj:`Interval.from_dataframe()` converts a pandas DataFrame to an Interval object.
- :obj:`Interval.from_list()` converts a list of tuples to an Interval object.
- :obj:`Interval.arange()` creates an Interval object with a specified start time, end time, and step size.
- :obj:`Interval.linspace()` creates an Interval object with a specified start time, end time, and number of intervals.

Use cases
+++++++++

.. tab:: Generic

    .. code-block:: python
  
          import numpy as np
          from temporaldata import Interval
      
          trials = Interval(
              start=np.array([0.2, 10.3, 30.7]),
              end=np.array([7.1, 20.5, 35.2]),
              transcript=np.array(["Hello, how are you?", "I'm good, thanks.", "What's new?"]),
              speaker=np.array(["Alice", "Bob", "Alice"]),
          )

.. tab:: Neuroscience

    .. code-block:: python
  
          import numpy as np
          from temporaldata import Interval
      
          neural_events = Interval(
              start=np.array([0.2, 10.3, 30.7]),
              end=np.array([7.1, 20.5, 35.2]),
              spike_rate=np.array([45.2, 32.1, 67.8]),  # spikes/sec
              brain_state=np.array(["rest", "active", "rest"]),
              cell_type=np.array(["pyramidal", "interneuron", "pyramidal"]),
          )

:obj:`IrregularTimeSeries`
**************************

The :obj:`IrregularTimeSeries` object is another subclass of ArrayDict designed to handle
event-based and irregularly sampled time series data. Unlike traditional time series,
where timestamps are equally spaced, this class accommodates timestamps that can be 
irregularly spaced, making it ideal for data with missing values or events occurring 
at irregular intervals.

.. code-block:: python
  
      import numpy as np
      from temporaldata import IrregularTimeSeries
  
      mouse_clicks = IrregularTimeSeries(
          timestamps=np.array([1.0, 2.1, 4.9]),
          cursor_position=np.array([[10, 20], [15, 25], [5, 10]]),
          button_pressed=np.array([True, False, True]),
      )


:obj:`RegularTimeSeries`
**************************

