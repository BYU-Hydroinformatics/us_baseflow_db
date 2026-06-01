# Overview

The objective of this project is to develop a national baseflow database for the United States. This will consist of two main parts: (1) a comprehensive database of daily US stream gage data with each date labeled with a 0 or 1 indicating whether the flow on that date is baseflow-dominant or not, and (2) python code that can be used to update the database as new data becomes available.

There are a number of use cases for this database, including:
- Anlyzing trends in baseflow over time and across different regions of the country
- Analyzing the impact of various factors (e.g. land use, climate change) on baseflow
- Correlating baseflow changes with groundwater storage changes, including the use of GRACE data
- Application of the GWBASE algorithm to anlyze the correlation between baseflow and groundwater level changes in selected basins
- Analyzing the predictive skill of PyBFS using a retrospective analysis of the database to identify sequences of baseflow-dominant where the forecast skill can be analyzed.

Historically, researchers have used a variety of methods to identify baseflow-dominant days, including standard baseflow separation techniques. Aghababei et al. (2023) developed a machine learning-based approach to identify baseflow-dominant days, which has shown promise in improving the accuracy of baseflow identification. Xie, et al. (2020) evaluated several typical methods for baseflow separation in the contiguous United States, providing insights into the strengths and weaknesses of each method. They used the Strict algorithm to judge the performance of each method, which is a commonly used algorithm for baseflow separation. 

Each of these techniques has its own advantages and disadvantages, and the choice of method will depend on the specific research question being addressed. For example, machine learning-based approaches peform well when compared with the hand-labeled datasets they were trained on, but may not perform as well when applied to new data that differs from the training data. The separation algorithms also can produce a baseflow estimate that is greater than the total streamflow. On the other hand, standard baseflow separation techniques can perform well on some gages, but may not perform well on others, and may require manual tuning of parameters. In our experience, the Strict algorithm is too restrictive in some cases, and in other cases includes data points high on the recession limb of the hydrograph that are not baseflow-dominant.

Our hypothesis is that an ensemble approach that combines multiple methods for identifying baseflow-dominant days will outperform any single method. By leveraging the strengths of each method, we can improve the accuracy and robustness of our baseflow identification. We will evaluate the performance of our ensemble approach using a combination of hand-labeled datasets and standard baseflow separation techniques, including the Strict algorithm. For the separation techniques, daily streamflow values will be considered baseflow-dominant if the baseflow component is within X% of the total streamflow. The final determination as to whether a day is baseflow-dominant will be made by the ensemble approach - for example, if a day is labeled as baseflow-dominant by a majority (or some percentage) of the methods. Users of the database and associated tools will be able to adjust these thresholds as needed for their specific applications.

# Data Sources

The primary data source for this project will be the USGS National Water Information System (NWIS), which provides daily streamflow data for thousands of gages across the United States. We will aggregate these data into daily values for each gage.

# Methods

We will use three main sets of methods to identify baseflow-dominant days: (1) machine learning-based approaches, (2) standard baseflow separation techniques, and (3) the PyBFS algorithm.

## Machine Learning-Based Approach

This will be the based on the aghababei et al. (2023) approach, which uses a machine learning model to identify baseflow-dominant days based on a variety of features derived from the streamflow data. We will simply run this approach on our dataset to get a classification for each day.

## Standard Baseflow Separation Techniques

For the baseflow separation, we will use the algorithms provided in the baseflowx python package. 

https://github.com/BYU-Hydroinformatics/baseflowx

https://pypi.org/project/baseflowx/

https://baseflow-explorer.onrender.com/

https://baseflowx.readthedocs.io/en/latest/

We will need to explore the parameters used by the baseflow separation algorithms to determine the optimal settings for our application. The when regenerating the national database, we will use the same parameters for all gages to ensure consistency. However, users of the database and associated tools will be able to adjust these parameters as needed for their specific applications.

## PyBFS Algorithm

We should also use PyBFS to identify baseflow-dominant days.

https://pybfs.readthedocs.io/en/latest/

https://pybfs.readthedocs.io/en/latest/

https://pypi.org/project/pybfs/

## Ensemble Approach

For the ensemble approach, we will combine the results from the machine learning-based approach, the standard baseflow separation techniques, and the PyBFS algorithm. For the standard baseflow separation techniques and the PyBFS algorithm, we will consider a day to be baseflow-dominant if the baseflow component is within X% of the total streamflow. 

The final determination as to whether a day is baseflow-dominant will be made by the ensemble approach - for example, if a day is labeled as baseflow-dominant by a majority (or some percentage) of the methods. We will evaluate different methods for combining these results, such as majority voting or weighted averaging, to determine the best approach for identifying baseflow-dominant days.




Xie, J., Liu, X., Wang, K., Yang, T., Liang, K., & Liu, C. (2020). Evaluation of typical methods for baseflow separation in the contiguous United States. Journal of Hydrology, 583, 124628. https://doi.org/10.1016/j.jhydrol.2020.124628