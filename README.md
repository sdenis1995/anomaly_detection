Project dedicated to supercomputer job classification method (excluding monitoring data extraction and processing).

To use this method, following packages are needed:
For Python:
rpy2, sklearn

For R:
ecp

The main method includes following steps: gathering time series data, detecting change points, classifying resulting
intervals with Random Forest classifier and getting the class for the job depending on interval classes.

job_classifier.py includes main methods of job classification. They are implemented in AnomalyDetector class. It has
methods for change point detection using ecp package for R ( get_change_points ), getting change points and resulting
interval classes ( process_job ), and getting resulting class for the job ( predict ). There is get_job_class function,
that returns the class of the function based on interval classes, changepoints and number of processes that job was run
on, that is used in predict method on AnomalyDetector class. You can change the function to whatever criteria that suits
your environment.

All main parameters for methods can be adjusted in config.cfg file.
For ecp package, there are only two parameters that matter the most: cluster size (ecp_min_cluster_size) and change
point significance level (ecp_significance_level). Cluster size is the minimal number of points between change points,
and significance level is needed for determining, whether to consider selected point as change point (the higher the
value - more likely the point will be considered as change point).
For Random Forest classifier the main parameter is number of trees (NumTreesForClassifier).

In our method, we also postprocess detected change points to pack them into their own small intervals if needed, so they
wouldn't interfere with the features of the neighbouring intervals. For that purpose, we have janky postfilter function
in functions.py. You can use your own postfilter functions and ignore ours.

We also include functions implementing feature addition method to select set of features that show best accuracy. All
methods are included in feature_addition_lib.py. To select best feature to add in each step, cross validation is used.
To remove any fluctuations in accuracy, several iterations of cross validation is performed. The amount of iteration and
amount of subsets used in cross validation is set in config file (cross_val_iters_for_feature_addition and cross_val_cv
respectively). Also, for cross validation Random Forest is used, and the amount of trees in the classifier is set by
NumTreesForFeatureSelection option in config file (it is better to set it to lower value than in the main classifier to
speed feature addition method up).
Main method is feature_addition_parallel, it has intuitive parameters. base_features is array of initial features to start
the method with (array of indices), if it is set to None then random features are selected (amount of random features are
set by default_random_feature_count option in config file). There are several ways to finish the process of feature selection,
we implemented two of them: when we reach certain accuracy and certain number of features. It is controlled by breakpoint_type
parameter (allowed values are 'limit' and 'accuracy'). feature_addition_feature_limit option in config sets the limit
of number of features for 'limit' breakpoint type, feature_addition_accuracy_limit option sets the accuracy for 'accuracy'
breakpoint type.


Software is being developed in Research Computing Center, Lomonosov Moscow State University.