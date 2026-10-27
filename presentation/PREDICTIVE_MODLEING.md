# Predictive modelling — experiments and results

This file is supposed to include information about the second part of the assignment.

The approach will be to start with something simpler and gradually increase the complexity of the machine learning approach to be used to address the task of computing the probability of a product to be available at a certain outlet.

An assumption is that fields of outlets are not computed based on future observations, therefore it is not considered information leakage using those features and values as input to the model.

# Non-parametric approach

The first approach to be tested is a non-parametric one that can only be used as a baseline to benchmark the rest of the models.

We basically compute based on the training dataset the global availability probability (global average), the availability rate by outlet_type, by product and the combination of both. This results into 4 baselines.

For missing values we just create a new label "*_MISSING_*" since this approach is the safest one, compared to assuming the most popular value.

The results are shown below:

| model | validation | alpha | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| product_outlet_type | future_known | 0.0 | 1375 | 0.623273 | 0.669452 | 0.488761 | 0.157980 | 0.765091 | 0.840719 | 0.761665 |
| product | future_known | 0.0 | 1375 | 0.623273 | 0.651324 | 0.570165 | 0.194940 | 0.685818 | 0.745783 | 0.579587 |
| outlet_type | future_known | 0.0 | 1375 | 0.623273 | 0.688182 | 0.610758 | 0.210545 | 0.675636 | 0.692683 | 0.533070 |
| global | future_known | 0.0 | 1375 | 0.623273 | 0.659441 | 0.665306 | 0.236112 | 0.623273 | 0.500000 | 0.376727 |
| product_outlet_type | future_unseen | 0.0 | 209 | 0.703349 | 0.695852 | 0.411491 | 0.127079 | 0.822967 | 0.871516 | 0.761180 |
| outlet_type | future_unseen | 0.0 | 209 | 0.703349 | 0.714405 | 0.517545 | 0.171303 | 0.717703 | 0.759985 | 0.506403 |
| product | future_unseen | 0.0 | 209 | 0.703349 | 0.659330 | 0.550311 | 0.184956 | 0.717703 | 0.721253 | 0.491879 |
| global | future_unseen | 0.0 | 209 | 0.703349 | 0.659441 | 0.612391 | 0.210577 | 0.703349 | 0.500000 | 0.296651 |

Here is the respective results on the test set:

| model | alpha | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| product_outlet_type | 0 | 193 | 0.683938 | 0.678902 | 0.427147 | 0.139889 | 0.766839 | 0.872889 | 0.708850 |
| product | 0 | 193 | 0.683938 | 0.644368 | 0.513224 | 0.172984 | 0.730570 | 0.784277 | 0.566895 |
| outlet_type | 0 | 193 | 0.683938 | 0.699491 | 0.513537 | 0.169934 | 0.725389 | 0.781110 | 0.553927 |
| global | 0 | 193 | 0.683938 | 0.659441 | 0.625218 | 0.216767 | 0.683938 | 0.500000 | 0.316062 |

### Experiment summary — future-unseen validation

| No. | Experiment / configuration | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ | PR AUC (unavailable) ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Global availability rate | 0.612391 | 0.210577 | 0.703349 | 0.500000 | 0.296651 |
| 2 | Availability by product | 0.550311 | 0.184956 | 0.717703 | 0.721253 | 0.491879 |
| 3 | Availability by outlet type | 0.517545 | 0.171303 | 0.717703 | 0.759985 | 0.506403 |
| 4 | Availability by product and outlet type | 0.411491 | 0.127079 | 0.822967 | 0.871516 | 0.761180 |

The detailed results for both validation sets remain in the experiment tables above.

# Logistic regression

First of all, we exclude "outlet" since we want to prevent our model memorizing the outlet and currenlty we are not interested in historical data.

We exclude the raw date since it does not make sense to include a string as a date (we include the extracted day of the week, month and day of month). Also, including both the product and the product_type may not add information since those two features are perfectly correlated.

For the numeric missing values we experiemnt with using the mean and the median (which makes more sense epecially for integer values), while for the categorical ones we experiment using a new "MISSING" value, and the most popular value.

Also, we standardize numerical values so that all those features are put in a comparable scale based on the training mean and standard deviation.

The first experiment includes using all the features as numeric values, l2 loss function and all of the features available and the results are shown below:

| model | experiment | validation | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | future_known | 1375 | 0.623273 | 0.678700 | 0.505296 | 0.165997 | 0.751273 | 0.825807 | 0.752852 |
| logistic_regression | logreg_default | future_unseen | 209 | 0.703349 | 0.702239 | 0.412610 | 0.130315 | 0.827751 | 0.869980 | 0.744956 |

Here are the results on the unseen test set:

| model | experiment | dataset | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | test_future_unseen | 193 | 0.683938 | 0.685588 | 0.371595 | 0.114002 | 0.854922 | 0.903751 | 0.842721 |

Next, we repeat the experiment with just excluding the "product_type", since that should add redundant information since we already have the "product" field. The results are shown below:

| model | experiment | validation | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | future_known | 1375 | 0.623273 | 0.678573 | 0.505163 | 0.165973 | 0.751273 | 0.825714 | 0.752822 |
| logistic_regression | logreg_default | future_unseen | 209 | 0.703349 | 0.702140 | 0.412563 | 0.130311 | 0.827751 | 0.869980 | 0.744773 |

The results indicate negligible improvements.

The next experiment includes one-hot encoding the following features: ["sales_performance_indicator", "outlet_volume_potentiality", "outlet_category_volume_potentiality", "outlet_satisfaction", "price_range_indicator", "month", "day_of_week", "day_of_month"] instead of using ordinal encoding. This results in more coefficients and a bigger model. The results are shown below:

| model | experiment | validation | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | future_known | 1375 | 0.623273 | 0.741842 | 0.538621 | 0.177290 | 0.732364 | 0.831316 | 0.758419 |
| logistic_regression | logreg_default | future_unseen | 209 | 0.703349 | 0.769134 | 0.435626 | 0.139515 | 0.784689 | 0.863068 | 0.743712 |

The performance decreases and the reason is because we do not give to the model any information on how close "High" is from "Low" and whether "Medium" is closer. Therefore, that is something expected. However, looking at the coefficients we observe that now it is detected that the "*_MISSING_*" value in outlet_category_volume_potentiality contributes to a lower availability probability.

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| categorical__product_product_2 | 2.757330 | 2.757330 |
| categorical__product_product_8 | -1.451601 | 1.451601 |
| categorical__product_product_5 | -1.186213 | 1.186213 |
| categorical__product_product_7 | 0.980195 | 0.980195 |
| categorical__outlet_type_Cafe | -0.979070 | 0.979070 |
| categorical__outlet_type_Kiosk | 0.836064 | 0.836064 |
| categorical__outlet_category_volume_potentiality___MISSING__ | -0.748182 | 0.748182 |

The next experiment, will be to see the impact of missing values handling. Therefore, we change from using median for numerical values and an additional "*_MISSING_*" value for categorical features, to the mean and the most frequent value, respectively. Here are the results:

| model | experiment | validation | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | future_known | 1375 | 0.623273 | 0.678556 | 0.505297 | 0.166006 | 0.751273 | 0.825671 | 0.752635 |
| logistic_regression | logreg_default | future_unseen | 209 | 0.703349 | 0.702104 | 0.412591 | 0.130302 | 0.827751 | 0.870200 | 0.745238 |

The results are very close, which shows that there is no certain patterns followed in the missing categorical values and that may not give significant information to the model. Ths is something we could tell from the percentage of missing values in categorical features. For numerical features, the mean and median are close, anyways.

The next experiemnt is to exclude features with high missing values frequency: ["outlet_satisfaction", "price_range_indicator", "consumer_sentiment" "popularity_with_consumers"]. The results are shown below:

| model | experiment | validation | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | future_known | 1375 | 0.623273 | 0.679777 | 0.503448 | 0.165150 | 0.755636 | 0.827539 | 0.755133 |
| logistic_regression | logreg_default | future_unseen | 209 | 0.703349 | 0.701061 | 0.416250 | 0.131597 | 0.832536 | 0.863507 | 0.743822 |

This experiment shows that the last model slightly outperforms the model trained on the entire set of features.

The last experiment includes experiment with a different regularizer. We use elasticnet instead of l2, saga solver and l1_ratio = 0.5 and we get the following table:

| model | experiment | validation | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logreg_default | future_known | 1375 | 0.623273 | 0.678574 | 0.505265 | 0.165963 | 0.751273 | 0.825804 | 0.752847 |
| logistic_regression | logreg_default | future_unseen | 209 | 0.703349 | 0.702129 | 0.412806 | 0.130338 | 0.827751 | 0.869761 | 0.744778 |

Again, the results seem to be very close.

For error analysis, we also inspect the coeafficients with the greatest absolute values:

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| categorical__product_product_2 | 2.770192 | 2.770192 |
| categorical__product_product_8 | -1.433995 | 1.433995 |
| categorical__product_product_5 | -1.154362 | 1.154362 |
| categorical__product_product_7 | 0.992078 | 0.992078 |
| categorical__outlet_type_Cafe | -0.927472 | 0.927472 |
| categorical__outlet_type_Kiosk | 0.891909 | 0.891909 |
| categorical__outlet_type_Store | 0.757379 | 0.757379 |
| categorical__product_product_3 | 0.627208 | 0.627208 |
| categorical__product_type_JUICE | 0.627208 | 0.627208 |
| categorical__product_type_SSD | 0.624893 | 0.624893 |
| categorical__product_product_1 | -0.572112 | 0.572112 |
| categorical__product_type_WATER | -0.572112 | 0.572112 |
| categorical__product_product_9 | -0.419019 | 0.419019 |
| numeric__sales_performance_indicator | -0.346340 | 0.346340 |
| categorical__outlet_type_Restaurant | -0.330951 | 0.330951 |
| categorical__product_type_TEA | -0.284043 | 0.284043 |
| categorical__product_product_4 | -0.284043 | 0.284043 |
| numeric__outlet_volume_potentiality_ord | -0.241322 | 0.241322 |
| numeric__consumer_disposable_income_index | 0.183832 | 0.183832 |
| binary__cooler_presence_binary | 0.178037 | 0.178037 |
| numeric__number_of_competing_outlets | 0.150191 | 0.150191 |
| numeric__outlet_category_volume_potentiality_ord | 0.135177 | 0.135177 |
| numeric__missingindicator_outlet_category_volume_potentiality_ord | -0.132664 | 0.132664 |
| numeric__surrounding_traffic | -0.130090 | 0.130090 |
| categorical__product_product_6 | -0.130002 | 0.130002 |

We also print the failures of our model, when the magnitude of the actual availability and predicted availability is bigger by product and outlet_type.

**Error analysis by product**

| product | n_observations | actual_availability | predicted_availability | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- |
| product_3 | 23 | 0.739130 | 0.871345 | 0.202734 | 0.621234 |
| product_1 | 33 | 0.393939 | 0.413592 | 0.217195 | 0.618048 |
| product_4 | 16 | 0.625000 | 0.497455 | 0.177142 | 0.524788 |
| product_5 | 22 | 0.590909 | 0.637715 | 0.151414 | 0.471023 |
| product_8 | 20 | 0.600000 | 0.538386 | 0.086419 | 0.339627 |
| product_9 | 23 | 0.739130 | 0.769644 | 0.102204 | 0.333133 |
| product_6 | 24 | 0.833333 | 0.751194 | 0.098853 | 0.330250 |
| product_7 | 29 | 0.931034 | 0.891262 | 0.063111 | 0.219676 |
| product_2 | 19 | 0.947368 | 0.985348 | 0.050695 | 0.214855 |

**Error analysis by outlet_type**

| outlet_type | n_observations | actual_availability | predicted_availability | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- |
| Restaurant | 64 | 0.484375 | 0.539415 | 0.228687 | 0.667827 |
| NaN | 3 | 0.000000 | 0.277052 | 0.096974 | 0.346063 |
| Store | 26 | 0.846154 | 0.814415 | 0.105765 | 0.345146 |
| Cafe | 19 | 0.421053 | 0.392348 | 0.097219 | 0.344894 |
| Kiosk | 97 | 0.886598 | 0.853214 | 0.079554 | 0.278048 |

We also want to examine high-confidence errors:

| outlet | product | outlet_type | survey_date | y_true | y_prob | absolute_error |
| --- | --- | --- | --- | --- | --- | --- |
| outlet_21846 | product_2 | Restaurant | 2023-03-28 | 0 | 0.978125 | 0.978125 |
| outlet_2956 | product_3 | Kiosk | 2023-03-27 | 0 | 0.948731 | 0.948731 |
| outlet_35811 | product_3 | Kiosk | 2023-03-20 | 0 | 0.919635 | 0.919635 |
| outlet_22901 | product_3 | Store | 2023-03-28 | 0 | 0.910772 | 0.910772 |
| outlet_13475 | product_7 | Restaurant | 2023-03-27 | 0 | 0.859575 | 0.859575 |
| outlet_21241 | product_1 | Restaurant | 2023-03-22 | 1 | 0.150231 | 0.849769 |
| outlet_13475 | product_3 | Restaurant | 2023-03-23 | 0 | 0.816773 | 0.816773 |
| outlet_874 | product_1 | Restaurant | 2023-03-26 | 1 | 0.226415 | 0.773585 |
| outlet_874 | product_1 | Restaurant | 2023-03-13 | 1 | 0.232943 | 0.767057 |
| outlet_2956 | product_5 | Kiosk | 2023-03-23 | 0 | 0.764881 | 0.764881 |

### Experiment summary — future-unseen validation

| No. | Experiment / configuration | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ | PR AUC (unavailable) ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | All features, ordinal encoding, median/missing, L2 | 0.412610 | 0.130315 | 0.827751 | 0.869980 | 0.744956 |
| 2 | Exclude product_type | 0.412563 | 0.130311 | 0.827751 | 0.869980 | 0.744773 |
| 3 | One-hot ordinal/calendar features | 0.435626 | 0.139515 | 0.784689 | 0.863068 | 0.743712 |
| 4 | Mean / most frequent imputation | 0.412591 | 0.130302 | 0.827751 | 0.870200 | 0.745238 |
| 5 | Exclude high-missing features | 0.416250 | 0.131597 | 0.832536 | 0.863507 | 0.743822 |
| 6 | Elastic Net, SAGA, l1_ratio=0.5 | 0.412806 | 0.130338 | 0.827751 | 0.869761 | 0.744778 |

The detailed results for both validation sets remain in the experiment tables above.

# XGBoost

The next approach to be tested is xgboost. We will use the undeen validation for early stopping. Again, for the preprocessor and how to handle missing values, we use the training data to compute most frequent and median and then we apply the same transformations to the validation and test sets.

Here are the results when we use the configuration used for logistic regression, ordinal features and median for missing values handling.

| model | experiment | validation | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | future_known | 402 | 1375 | 0.623273 | 0.682356 | 0.431273 | 0.135536 | 0.813818 | 0.881139 | 0.844006 |
| xgboost | xgboost_default | future_unseen | 402 | 209 | 0.703349 | 0.711565 | 0.395931 | 0.122078 | 0.822967 | 0.877332 | 0.791864 |

This clearly shows that XG Boost performs significantly better than the best logistic regression approach.

Also, experimenting with "native" value for handling the missing values (instead of using the median) we observe a decrease on the performance of the model:

| model | experiment | validation | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | future_known | 142 | 1375 | 0.623273 | 0.671637 | 0.456225 | 0.146821 | 0.794182 | 0.865451 | 0.822517 |
| xgboost | xgboost_default | future_unseen | 142 | 209 | 0.703349 | 0.699387 | 0.411419 | 0.128075 | 0.832536 | 0.871681 | 0.771935 |

Tat can be the case since the percentage of missing values is not high enough, and therefore, xgboost may create a separate branch for missing values which may be less effective and more noisy.

We also want to experiment with one hot encoding vs ordinal features, therefore, when one hot encoding ["month", "day_of_week", "day_of_month","sales_performance_indicator", "outlet_volume_potentiality_ord", "outlet_category_volume_potentiality_ord", "outlet_satisfaction", "price_range_indicator"] we get:

| model | experiment | validation | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | future_known | 214 | 1375 | 0.623273 | 0.672810 | 0.446557 | 0.142556 | 0.804364 | 0.870718 | 0.829589 |
| xgboost | xgboost_default | future_unseen | 214 | 209 | 0.703349 | 0.704053 | 0.406197 | 0.125910 | 0.813397 | 0.873820 | 0.784124 |

which shows a slight decrease in the performance of the model, consistently with the results on logistic regression.

For xgboost and error, analysis we can observe the importance of features, which reflects the contribution of features in splits

| feature | importance |
| --- | --- |
| categorical__product_type_WATER | 0.131024 |
| categorical__product_product_1 | 0.078757 |
| categorical__product_product_2 | 0.077738 |
| categorical__outlet_type_Kiosk | 0.062647 |
| categorical__outlet_type_Cafe | 0.062089 |
| categorical__outlet_type_Restaurant | 0.059935 |
| categorical__product_product_7 | 0.054961 |
| categorical__product_product_3 | 0.053473 |
| categorical__product_type_SSD | 0.040584 |
| categorical__product_product_6 | 0.039646 |
| categorical__product_product_9 | 0.027381 |
| categorical__product_type_TEA | 0.027112 |
| numeric__sales_performance_indicator | 0.025680 |
| categorical__product_product_8 | 0.024593 |
| categorical__product_product_4 | 0.021640 |
| categorical__product_type_JUICE | 0.018304 |
| numeric__price_range_indicator | 0.017863 |
| numeric__outlet_volume_potentiality_ord | 0.017722 |
| categorical__outlet_type_Store | 0.017563 |
| numeric__outlet_category_volume_potentiality_ord | 0.016363 |
| numeric__cooler_presence_binary | 0.014994 |
| numeric__popularity_with_consumers | 0.011825 |
| categorical__product_product_5 | 0.011699 |
| numeric__consumer_disposable_income_index | 0.010693 |
| numeric__consumer_spending_per_capita_on_food_beverages | 0.010259 |

For error analysis, we follow similar approach to logistic regression, printing out the samples caused our model to fail:

**Error analysis by product**

| product | n_observations | actual_availability | predicted_availability | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- |
| product_3 | 23 | 0.739130 | 0.889012 | 0.199253 | 0.672561 |
| product_1 | 33 | 0.393939 | 0.413074 | 0.194093 | 0.562903 |
| product_4 | 16 | 0.625000 | 0.551932 | 0.168490 | 0.508415 |
| product_5 | 22 | 0.590909 | 0.635212 | 0.149070 | 0.459419 |
| product_6 | 24 | 0.833333 | 0.796215 | 0.112912 | 0.336331 |
| product_8 | 20 | 0.600000 | 0.533440 | 0.074995 | 0.308473 |
| product_9 | 23 | 0.739130 | 0.742967 | 0.078025 | 0.280623 |
| product_2 | 19 | 0.947368 | 0.986594 | 0.051196 | 0.230467 |
| product_7 | 29 | 0.931034 | 0.904186 | 0.054275 | 0.185809 |

**Error analysis by outlet_type**

| outlet_type | n_observations | actual_availability | predicted_availability | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- |
| NaN | 3 | 0.000000 | 0.420920 | 0.206044 | 0.596601 |
| Restaurant | 64 | 0.484375 | 0.558477 | 0.186120 | 0.574402 |
| Store | 26 | 0.846154 | 0.815817 | 0.141586 | 0.444468 |
| Cafe | 19 | 0.421053 | 0.423953 | 0.127984 | 0.403815 |
| Kiosk | 97 | 0.886598 | 0.849954 | 0.070841 | 0.257417 |

We also decide to increase the depth of the trees to max_depth=5 to allow bigger model complexity, the results show a small improvement, which is expected since even the previous model has converged, however we add parameters - expressivity to the model

| model | experiment | validation | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | future_known | 117 | 1375 | 0.623273 | 0.667101 | 0.417843 | 0.131295 | 0.827636 | 0.889058 | 0.853417 |
| xgboost | xgboost_default | future_unseen | 117 | 209 | 0.703349 | 0.697443 | 0.390894 | 0.120825 | 0.851675 | 0.886000 | 0.793587 |

Since we observe overfitting and early stopping, it absolutely makes sense to also experiment with a regularization parameter and increasing the number of min_child_weight which increases the requirement of support for splitting decisions. The results show more iterations, which is expected but the final results appear to be very close:

| model | experiment | validation | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | future_known | 123 | 1375 | 0.623273 | 0.670097 | 0.422410 | 0.133558 | 0.819636 | 0.885795 | 0.848253 |
| xgboost | xgboost_default | future_unseen | 123 | 209 | 0.703349 | 0.705205 | 0.395662 | 0.122640 | 0.832536 | 0.881720 | 0.793087 |

Here is also the performance of a model trained on only core features, excluding high missing values features ["outlet_satisfaction", "price_range_indicator", "consumer_sentiment", "popularity_with_consumers"]:

| model | experiment | validation | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | future_known | 96 | 1375 | 0.623273 | 0.675071 | 0.473720 | 0.154528 | 0.785455 | 0.856719 | 0.80610 |
| xgboost | xgboost_default | future_unseen | 96 | 209 | 0.703349 | 0.694755 | 0.421997 | 0.132533 | 0.832536 | 0.872120 | 0.77565 |

The increase on the performance using xgboost, compared to logistic regression is expected since we jump to more expressivity:

Number of trees: 433

Total leaves: 3265

Total split nodes: 2832

Here is how the model performs on the test set

| model | experiment | dataset | best_iteration | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | xgboost_default | test_future_unseen | 402 | 193 | 0.683938 | 0.682288 | 0.335644 | 0.1008 | 0.870466 | 0.926043 | 0.87 |

### Experiment summary — future-unseen validation

| No. | Experiment / configuration | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ | PR AUC (unavailable) ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Ordinal features, median imputation (default) | 0.395931 | 0.122078 | 0.822967 | 0.877332 | 0.791864 |
| 2 | Native handling of missing values | 0.411419 | 0.128075 | 0.832536 | 0.871681 | 0.771935 |
| 3 | One-hot ordinal/calendar features | 0.406197 | 0.125910 | 0.813397 | 0.873820 | 0.784124 |
| 4 | Increase max_depth to 5 | 0.390894 | 0.120825 | 0.851675 | 0.886000 | 0.793587 |
| 5 | Increase regularisation and min_child_weight | 0.395662 | 0.122640 | 0.832536 | 0.881720 | 0.793087 |
| 6 | Exclude high-missing features | 0.421997 | 0.132533 | 0.832536 | 0.872120 | 0.77565 |

The detailed results for both validation sets remain in the experiment tables above.

# Neural network

The last approach to be tested is using neural networks to address the problem.

We use the following architecture:

```text
AvailabilityMLP(

  (network): Sequential(

    (0): Linear(in_features=47, out_features=32, bias=True)

    (1): ReLU()

    (2): Dropout(p=0.1, inplace=False)

    (3): Linear(in_features=32, out_features=16, bias=True)

    (4): ReLU()

    (5): Dropout(p=0.1, inplace=False)

    (6): Linear(in_features=16, out_features=1, bias=True)

  )

)

Number of trainable parameters: 2081
```

The first experiment gives:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | mlp_default | train | 6 | 2081 | 12133 | 0.659441 | 0.671290 | 0.397361 | 0.127405 | 0.820902 | 0.887737 | 0.807278 |
| mlp | mlp_default | future_known | 6 | 2081 | 1375 | 0.623273 | 0.717385 | 0.487100 | 0.148610 | 0.796364 | 0.862400 | 0.817431 |
| mlp | mlp_default | future_unseen | 6 | 2081 | 209 | 0.703349 | 0.733329 | 0.389790 | 0.118283 | 0.846890 | 0.883366 | 0.788205 |

Since we observe that the best model is found at epoch 6, we suspect overfitting, therefore, we decrease the sice of the model dropping the hidden layers from 32->16 to 16->8 and 913 total parameters. Here are the results shown below:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | mlp_default | train | 29 | 913 | 12133 | 0.659441 | 0.670061 | 0.372439 | 0.117405 | 0.832276 | 0.902243 | 0.838015 |
| mlp | mlp_default | future_known | 29 | 913 | 1375 | 0.623273 | 0.691197 | 0.446760 | 0.135358 | 0.824727 | 0.875484 | 0.837298 |
| mlp | mlp_default | future_unseen | 29 | 913 | 209 | 0.703349 | 0.706969 | 0.361706 | 0.107963 | 0.870813 | 0.895655 | 0.823930 |

The results indicate a significant improvement in the model performance.

Next, we continue with increasing the dropout to 0.3 which is supposed to help with regularization.  No significant improvements were observed:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | mlp_default | train | 49 | 913 | 12133 | 0.659441 | 0.669639 | 0.373051 | 0.117595 | 0.838457 | 0.903813 | 0.840952 |
| mlp | mlp_default | future_known | 49 | 913 | 1375 | 0.623273 | 0.682626 | 0.459445 | 0.138581 | 0.814545 | 0.871107 | 0.827157 |
| mlp | mlp_default | future_unseen | 49 | 913 | 209 | 0.703349 | 0.706251 | 0.363369 | 0.110379 | 0.856459 | 0.896204 | 0.819214 |

Next we decrease the learning rate from 1e-3 to 3e-4. The results, again, do not show significant impact.

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | mlp_default | train | 57 | 913 | 12133 | 0.659441 | 0.670776 | 0.386418 | 0.122722 | 0.825929 | 0.894759 | 0.821746 |
| mlp | mlp_default | future_known | 57 | 913 | 1375 | 0.623273 | 0.693291 | 0.463122 | 0.141660 | 0.816000 | 0.866991 | 0.821904 |
| mlp | mlp_default | future_unseen | 57 | 913 | 209 | 0.703349 | 0.712911 | 0.373277 | 0.112980 | 0.866029 | 0.886877 | 0.780063 |

Note that here the validation set used for early stopping is small and can be noisy, therefore, it is important not to rely on the results heavily, however, they can still give a picture of how the model performs.

Here is how the smaller, best network performs on the test set:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | mlp_default | test_future_unseen | 29 | 913 | 193 | 0.683938 | 0.697445 | 0.34116 | 0.101283 | 0.860104 | 0.913065 | 0.848372 |

Here is when trying to capture patterns of failure modes of our model:

**Error analysis by product**

| product | n_observations | actual_availability | predicted_availability | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- |
| product_3 | 23 | 0.739130 | 0.862099 | 0.190586 | 0.679776 |
| product_4 | 16 | 0.625000 | 0.545688 | 0.187053 | 0.579819 |
| product_1 | 33 | 0.393939 | 0.437072 | 0.183710 | 0.538452 |
| product_5 | 22 | 0.590909 | 0.630609 | 0.123385 | 0.402979 |
| product_6 | 24 | 0.833333 | 0.792059 | 0.087921 | 0.280046 |
| product_8 | 20 | 0.600000 | 0.540104 | 0.057699 | 0.240578 |
| product_2 | 19 | 0.947368 | 0.990714 | 0.051525 | 0.236099 |
| product_7 | 29 | 0.931035 | 0.896662 | 0.049609 | 0.170606 |
| product_9 | 23 | 0.739130 | 0.707057 | 0.031707 | 0.134084 |

**Error analysis by outlet_type**

| outlet_type | n_observations | actual_availability | predicted_availability | brier_score | log_loss |
| --- | --- | --- | --- | --- | --- |
| Restaurant | 64 | 0.484375 | 0.496805 | 0.162516 | 0.535855 |
| Store | 26 | 0.846154 | 0.829879 | 0.135477 | 0.406315 |
| Cafe | 19 | 0.421053 | 0.432934 | 0.110087 | 0.374339 |
| NaN | 3 | 0.000000 | 0.286288 | 0.084960 | 0.340260 |
| Kiosk | 97 | 0.886598 | 0.879378 | 0.064889 | 0.233034 |

Additionally, we use one hot encoding for ordinal and calendar features and we observe the following results:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | mlp_default | train | 16 | 1857 | 12133 | 0.659441 | 0.656795 | 0.388881 | 0.124558 | 0.818759 | 0.893960 | 0.820064 |
| mlp | mlp_default | future_known | 16 | 1857 | 1375 | 0.623273 | 0.725018 | 0.496860 | 0.151060 | 0.792727 | 0.863745 | 0.824739 |
| mlp | mlp_default | future_unseen | 16 | 1857 | 209 | 0.703349 | 0.755089 | 0.396661 | 0.120089 | 0.837321 | 0.886438 | 0.792226 |

The number of parameters increases significantly which makes the model easier to overfit (this is also reflected in the best epoch).

### Experiment summary — future-unseen validation

| No. | Experiment / configuration | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ | PR AUC (unavailable) ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 → 16 hidden layers; dropout 0.1 | 0.389790 | 0.118283 | 0.846890 | 0.883366 | 0.788205 |
| 2 | 16 → 8 hidden layers; dropout 0.1 | 0.361706 | 0.107963 | 0.870813 | 0.895655 | 0.823930 |
| 3 | 16 → 8 hidden layers; dropout 0.3 | 0.363369 | 0.110379 | 0.856459 | 0.896204 | 0.819214 |
| 4 | Learning rate 3e-4 | 0.373277 | 0.112980 | 0.866029 | 0.886877 | 0.780063 |
| 5 | One-hot ordinal/calendar features | 0.396661 | 0.120089 | 0.837321 | 0.886438 | 0.792226 |

The detailed results for both validation sets remain in the experiment tables above.

## Feature-based outlets (outlet embeddings)

An additional experiment that coudld be useful is to embed the outlets separately, and then pass the outlet embeddings with the rest of the features to the model. This can also offer better explainability since we could observe which outlets appear closer to each other and the features they share.

For the first model experiment we use the following model architecture:

```text
OutletEmbeddingMLP(

  (outlet_encoder): Sequential(

    (0): Linear(in_features=31, out_features=16, bias=True)

    (1): ReLU()

    (2): Linear(in_features=16, out_features=8, bias=True)

  )

  (prediction_head): Sequential(

    (0): Linear(in_features=29, out_features=16, bias=True)

    (1): ReLU()

    (2): Dropout(p=0.1, inplace=False)

    (3): Linear(in_features=16, out_features=1, bias=True)

  )

)

Trainable parameters: 1145
```

Here is the performance of the first model:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outlet_embedding_mlp | outlet_embeddings_default | train | 18 | 1145 | 12133 | 0.659441 | 0.669538 | 0.378599 | 0.120860 | 0.825352 | 0.898109 | 0.827423 |
| outlet_embedding_mlp | outlet_embeddings_default | future_known | 18 | 1145 | 1375 | 0.623273 | 0.684472 | 0.454326 | 0.136783 | 0.815273 | 0.872839 | 0.838069 |
| outlet_embedding_mlp | outlet_embeddings_default | future_unseen | 18 | 1145 | 209 | 0.703349 | 0.693193 | 0.380623 | 0.117941 | 0.827751 | 0.891047 | 0.792482 |

We also experiment using 4 embedding dimension, which results in 1013 parameters:

```text
OutletEmbeddingMLP(
  (outlet_encoder): Sequential(

    (0): Linear(in_features=31, out_features=16, bias=True)

    (1): ReLU()

    (2): Linear(in_features=16, out_features=4, bias=True)

  )

  (prediction_head): Sequential(

    (0): Linear(in_features=25, out_features=16, bias=True)

    (1): ReLU()

    (2): Dropout(p=0.1, inplace=False)

    (3): Linear(in_features=16, out_features=1, bias=True)

  )

)

Trainable parameters: 1013
```

and we get the following results:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outlet_embedding_mlp | outlet_embeddings_default | train | 23 | 1013 | 12133 | 0.659441 | 0.658537 | 0.373129 | 0.117969 | 0.836314 | 0.901300 | 0.832276 |
| outlet_embedding_mlp | outlet_embeddings_default | future_known | 23 | 1013 | 1375 | 0.623273 | 0.668985 | 0.434732 | 0.132655 | 0.821091 | 0.876425 | 0.842531 |
| outlet_embedding_mlp | outlet_embeddings_default | future_unseen | 23 | 1013 | 209 | 0.703349 | 0.682490 | 0.358671 | 0.111539 | 0.832536 | 0.900044 | 0.832273 |

Decreasing the embedding dimension from 8 to 4 decreases the parameters of the model and prevents overfitting, and that is why we observe better generalization performance.

Note that before proceeding with the above outlet representation, we have checked that outlet features are stable across all observations. Outlet features are considered to be the following:

["sales_performance_indicator", "outlet_volume_potentiality_ord", "outlet_satisfaction", "price_range_indicator", "consumer_sentiment", "popularity_with_consumers", "consumer_disposable_income_index", "consumer_spending_per_capita_on_food_beverages", "surrounding_traffic", "touristic_point_of_interest", "number_of_competing_outlets", "cooler_presence_binary", "outlet_type"].

Next, we decide to inspect the embedding space to see if there is anything reasonable captured. Running cosine similarity and comparing the closest 5 outlets to "outlet_1170" we observe that the embeddings seem to capture useful and valuable information:

| Feature | outlet_1170 | outlet_5024 | outlet_27750 | outlet_361 | outlet_8211 | outlet_44886 |
| --- | --- | --- | --- | --- | --- | --- |
| cosine_similarity | 1.0 | 0.996458 | 0.994525 | 0.993796 | 0.991845 | 0.991293 |
| sales_performance_indicator | 4.0 | 3.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| outlet_volume_potentiality_ord | 3.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| outlet_satisfaction | NaN | 8.0 | NaN | 10.0 | NaN | NaN |
| price_range_indicator | NaN | NaN | NaN | NaN | NaN | NaN |
| consumer_sentiment | 96.11 | 96.67 | 98.57 | 100.0 | 100.0 | NaN |
| popularity_with_consumers | 35.33 | 35.33 | 41.05 | 17.88 | 15.11 | NaN |
| consumer_disposable_income_index | 115.231 | 123.433 | 103.506 | 137.278 | 122.659 | 117.102 |
| consumer_spending_per_capita_on_food_beverages | 2637.11 | 2681.389 | 2344.436 | 2886.508 | 2658.37 | 2662.476 |
| surrounding_traffic | 25.835 | 36.105 | 38.42 | 69.188 | 55.54 | 82.83 |
| touristic_point_of_interest | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| number_of_competing_outlets | 3.0 | 1.0 | 1.0 | 0.0 | 1.0 | 3.0 |
| cooler_presence_binary | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| outlet_type | Restaurant | Restaurant | Restaurant | Restaurant | Restaurant | Restaurant |

### Experiment summary — future-unseen validation

| No. | Experiment / configuration | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ | PR AUC (unavailable) ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Embedding dimension 8 | 0.380623 | 0.117941 | 0.827751 | 0.891047 | 0.792482 |
| 2 | Embedding dimension 4 | 0.358671 | 0.111539 | 0.832536 | 0.900044 | 0.832273 |

The detailed results for both validation sets remain in the experiment tables above.

Here is the performance of the outlet embedding model on the test set:

| model | experiment | dataset | best_epoch | n_parameters | n_samples | availability_rate | mean_predicted_probability | log_loss | brier_score | accuracy | roc_auc | pr_auc_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outlet_embedding_mlp | outlet_embeddings_default | test_future_unseen | 23 | 1013 | 193 | 0.683938 | 0.680569 | 0.344339 | 0.103595 | 0.870466 | 0.914555 | 0.859704 |


---

# Comparison of best models

The validation entries below use the configuration with the lowest **future-unseen validation log loss** within each model family. Test results are shown for the configurations explicitly reported in the original notes. Where the documented test experiment is not the same as the validation-winning configuration, its configuration is labelled separately rather than assuming equivalence.

| Model | Configuration | Dataset | Log loss ↓ | Brier ↓ | Accuracy ↑ | ROC AUC ↑ | PR AUC (unavailable) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Non-parametric | product + outlet_type | Test future unseen | 0.427147 | 0.139889 | 0.766839 | 0.872889 | 0.708850 |
| Neural network (MLP) | 16 → 8 hidden layers, dropout = 0.1 | Test future unseen | 0.34116 | 0.101283 | 0.860104 | 0.913065 | 0.848372 |
| Outlet embeddings | Embedding dimension = 4 | Test future unseen | 0.344339 | 0.103595 | 0.870466 | 0.914555 | 0.859704 |
| Logistic regression | All features, ordinal encoding (reported test run) | Test future unseen | 0.371595 | 0.114002 | 0.854922 | 0.903751 | 0.842721 |
| XGBoost | Default depth-3 configuration (reported test run) | Test future unseen | 0.335644 | 0.1008 | 0.870466 | 0.926043 | 0.87 |

