# Data Preparation

This file is supposed to include the data preprocessing and transformations applied that resulted in the generation of `processed_data` from the raw data provided.

## 1. Date processing

The first step in the data processing is to make sure we handle dates properly, since the respective column is of string data type. Therefore, we convert it to date type and extract important features such as the month, the day of week, the day of month and year.

| Original feature | Transformation / extracted features |
| --- | --- |
| `date` (string) | Convert to date type |
| `date` | `month`, `day_of_week`, `day_of_month`, `year` |

## 2. Ordinal and binary transformations

Then, we also convert `outlet_volume_potentiality` and `outlet_category_volume_potentiality` to ordinal values (integer) to model the distance between High, Medium, Low and Zero values. Also, `cooler_presence` is converted to binary values `{0, 1}`.

| Feature | Transformation |
| --- | --- |
| `outlet_volume_potentiality` | Ordinal integer encoding |
| `outlet_category_volume_potentiality` | Ordinal integer encoding |
| `cooler_presence` | Binary encoding `{0, 1}` |

## 3. Deduplication

Later, we deduplicate since we could find **87 duplicates**, so we deduplicate.

## 4. Train, validation, and test splitting

We split so that 80% of the total number of records is in the training set (since if we arbitrarily choose a date, records per day are not uniformly distributed and we may have underrepresentation), and 10% in the validation and test sets, respectively.

This results in the following thresholds:

| Boundary | Date |
| --- | --- |
| Training ends | `2023-03-11` |
| Validation ends | `2023-03-28` |
| Test begins after | `2023-03-28` |

After that, we split so that we have:

| Outlet group | Number of outlets |
| --- | ---: |
| Development outlets | 800 |
| Unseen validation outlets | 100 |
| Unseen test outlets | 100 |

Then we create the training set by combining development outlets and dates, one validation set that has unseen outlets and future surveys and another one with seen outlets and future observations. The test set only has surveys from the test date period and completely unseen outlets.

The result of our splitting technique is the following:

| Split | Records | Outlets | Min date | Max date | Availability rate | Unavailable |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `train` | 12,133 | 764 | 2023-01-04 | 2023-03-11 | 0.659441 | 4,132 |
| `val_future_known` | 1,375 | 250 | 2023-03-12 | 2023-03-28 | 0.623273 | 518 |
| `val_future_unseen` | 209 | 40 | 2023-03-13 | 2023-03-28 | 0.703349 | 62 |
| `test_future_unseen` | 193 | 41 | 2023-03-29 | 2023-04-10 | 0.683938 | 61 |

## 5. Outlet attribute stability

It may make sense to create a lookup table for each outlet since, as we can see running the command below, we have stable outlet feature values:

```python
# %% Check outlet attribute stability

outlet_attributes = [
    "outlet_type",
    "sales_performance_indicator",
    "outlet_volume_potentiality",
    "cooler_presence",
    "outlet_satisfaction",
    "price_range_indicator",
    "consumer_sentiment",
    "popularity_with_consumers",
    "consumer_disposable_income_index",
    "consumer_spending_per_capita_on_food_beverages",
    "surrounding_traffic",
    "touristic_point_of_interest",
    "number_of_competing_outlets",
]

stability_results = []

for col_name in outlet_attributes:

    varying_outlets = (
        df.groupBy("outlet")
        .agg(
            F.countDistinct(col_name).alias("n_distinct")
        )
        .filter(F.col("n_distinct") > 1)
        .count()
    )

    stability_results.append({
        "feature": col_name,
        "outlets_with_multiple_values": varying_outlets,
        "percentage_of_outlets": round(
            varying_outlets
            / df.select("outlet").distinct().count()
            * 100,
            2
        )
    })

stability_pd = pd.DataFrame(stability_results)

print(stability_pd.to_string(index=False))
```

## 6. Pending preprocessing steps

Handling missing values, standardizing continuous values, one-hot encoding nominal features, and feature selection is handled in modeling.
