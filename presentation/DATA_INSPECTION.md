# Data Inspection

This markdown file is supposed to include findings from the data inspection.

## Dataset overview

**Dataframe size:** 19,246 records (19 features).

## Dataset schema and feature inspection

### Schema

```text
root
|-- outlet: string (nullable = true)
|-- product: string (nullable = true)
|-- date: string (nullable = true)
|-- availability: integer (nullable = true)
|-- outlet_type: string (nullable = true)
|-- product_type: string (nullable = true)
|-- sales_performance_indicator: integer (nullable = true)
|-- outlet_volume_potentiality: string (nullable = true)
|-- outlet_category_volume_potentiality: string (nullable = true)
|-- cooler_presence: string (nullable = true)
|-- outlet_satisfaction: integer (nullable = true)
|-- price_range_indicator: integer (nullable = true)
|-- consumer_sentiment: double (nullable = true)
|-- popularity_with_consumers: double (nullable = true)
|-- consumer_disposable_income_index: double (nullable = true)
|-- consumer_spending_per_capita_on_food_beverages: double (nullable = true)
|-- surrounding_traffic: double (nullable = true)
|-- touristic_point_of_interest: integer (nullable = true)
|-- number_of_competing_outlets: integer (nullable = true)
```

## Feature inspection

### outlet

The identifier for the outlet where the survey was conducted


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 1,000  |
| Missing values     | 0      |
| Missing percentage | 0%     |


### product

The identifier of the product


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 9      |
| Missing values     | 0      |
| Missing percentage | 0%     |


### date

The date of the survey


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 97     |
| Missing values     | 0      |
| Missing percentage | 0%     |



| earliest_date | latest_date | unique_dates | non_null_dates |
| ------------- | ----------- | ------------ | -------------- |
| 2023-01-04    | 2023-04-10  | 97           | 19246          |


### availability

Binary indicator whether the product was available (1) or not available (0) on the given date and outlet


| Property           | Value |
| ------------------ | ----- |
| Data type          | int   |
| Distinct values    | 2     |
| Missing values     | 0     |
| Missing percentage | 0%    |



| availability | count | percentage |
| ------------ | ----- | ---------- |
| 1            | 12596 | 65.45      |
| 0            | 6650  | 34.55      |


### outlet_type

The type/channel of the outlet

['Kiosk', 'Store', 'Cafe', 'Restaurant']


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 4      |
| Missing values     | 237    |
| Missing percentage | 1.23%  |



| outlet_type | count | percentage |
| ----------- | ----- | ---------- |
| Kiosk       | 6731  | 34.97      |
| Restaurant  | 6040  | 31.38      |
| Cafe        | 4160  | 21.61      |
| Store       | 2078  | 10.8       |
| NULL        | 237   | 1.23       |


### product_type

The type of the product

['WATER', 'JUICE', 'SSD', 'TEA']


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 4      |
| Missing values     | 0      |
| Missing percentage | 0      |



| product_type | count | percentage |
| ------------ | ----- | ---------- |
| SSD          | 12705 | 66.01      |
| WATER        | 2656  | 13.8       |
| TEA          | 2029  | 10.54      |
| JUICE        | 1856  | 9.64       |


### sales_performance_indicator

Indicator ranging from 1 to 5, 1 being the highest in terms of sales performance and 5 being the lowest


| Property           | Value |
| ------------------ | ----- |
| Data type          | int   |
| Distinct values    | 4     |
| Missing values     | 237   |
| Missing percentage | 1.23  |



| sales_performance_indicator | count | percentage |
| --------------------------- | ----- | ---------- |
| 3                           | 7562  | 39.29      |
| 2                           | 5505  | 28.6       |
| 4                           | 5489  | 28.52      |
| 5                           | 453   | 2.35       |
| NULL                        | 237   | 1.23       |


### outlet_volume_potentiality

Potentiality of outlet in terms of volume, compared to other similar outlets. "High" means that there are other similar outlets that have achieved much higher volume, while "Zero" means that the performance is at its maximum. This relates to sales across all product categories

['High', 'Medium', 'Low', 'Zero']


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 4      |
| Missing values     | 237    |
| Missing percentage | 1.23   |



| outlet_volume_potentiality | count | percentage |
| -------------------------- | ----- | ---------- |
| High                       | 7893  | 41.01      |
| Medium                     | 6816  | 35.42      |
| Low                        | 4171  | 21.67      |
| NULL                       | 237   | 1.23       |
| Zero                       | 129   | 0.67       |


### outlet_category_volume_potentiality

Similar to above, but for the specific product category


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 4      |
| Missing values     | 203    |
| Missing percentage | 1.05   |



| outlet_category_volume_potentiality | count | percentage |
| ----------------------------------- | ----- | ---------- |
| Low                                 | 5452  | 28.33      |
| High                                | 5377  | 27.94      |
| Medium                              | 5065  | 26.32      |
| Zero                                | 3149  | 16.36      |
| NULL                                | 203   | 1.05       |


### cooler_presence

Binary indicator whether a cooler is present in the outlet (1), or not (0)

['Yes', 'No']


| Property           | Value  |
| ------------------ | ------ |
| Data type          | string |
| Distinct values    | 2      |
| Missing values     | 237    |
| Missing percentage | 1.23   |



| cooler_presence | count | percentage |
| --------------- | ----- | ---------- |
| Yes             | 16156 | 83.94      |
| No              | 2853  | 14.82      |
| NULL            | 237   | 1.23       |


### outlet_satisfaction

Level of satisfaction of outlet owner with the services (10=Most satisfied, 1=Least satisfied)

int 

distinct values: 10

missing values: 16952

missing percentage: 88.08%


| outlet_satisfaction | count | percentage |
| ------------------- | ----- | ---------- |
| NULL                | 16952 | 88.08      |
| 10                  | 946   | 4.92       |
| 8                   | 417   | 2.17       |
| 9                   | 284   | 1.48       |
| 6                   | 181   | 0.94       |
| 7                   | 124   | 0.64       |
| 5                   | 123   | 0.64       |
| 3                   | 76    | 0.39       |
| 1                   | 69    | 0.36       |
| 2                   | 42    | 0.22       |
| 4                   | 32    | 0.17       |


### price_range_indicator

Indicator regarding the prices of the outlet, as seen through various websites e.g. google, tripadvisor. (4) means very expensive/premium, while (1) means very affordable

int 

distinct values: 4

missing values: 13828

missing percentage: 71.85%


| price_range_indicator | count | percentage |
| --------------------- | ----- | ---------- |
| NULL                  | 13828 | 71.85      |
| 2                     | 2745  | 14.26      |
| 1                     | 2639  | 13.71      |
| 3                     | 33    | 0.17       |
| 4                     | 1     | 0.01       |


### consumer_sentiment

Score from 0 to 100 indicated how the consumers feel about the outlet, based on social media reviews (100=Extremely satisfied, 0=Least satisfied)

double 

distinct values: 398

missing values: 11456

missing percentage: 59.52%


| summary | consumer_sentiment |
| ------- | ------------------ |
| count   | 7790               |
| min     | 20.0               |
| 25%     | 81.41              |
| 50%     | 87.97              |
| mean    | 86.08689216944849  |
| 75%     | 92.45              |
| max     | 100.0              |
| stddev  | 10.068908550919609 |


### popularity_with_consumers

Similar to above, taking into account also the number of reviews, in order to more closely approximate the popularity of the outlet

double 

distinct values: 223

missing values: 11456

missing percentage: 59.52%


| summary | popularity_with_consumers |
| ------- | ------------------------- |
| count   | 7790                      |
| min     | 7.02                      |
| 25%     | 37.31                     |
| 50%     | 48.2                      |
| mean    | 48.12649037227206         |
| 75%     | 59.16                     |
| max     | 95.72                     |
| stddev  | 18.739633474568635        |


### consumer_disposable_income_index

Disposable income index regarding people who live around the outlet. 100 is the mean of the country population, and values above or below indicate more expensive or affordable areas

double 

distinct values: 714

missing values: 237

missing percentage: 1.23%


| summary | consumer_disposable_income_index |
| ------- | -------------------------------- |
| count   | 19009                            |
| min     | 97.402                           |
| 25%     | 118.731                          |
| 50%     | 131.334                          |
| mean    | 136.3028909463958                |
| 75%     | 148.09                           |
| max     | 250.899                          |
| stddev  | 24.83019142102155                |


### consumer_spending_per_capita_on_food_beverages

Yearly spending in euros on food and beverages per capita for people living around the outlet

double 

distinct values: 972

missing values: 237

missing percentage: 1.23%


| summary | consumer_spending_per_capita_on_food_beverages |
| ------- | ---------------------------------------------- |
| count   | 19009                                          |
| min     | 2127.634                                       |
| 25%     | 2634.788                                       |
| 50%     | 2732.442                                       |
| mean    | 2712.605494081735                              |
| 75%     | 2871.304                                       |
| max     | 2984.232                                       |
| stddev  | 177.22000471307734                             |


### surrounding_traffic

Average monthly foot traffic (people passing) around the proximity of the outlet

double 

distinct values: 701

missing values: 237

missing percentage: 1.23%


| summary | surrounding_traffic |
| ------- | ------------------- |
| count   | 19009               |
| min     | 6.045               |
| 25%     | 44.363              |
| 50%     | 63.511              |
| mean    | 75.02134304803114   |
| 75%     | 96.75               |
| max     | 265.375             |
| stddev  | 44.14035721798992   |


### touristic_point_of_interest

Number of touristic point of interest (museums, monuments etc.) around the proximity of the outlet

int 

distinct values: 15

missing values: 237

missing percentage: 1.23%


| summary | touristic_point_of_interest |
| ------- | --------------------------- |
| count   | 19009                       |
| min     | 0                           |
| 25%     | 0                           |
| 50%     | 0                           |
| mean    | 0.8212951759692777          |
| 75%     | 1                           |
| max     | 18                          |
| stddev  | 2.1368128253418566          |


### number_of_competing_outlets

Number of competing outlets (same outlet_type) around the given outlet

int 

distinct values: 22

missing values: 237

missing percentage: 1.23%


| summary | number_of_competing_outlets |
| ------- | --------------------------- |
| count   | 19009                       |
| min     | 0                           |
| 25%     | 1                           |
| 50%     | 2                           |
| mean    | 2.738281866484297           |
| 75%     | 4                           |
| max     | 21                          |
| stddev  | 3.267620521151811           |


## Bivariational Inspection

After inspecting each feature on its own, we dive ddep into inspecting the relationship and correlation between features together.

We ran Pearson correlation over numeric features, we use Cramer's V for categorical features, and to check the numerical to categorical features association we use Eta.

Additionally, we are interested into the correlation of the target variable, "availability" to the rest of the features. Therefore, the first step is to map the categporical features with ordinal information to intergers, trying to model the ordinal information (i.e.sales_performance_indicator, outlet_volume_potnetiality, price_range_indicator, outlet_category_volume_potentiality, outlet_satisfaction) and the binary feature of cooler_presence to 0 - 1 values. Then, we compute the correlation between features and the "availability", using Spearman for ordinal features, Point-biserial for numerical features, and Phi for binary features. The results are show below:


| Feature                                        | Method              | Correlation | N observations |
| ---------------------------------------------- | ------------------- | ----------- | -------------- |
| sales_performance_indicator                    | Spearman            | -0.167544   | 19009          |
| outlet_volume_potentiality_ord                 | Spearman            | -0.054302   | 19009          |
| price_range_indicator                          | Spearman            | -0.048350   | 5418           |
| consumer_sentiment                             | Point-biserial      | -0.033939   | 7790           |
| surrounding_traffic                            | Point-biserial      | -0.026418   | 19009          |
| consumer_spending_per_capita_on_food_beverages | Point-biserial      | -0.012414   | 19009          |
| popularity_with_consumers                      | Point-biserial      | 0.005488    | 7790           |
| number_of_competing_outlets                    | Point-biserial      | 0.021110    | 19009          |
| consumer_disposable_income_index               | Point-biserial      | 0.044448    | 19009          |
| touristic_point_of_interest                    | Point-biserial      | 0.070302    | 19009          |
| outlet_category_volume_potentiality_ord        | Spearman            | 0.100464    | 19043          |
| cooler_presence_binary                         | Phi (binary-binary) | 0.150777    | 19009          |
| outlet_satisfaction                            | Spearman            | 0.171371    | 2294           |


Then we try to identify patterns and trends associated to the availability, therefore, we compute the availability rate by product, product_type, and outlet_type:


| product   | product_type | n_surveys | availability_rate   |
| --------- | ------------ | --------- | ------------------- |
| product_1 | WATER        | 2656      | 0.3569277108433735  |
| product_4 | TEA          | 2029      | 0.43173977328733365 |
| product_8 | SSD          | 1709      | 0.5260386190754828  |
| product_5 | SSD          | 1641      | 0.5886654478976234  |
| product_9 | SSD          | 2457      | 0.6332926332926333  |
| product_6 | SSD          | 2564      | 0.718798751950078   |
| product_3 | JUICE        | 1856      | 0.8362068965517241  |
| product_7 | SSD          | 2566      | 0.8659392049883087  |
| product_2 | SSD          | 1768      | 0.9807692307692307  |


Here is also some stats per outlet:


| summary | n_observations     | n_survey_dates    | n_products        | availability_rate  |
| ------- | ------------------ | ----------------- | ----------------- | ------------------ |
| count   | 1000               | 1000              | 1000              | 1000               |
| mean    | 19.246             | 13.452            | 6.972             | 0.5426299862350773 |
| stddev  | 14.381805182949744 | 8.937313442616388 | 2.020717520906696 | 0.2962236333715604 |
| min     | 1                  | 1                 | 1                 | 0.0                |
| 25%     | 8                  | 7                 | 5                 | 0.375              |
| 50%     | 16                 | 11                | 8                 | 0.5652173913043478 |
| 75%     | 25                 | 18                | 9                 | 0.7916666666666666 |
| max     | 90                 | 55                | 9                 | 1.0                |


It is important here to note that I couldn't identify a certain pattern looking at the number of surveys, number of dates and the availability rate.

Next we observe availability trend over time where we can observe that the availability rate increases significantly in certain periods. It is important to note that the availability heavily depends on the number of surveys and observations.

Here is the availability rate and the number of surveys for almost all of the features:


| cooler_presence                     | n_surveys | availability_rate   |
| ----------------------------------- | --------- | ------------------- |
| NULL                                | 237       | 0.4388185654008439  |
| No                                  | 2853      | 0.4868559411146162  |
| Yes                                 | 16156     | 0.6872369398365932  |
| sales_performance_indicator         | n_surveys | availability_rate   |
| 2                                   | 5505      | 0.7540417801998184  |
| 3                                   | 7562      | 0.6707220312086749  |
| 4                                   | 5489      | 0.5527418473310257  |
| 5                                   | 453       | 0.5187637969094923  |
| NULL                                | 237       | 0.4388185654008439  |
| outlet_volume_potentiality          | n_surveys | availability_rate   |
| Low                                 | 4171      | 0.7283625029968832  |
| Zero                                | 129       | 0.7131782945736435  |
| High                                | 7893      | 0.6429747877866464  |
| Medium                              | 6816      | 0.6289612676056338  |
| NULL                                | 237       | 0.4388185654008439  |
| outlet_category_volume_potentiality | n_surveys | availability_rate   |
| Low                                 | 5452      | 0.7109317681584739  |
| High                                | 5377      | 0.6858843221127022  |
| Medium                              | 5065      | 0.6837117472852912  |
| Zero                                | 3149      | 0.47284852334074307 |
| NULL                                | 203       | 0.39408866995073893 |
| outlet_satisfaction                 | n_surveys | availability_rate   |
| 10                                  | 946       | 0.7653276955602537  |
| 8                                   | 417       | 0.7314148681055156  |
| 6                                   | 181       | 0.7016574585635359  |
| NULL                                | 16952     | 0.6510736196319018  |
| 3                                   | 76        | 0.631578947368421   |
| 9                                   | 284       | 0.6197183098591549  |
| 2                                   | 42        | 0.5714285714285714  |
| 1                                   | 69        | 0.5072463768115942  |
| 5                                   | 123       | 0.4878048780487805  |
| 7                                   | 124       | 0.3951612903225806  |
| 4                                   | 32        | 0.34375             |
| price_range_indicator               | n_surveys | availability_rate   |
| NULL                                | 13828     | 0.7079838009835118  |
| 3                                   | 33        | 0.5454545454545454  |
| 1                                   | 2639      | 0.5430087154225085  |
| 2                                   | 2745      | 0.4936247723132969  |
| 4                                   | 1         | 0.0                 |
| touristic_point_of_interest         | n_surveys | availability_rate   |
| 18                                  | 54        | 0.8888888888888888  |
| 13                                  | 54        | 0.8888888888888888  |
| 15                                  | 54        | 0.8888888888888888  |
| 14                                  | 70        | 0.8857142857142857  |
| 17                                  | 54        | 0.7777777777777778  |
| 4                                   | 387       | 0.7596899224806202  |
| 3                                   | 643       | 0.7433903576982893  |
| 5                                   | 47        | 0.723404255319149   |
| 1                                   | 3483      | 0.7065747918461097  |
| 8                                   | 36        | 0.6944444444444444  |
| 2                                   | 1407      | 0.6666666666666666  |
| 9                                   | 32        | 0.65625             |
| 0                                   | 12592     | 0.6309561626429478  |
| 6                                   | 27        | 0.6296296296296297  |
| 7                                   | 69        | 0.4492753623188406  |
| NULL                                | 237       | 0.4388185654008439  |
| number_of_competing_outlets         | n_surveys | availability_rate   |
| 17                                  | 129       | 0.8294573643410853  |
| 16                                  | 212       | 0.7971698113207547  |
| 14                                  | 75        | 0.7866666666666666  |
| 19                                  | 8         | 0.75                |
| 7                                   | 514       | 0.7276264591439688  |
| 8                                   | 297       | 0.6902356902356902  |
| 18                                  | 29        | 0.6896551724137931  |
| 3                                   | 2073      | 0.6864447660395562  |
| 2                                   | 3647      | 0.6843981354537977  |
| 4                                   | 1467      | 0.6714383094751193  |
| 1                                   | 4533      | 0.6671078755790867  |
| 6                                   | 694       | 0.6296829971181557  |
| 5                                   | 821       | 0.6163215590742996  |
| 0                                   | 3920      | 0.6081632653061224  |
| 13                                  | 176       | 0.6022727272727273  |
| 11                                  | 54        | 0.5555555555555556  |
| 15                                  | 27        | 0.5555555555555556  |
| 12                                  | 75        | 0.52                |
| 9                                   | 223       | 0.4484304932735426  |
| NULL                                | 237       | 0.4388185654008439  |
| 10                                  | 31        | 0.22580645161290322 |
| 20                                  | 2         | 0.0                 |
| 21                                  | 2         | 0.0                 |


