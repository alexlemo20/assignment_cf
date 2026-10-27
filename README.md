# HOW TO RUN THE CODE
First, the virtual environment is needed to be created running
``py
python -m venv .venv
``
Then, after activating the environment, the dependencies need to be instaled running
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

We start with the data inspection.
The first thing we did, was a univariate analysis on the dataset and its features, to understand each feature independently. The next step will be a bivariate analysis and understanding how features relate to one another and to product availability. 

More infromation can be found in presentation/DATA_INSPECTION.md.

The next step is to apply the data preprocessing to prepare the data for the model to be trained. We need to give to the model ordinal features, month - week and dayof the week information, handle missing values somehow. Split to check how the model performs in unseen outlets, unseen products (this may not be possible since we only have 9 products and then the dataset will include very few data points) and future dates. The thinking for this process can be found in presentation/DATA_PREPARATION.md.

We need to use an avg predictor, a by group predictor, logreg and xgboost, include historical information on the sequences. Experiemnt with ordinal vs one-hot encoding. Experiment with and without some features. An important assumption is that: To prevent information leakage you need to make sure that feature values are not computed on information available after the survey date. We could also use a small neural netwrok for the task (explore the embedding space for outlets etc). More information on the predictive modeling can be found under presentation/PREDICTIVE_MODELING.md.


# DATA & EXPLORATION
Explore the data and provide insights into any patterns or trends in product availability. Are there particular outlets or dates where product availability for target products is consistently high or low? Are there any interesting elements among the outlets attributes that were provided? Suggest potential factors or reasons for low availability and propose strategies for improvement (could be outside the ones given in the dataset).

# PREDICTIVE MODELING
Using an appropriate machine learning algorithm, develop an approach that will predict what is the probability that an outlet with given characteristics is going to have the target product available. There is no constraint regarding the modeling approach you choose to follow. Discuss the strengths and weaknesses of the modeling approach you chose, the model performance and suggest possible improvements.

# TARGETING
The dataset that was provided contains a small subset of the outlets and products. In reality there are hundreds of products and hundreds of thousands of outlets. Today, based on simple grouping criteria (example all cafes), a given product is set as target for the outlet. For example, product_1 is a target for cafes and column "available" indicates if the target is met or not. Imagine you are tasked with automating this targeting and develop an approach that will automatically identify which outlets should be matched with which products as targets. How would you approach it as a problem, what are the key questions you would ask from business stakeholders, what are the business and technical aspects that should be considered, in order to develop this as an end-to-end solution?

