# Exploratory Data Analysis & Model Selection

This document summarizes the tests, observations, and conclusions from experiments.

## 1. Data Preparation
* We used both categorical features (like Location, Company, Contract Type) and text features (Job Description).
* Missing values in categorical columns were filled with the word "Missing".
* We applied a logarithmic transformation (np.log1p) to the target variable (SalaryNormalized). This helps the neural network learn better because salaries have a skewed distribution.

## 2. Tested Models
We experimented with many different neural network architectures to find the best one:
1. **Baseline Model:** Uses only categorical features.
2. **Self-Taught Model:** Learns text embeddings from scratch during training.
3. **Word2Vec Models (Way 1, 2, 3):** Uses pre-trained Google Word2Vec embeddings. We tested frozen weights, fine-tuning, and handling rare words.
4. **Advanced Architectures:** We added 1D Convolutions, Residual Blocks, Dropout, Batch Normalization, and Attention Pooling to the models to see if they improve the score.

## 3. Results and Statistical Testing
We evaluated the models on the Test Set using RMSLE (Root Mean Squared Logarithmic Error) and MAE (Mean Absolute Error). 

**Top 3 Models:**
1. Self-taught with 6 Residual Blocks (RMSLE: 0.067)
2. Self-taught with Conv1D (RMSLE: 0.073)
3. Basic Self-taught (RMSLE: 0.074)

**Statistical Tests:**
* We ran a Friedman Chi-Square test. The p-value was 0.0000, which means there is a significant difference between all models.
* We then ran a Nemenyi post-hoc test to compare the top models. The test showed **no statistically significant difference** between the top 3 models.

## 4. Final Model Selection
Because the top 3 models have very similar performance, we looked at their complexity:
* The 6x ResBlock model is heavy (~3.2 million parameters).
* The Conv1D model is very slow to train.
* **The Basic Self-taught model** is light (~2.5 million parameters) and very fast to train (around 9 seconds).

**Conclusion:** We selected the **Basic Self-taught model** as our final choice. It gives excellent results while being the most efficient and fastest to train.

## 5. Error Analysis and Data Issues
We analyzed the biggest mistakes made by our winning model. We found two interesting things:

1. **Data Inconsistency:** In the top 10 worst predictions, the model predicted around £40k-50k, but the actual data showed £5k-7k. When we looked at the job descriptions, we noticed the salary was given per month, not per year. Our model actually made a good prediction for an annual salary, but the data was misleading.
2. **High-Paying Jobs:** The model is generally unbiased, but it struggles with extreme values. It often underpredicts salaries for very high-paying jobs. When it does predict a high salary, it tends to overestimate it. Basically, it is hard for the model to perfectly value extreme, top-tier roles.
