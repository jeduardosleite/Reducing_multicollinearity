# Module 35 - Exercise 1

<img width="833" height="450" alt="image" src="https://github.com/user-attachments/assets/9064a65c-c825-4b19-bf5c-7d005ea9c1e6" />

This activity aims to continue the data analysis from the previous exercise. This time, I will explore the concept of multicollinearity, exploring its parameters, identification, and ways to address this problem that distorts the model.

Initially, I won't focus on improving R-squared, but rather on increasing its F-statistic, that is, its statistical reliability.

Another important point for this exercise is that I will be using the last 3 months.

## Packs
```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
import scipy.stats as ss 
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
```

---

## Dataset

Initially, I applied the log to "renda" to adjust its value to the model. This was a tactic used in previous work that brought benefits to interpretation.

```python
# Fill nulls in tempo_emprego with the average
df['tempo_emprego'] = df['tempo_emprego'].fillna(df['tempo_emprego'].mean())

# applied the log
df['renda_log'] = np.log(df['renda'])

# Converting to date time
df['data_ref'] = pd.to_datetime(df['data_ref'])

# Extracting the last 3 months
ultima_data = df['data_ref'].max()
tres_meses = ultima_data - pd.DateOffset(months=3)

# Separating into train and test
train = df[df['data_ref'] <= tres_meses]
test = df[df['data_ref'] > tres_meses]
```

## Model Distribution

The data cluster around a central value, forming a bell-shaped pattern. Therefore, we can conclude a normal distribution. 

<img width="592" height="291" alt="image" src="https://github.com/user-attachments/assets/d1709d05-00e6-49db-b266-bd17d32a3277" />

---

## Independence from waste

Into the notebook we can see the complete analysis from boxplot and scatter plot graphs.

<img width="857" height="896" alt="image" src="https://github.com/user-attachments/assets/f92a9f31-c8b2-4637-970c-4ec3593e1327" />

---

## Outliers

### Residual Student

All the values ​​above are above 3, meaning they are clear outliers in the model.
- Positive residuals: model underestimated income
- Negative residuals: model overestimated income

However, not every outlier is necessarily a problem; it may indicate that the model does not explain certain combinations of variables well.
An outlier becomes critical when the influence (measured by Cooks Distance) is high. This will be analyzed below.

<img width="224" height="396" alt="image" src="https://github.com/user-attachments/assets/57535c2e-1a7f-44e0-8f5f-6c730c0ba9de" />

---

## Influence

### Cooks Distance
Measures the influence of each point on the model fit. The higher the value, the more influential.

### Hat_diag (Leverage)
Measures how isolated the X value is from the others. It ranges from **0** to **1**; the closer to 1, the more isolated it is in the predictor space. If it is higher, this is an outlier point in *x*.

Generally speaking, to present a risk to the data, both values ​​must be high.

- A point can have high **leverage** but not be influential.
- A point can have a high *residual*, that is, a significant **cook distance**; but if it is isolated in the X values, it is also not influential.

### Analysis of the two metrics

To identify the lines where **both metrics** are significant, I created the code below. I set a threshold for these two metrics:

```python
limite_cook = resumo_influencia["cooks_distance"].quantile(0.99)
limite_hat = resumo_influencia["alavancagem"].quantile(0.99)
```

Following the logic that *an isolated metric doesn't provide much information*, I applied the filter with the thresholds to the dataframe, where the output will be the individuals who exceed **both metrics**.

```python
suspeitos = resumo_influencia[
    (resumo_influencia["cooks_distance"] > limite_cook) &
    (resumo_influencia["alavancagem"] > limite_hat)
]
```

## Why the 99th percentile?

Since the dataset contains 150,000 rows, the 99th percentile aims to capture only the extreme cases, highlighting the truly influential cases without getting lost in the thousands of *mild suspicions*.

Note that, previously, just over **5,000** suspects were identified. This percentile aims to filter potential cases within this universe, making the analysis more efficient and assertive.

---

























