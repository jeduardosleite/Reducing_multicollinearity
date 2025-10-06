[My Linkedin](https://www.linkedin.com/in/jos%C3%A9-eduardo-souza-leite/)
- pip install -r requirements.txt

# Indice

- [Introduction](#module-35---exercise-1)
- [Packs](#packs)
- [Dataset](#dataset)
- [Model distribution](#model-distribution)
- [Independence from waste](#independence-from-waste)
- [Outliers](#outliers)
  - [Residual student](#residual-student)
- [Influence](#influence)
  - [Cooks distance](#cooks-distance)
  - [Hat_diag (leverage)](#hat_diag-leverage)
  - [Analysis of the two metrics](#analysis-of-the-two-metrics)
- [Why the 99th percentile?](#why-the-99th-percentile)
- [Suspects](#suspects)
- [Multicollinearity](#multicollinearity)
  - [Matrix correlation](#matrix-correlation)
  - [Correlation interpretation](#correlation-interpretation)
  - [Correlation of variables](#correlation-of-variables)
- [VIF (Variance Inflation Factor)](#vif-variance-inflation-factor)
  - [How to mitigate the VIF value?](#how-to-mitigate-the-vif-value)
- [Final conclusion](#final-conclusion)

<img width="833" height="450" alt="image" src="https://github.com/user-attachments/assets/9064a65c-c825-4b19-bf5c-7d005ea9c1e6" />

# Exercise 1 - Module 35

This activity aims to continue the data analysis from the previous exercise. This time, I will explore the concept of multicollinearity, exploring its parameters, identification, and ways to address this problem that distorts the model.

Initially, I won't focus on improving R-squared, but rather on increasing its F-statistic, that is, its statistical reliability.

Another important point for this exercise is that I will be using the last 3 months.

[🔝 Back to indice](#Indice)
---

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

[🔝 Back to indice](#Indice)
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

[🔝 Back to indice](#Indice)
---

## Model Distribution

The data cluster around a central value, forming a bell-shaped pattern. Therefore, we can conclude a normal distribution. 

<img width="592" height="291" alt="image" src="https://github.com/user-attachments/assets/d1709d05-00e6-49db-b266-bd17d32a3277" />

[🔝 Back to indice](#Indice)
---

## Independence from waste

Overall, the medians of the residuals were close to zero, suggesting that there is no strong bias by category. However, many values ​​of *-3* or *3* were noted, highlighting possible outliers, which will be analyzed and addressed later.

Among these variables, I highlight two that caught my attention:

1) ```tempo_emprego```, which presented a funnel shape, indicating heteroscedasticity that will be investigated.

2) ```tipo_renda``` shows greater dispersion and more positive outliers, highlighting heterogeneity.

<img width="857" height="896" alt="image" src="https://github.com/user-attachments/assets/f92a9f31-c8b2-4637-970c-4ec3593e1327" />

[🔝 Back to indice](#Indice)
---

## Outliers

### Residual Student

All the values ​​above are above 3, meaning they are clear outliers in the model.
- Positive residuals: model underestimated income
- Negative residuals: model overestimated income

However, not every outlier is necessarily a problem; it may indicate that the model does not explain certain combinations of variables well.
An outlier becomes critical when the influence (measured by Cooks Distance) is high. This will be analyzed below.

<img width="224" height="396" alt="image" src="https://github.com/user-attachments/assets/57535c2e-1a7f-44e0-8f5f-6c730c0ba9de" />

[🔝 Back to indice](#Indice)
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

[🔝 Back to indice](#Indice)
---

## Why the 99th percentile?

Since the dataset contains 150,000 rows, the 99th percentile aims to capture only the extreme cases, highlighting the truly influential cases without getting lost in the thousands of *mild suspicions*.

Note that, previously, just over **5,000** suspects were identified. This percentile aims to filter potential cases within this universe, making the analysis more efficient and assertive.

[🔝 Back to indice](#Indice)
---

## Suspects

I maked a dataframe correlationing the suspects with dataset original, that is, those who exceeded the established limit for **cook's distance** and **leverage**. 

```python
new_test.loc[suspeitos.index]
```

<img width="743" height="569" alt="image" src="https://github.com/user-attachments/assets/7a3edf8d-227b-4d98-94b3-b86ac4141310" />

Originally, the dataset had **150.000** rows, now with the suspect filter, it has **507** rows, this making it more easly for analyse and interpretation.

[🔝 Back to indice](#Indice)
---

## Multicollinearity

This is the central question of this exercise: how to increase ```f-score``` and reduce ```multicollinearity```?

### Matrix Correlation

Firstly, i maked a table with values interpretation paramns. Then, i analyzed the correlation between *"renda_log"* and the other variables.

### Correlation Interpretation

| Value       | Interpretation                                                   |
|--------------|------------------------------------------------------------------|
| 1            | Perfect positive correlation (as one increases, the other always increases) |
| 0.7–0.9      | Strong positive correlation                                      |
| 0.4–0.6      | Moderate positive correlation                                   |
| 0.1–0.3      | Weak positive correlation                                       |
| 0            | No monotonic correlation                                        |
| -0.1 to -0.3 | Weak negative correlation                                       |
| -0.4 to -0.6 | Moderate negative correlation                                  |
| -0.7 to -0.9 | Strong negative correlation                                    |
| -1           | Perfect negative correlation (as one increases, the other always decreases) |

---

### Correlation of Variables

| # | variável | Correlation | Strength |
|---|-----------|-------------|-----------|
| 0 | tempo_emprego | 0.600791 | Moderate positive |
| 1 | idade | 0.123072 | Weak positive |
| 2 | tipo_renda_Servidor público | 0.120360 | Weak positive |
| 29 | tipo_renda_Pensionista | -0.089147 | No correlation |
| 3 | posse_de_imovel_S | 0.080226 | No correlation |
| 4 | educacao_Superior completo | 0.048848 | No correlation |
| 28 | tipo_residencia_Com os pais | -0.046270 | No correlation |
| 5 | tipo_residencia_Casa | 0.038186 | No correlation |
| 27 | educacao_Superior incompleto | -0.032969 | No correlation |
| 26 | educacao_Médio | -0.030996 | No correlation |
| 25 | qtd_filhos_1 | -0.025695 | No correlation |
| 24 | estado_civil_Solteiro | -0.022533 | No correlation |
| 23 | tipo_residencia_Comunitário | -0.022150 | No correlation |
| 6 | tipo_residencia_Estúdio | 0.014872 | No correlation |
| 22 | estado_civil_União | -0.014698 | No correlation |
| 7 | tipo_renda_Empresário | 0.013164 | No correlation |
| 8 | estado_civil_Separado | 0.010371 | No correlation |
| 9 | qtd_filhos_4 | 0.008531 | No correlation |
| 21 | educacao_Pós graduação | -0.006755 | No correlation |
| 20 | estado_civil_Viúvo | -0.006096 | No correlation |
| 10 | qtd_filhos_7 | 0.005748 | No correlation |
| 19 | qtd_filhos_3 | -0.005550 | No correlation |
| 11 | tipo_renda_Bolsista | 0.005375 | No correlation |
| 18 | qtd_filhos_14 | -0.004630 | No correlation |
| 12 | sexo_M | 0.004393 | No correlation |
| 13 | qtd_filhos_2 | 0.004187 | No correlation |
| 17 | qt_pessoas_residencia | -0.003239 | No correlation |
| 14 | posse_de_veiculo_S | 0.002884 | No correlation |
| 16 | tipo_residencia_Governamental | -0.001923 | No correlation |
| 15 | qtd_filhos_5 | -0.000756 | No correlation |

[🔝 Back to indice](#Indice)
---

## VIF (Variance Inflation Factor)

Measures how much the variance of a regression coefficient is inflated due to multicollinearity—that is, when an explanatory variable is correlated with others in the model.

- ```VIF = 1``` → no multicollinearity.
- ```VIF between 1 and 5``` → moderate brightness, generally acceptable.
- ```VIF > 10``` → strong multicollinearity, indicating a problem in the model.

The VIF values ​​for some variables are extremely high. Variables such as number_of_people_in_residence, type_of_residence_house, and education_medium have very high values. But why does this happen?

It happens because there are high correlations between these variables. This causes very high VIFs because the VIF measures how much one variable can be predicted by the others. The more linearly dependent, the higher the VIF.

<img width="248" height="758" alt="image" src="https://github.com/user-attachments/assets/57973b7a-96ea-437c-939c-8138cb6f425b" />

### How to mitigate the VIF value?

To mitigate these values, I applied **PCA** to transform correlated variables into **orthogonal components**. By performing this substitution, all values ​​decreased (to some extent).

Before showing the results, I'll explain the code.

1) I grouped correlated variables. When analyzing the dataframe without PCA, I noticed that the variables related to **children**, **education**, and **residence** had extremely high VIF values.

```python
filhos_cols = ['qt_pessoas_residencia', 'qtd_filhos_1', 'qtd_filhos_2', 
               'qtd_filhos_3', 'qtd_filhos_4', 'qtd_filhos_5', 'qtd_filhos_7', 
               'qtd_filhos_14']

educ_cols = ['educacao_Médio', 'educacao_Superior completo', 'educacao_Superior incompleto', 'educacao_Pós graduação']

resid_cols = ['tipo_residencia_Casa', 'tipo_residencia_Com os pais', 'tipo_residencia_Estúdio', 
              'tipo_residencia_Comunitário', 'tipo_residencia_Governamental']

idade_tempo_cols = ['idade', 'tempo_emprego']
```

2) I applied **PCA** separately to each group, generating ortogonal components:
```python
filhos_pca = aplicar_pca(new_test_encoded, filhos_cols, n_components=2, prefix='filhos_PC')
educ_pca = aplicar_pca(new_test_encoded, educ_cols, n_components=2, prefix='educ_PC')
resid_pca = aplicar_pca(new_test_encoded, resid_cols, n_components=2, prefix='resid_PC')
idade_pca = aplicar_pca(new_test_encoded, idade_tempo_cols, n_components=1, prefix='idade_PC')
```

3) Then, i replaced the original variables with their ```principal components```, maintaining interpretability:
```python
cols_para_remover = filhos_cols + educ_cols + resid_cols
new_test_pca = pd.concat([
    new_test_encoded.drop(columns=cols_para_remover),
    filhos_pca, educ_pca, resid_pca
], axis=1)
```

4) I converted the remaining booleand variables to 0/1:
```python
bool_cols = new_test_pca.select_dtypes(include=['bool']).columns
new_test_pca[bool_cols] = new_test_pca[bool_cols].astype(int)
```

5) Finally, i calculated the **VIF** for the final variables:
```python
vif_data_pca = pd.DataFrame()
vif_data_pca['variavel'] = X_vif.columns
vif_data_pca['VIF'] = [vif(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data_pca = vif_data_pca.sort_values(by='VIF', ascending=False)
```

The **result**: the ```VIFs``` original variables dropped drastically. I was to eliminate almost all multicollinearity from the dataset.

<img width="276" height="573" alt="image" src="https://github.com/user-attachments/assets/ec607d89-4614-4458-9ebe-054e856124a6" />

[🔝 Back to indice](#Indice)
---

## Final Conclusion

### Tabela do OLS Regression Results

#### Before of PCA
- **F-statistic**: 3582
- **Df Model**:: 51
- **R²**:: 0.549

#### After of PCA
- **F-statistic**:: 5067
- **Df Model**:: 36
- **R²**:: 0.537

Reducing multicollinearity through PCA improved the F-statistic, making the model more *statistically robust*, with more *reliable coefficients*, *less noise*, and *less redundancy* between variables.

Although R² decreased slightly, maintaining the model is justified, considering the greater reliability of the results.

To specifically increase the OLS R², it would be necessary to:
- Add new variables that better explain renda_log;
- Evaluate nonlinear models, such as **Random Forest** or **Gradient Boosting**, which capture complex interactions between variables.

For the purposes of this analysis, the main result was achieved: reducing multicollinearity while maintaining the model's reliability and interpretability.

<img width="566" height="220" alt="image" src="https://github.com/user-attachments/assets/2cf44a00-5ad6-4344-af7c-9b8bab39901e" />

<img width="569" height="222" alt="image" src="https://github.com/user-attachments/assets/3b6a61e4-b58c-4713-b72b-31f9f65220c7" />
