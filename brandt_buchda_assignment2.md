# Question 1
## Part A
The model retains the following features after backwards selection:

| Predictor             | SSE    | Size | F-Statistic | F-Significance | F DoF 1 | F DoF 2 |
|-----------------------|--------|------|-------------|----------------|---------|---------|
| Tract Median Income   | 0.0251 | 12   | 4.5846      | 3.2879e-02     | 1       | 390     |
| Wall Material         | 0.0256 | 10   | 4.2433      | 5.7380e-03     | 3       | 390     |
| Bedrooms              | 0.0254 | 12   | 10.4144     | 1.3561e-03     | 1       | 390     |
| Garage Size           | 0.0259 | 12   | 18.1204     | 2.6006e-05     | 1       | 390     |
| Full Baths            | 0.0262 | 12   | 23.0524     | 2.2506e-06     | 1       | 390     |
| Land Acre             | 0.0265 | 12   | 26.9214     | 3.4148e-07     | 1       | 390     |
| Basement              | 0.0272 | 10   | 12.7510     | 5.7634e-08     | 3       | 390     |
| Building Square Feet  | 0.0274 | 12   | 40.9112     | 4.5645e-10     | 1       | 390     |


The following features were selected for removal:

|Predictor               |SSE       |Size      |F-Statistic    |F-Significance   |F DoF 1        |F DoF 2        |
|------------------------|----------|----------|---------------|-----------------|---------------|---------------|
|Central Air Conditioning|0.0242    |17        |0.6973         |4.0421e-01       |1              |385            |
|Half Baths              |0.0242    |16        |1.1058         |2.9365e-01       |1              |386            |
|Roof Material           |0.0246    |14        |2.7794         |6.3317e-02       |2              |387            |
|Age                     |0.0248    |13        |3.1723         |7.5676e-02       |1              |389            |

## Part B
![leverage.png](plots%2Fleverage.png)

## Part C
![residuals.png](plots%2Fresiduals.png)

## Part D
Selecting a leverage cutoff value of 0.15 selects the following observations for removal:

| Leverage | Intercept | Tract Median Income | Garage Size | Land Acre | Full Baths | Bedrooms | Building Square Feet | Basement | Wall Material | Prediction | Sale Price |
| -------- | --------- | ------------------- | ----------- | --------- | ---------- | -------- | -------------------- | -------- | ------------- | ---------- | ---------- |
| 0.5184   | 1         | 142.6               | 2.5         | 0.06777   | 5          | 5        | 5.956                | Full     | Stucco        | 3202       | 1950       |
| 0.5184   | 1         | 94.6                | 3           | 0.1033    | 4          | 4        | 3.952                | Full     | Stucco        | 2169       | 3675       |

These two observations are the only observations with Stucco for wall material therefore they apply significant leverage
to the prediction.

The following are the outlier observations that appear outside the Q1 minus 1.5 * the Inter quartile range in one or 
more of the residuals. In most case, the observation appears as an outlier in all forms of residual

| Sale Price | Prediction | Intercept | Tract Median Income | Building Square Feet | Land Acre | Garage Size | Bedrooms | Full Baths | Wall Material | Basement |
| ---------- | ---------- | --------- | ------------------- | -------------------- | --------- | ----------- | -------- | ---------- | ------------- | -------- |
| 700        | 1411       | 1         | 132.2               | 2.52                 | 0.0576    | 3           | 3        | 3          | Frame         | Full     |
| 895        | 2464       | 1         | 138.9               | 3.78                 | 0.07117   | 2           | 5        | 5          | Masonry       | Full     |
| 3000       | 1.799e+04  | 1         | 128.2               | 8.306                | 0.08386   | 3           | 5        | 5          | Masonry       | Full     |
| 1200       | 3322       | 1         | 141.8               | 5.019                | 0.0781    | 2           | 5        | 4          | Masonry       | Partial  |

