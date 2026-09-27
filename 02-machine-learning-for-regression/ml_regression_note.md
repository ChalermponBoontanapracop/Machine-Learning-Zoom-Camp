# Machine Learning for Linear Regression

## Linear Regression Process
In Regression Project, it is a Machine Learning Project where the Target is a numerical value that we want the Model to predict. The Dataset consists of Features (X) that are used as input and a Target (y) that we want to predict. We can set up the project in the following steps:

1.	Get the Data and Exploratory Data Analysis (EDA): Explore the Data and understand its characteristics.
2.	Data Preparation: Prepare the Dataset so that it can be used for Machine Learning. Train a Linear Regression Model: Use the prepared Data to train a Model for predicting a numerical Target.
3.	Linear Regression Implementation: Understand how Linear Regression works and how it can be implemented.
4.	Model Evaluation: Evaluate the quality of the Model using RMSE (Root Mean Squared Error).
5.	Feature Engineering: Create new Features that can provide useful information to the Model.
6.	Regularization: Deal with numerical stability problems and improve the Model using Regularization.
7.	Use the Model: Apply the trained Model to new Data to generate Predictions.

## Data Preparation
Data Preparation is the process of preparing the Dataset before using it for Machine Learning. After getting the Data, we load it into Pandas, inspect the Dataset, check the Data Types, and make the Column Names and String Values consistent. This helps make the Data cleaner and easier to work with in the next steps of the Machine Learning Project. The Data Preparation process can be summarized as:

* Get the Data: Obtain the Dataset that will be used for the Project.
* Load the Data: Load the Dataset into a Pandas DataFrame.
* Inspect the Data: Look at the Dataset and understand its basic structure.
* Clean Column Names: Make Column Names consistent, such as converting them to lowercase and replacing spaces with underscores.
* Check Data Types:  Identify which Columns contain numerical or string values.
* Normalize String Values: Make string values consistent by converting them to lowercase and replacing spaces with underscores.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is the process of exploring and understanding the Dataset before training a Model. After preparing the Data, we examine the values in each Column, understand their characteristics and distributions, identify unusual patterns, and check for Missing Values. This helps us understand the Data and identify issues that may need to be handled before using it for

* Explore Columns: Look at the values and characteristics of each Column.
* Check Unique Values: Identify the unique values and the number of unique values in each Column.
* Analyze Distribution: Use visualizations such as Histograms to understand how the values are distributed.
* Check Long-tail Distribution: Identify whether the Target contains a small number of extremely large values.
* Apply Log Transformation: Transform a Long-tail Distribution when needed to reduce the effect of very large values.
* Check Missing Values: Identify Columns containing Missing Values that may need to be handled before Training.

# Setting Up the Validation Framework

After cleaning and exploring the Dataset, the next step is to set up the **Validation Framework**. This framework allows us to train the Model, compare different Model configurations, and evaluate the Final Model on unseen Data.

In general, the Dataset is split into three parts:

1. **Training Set:** Used to train the Model.
2. **Validation Set:** Used to evaluate and compare different Model configurations.
3. **Test Set:** Used for the Final Evaluation after the Model has been selected.

A common Data Split is:

```math
60\% \text{ Training}
+
20\% \text{ Validation}
+
20\% \text{ Test}
=
100\% \text{ of the Dataset}
```

For each Dataset partition, we prepare a **Feature Matrix** $X$ and a **Target Vector** $y$:

```math
\text{Training Set}
\rightarrow
(X_{\text{train}},y_{\text{train}})
```

```math
\text{Validation Set}
\rightarrow
(X_{\text{val}},y_{\text{val}})
```

```math
\text{Test Set}
\rightarrow
(X_{\text{test}},y_{\text{test}})
```

The Feature Matrix $X$ contains the input variables used by the Model, while the Target Vector $y$ contains the values that the Model needs to predict.

The Validation Framework is created through the following process:

1. Calculate the number of observations required for each Dataset partition.
2. Shuffle the records to prevent the original order of the Dataset from influencing the Split.
3. Divide the shuffled indices into Training, Validation, and Test groups.
4. Use the indices to create the three Dataset partitions.
5. Separate the Features $X$ from the Target $y$ in each partition.

The complete process can be summarized as:

```math
\text{Dataset}
\rightarrow
\text{Calculate Split Sizes}
\rightarrow
\text{Shuffle Records}
\rightarrow
\text{Split the Data}
\rightarrow
(X_{\text{train}},y_{\text{train}}),
(X_{\text{val}},y_{\text{val}}),
(X_{\text{test}},y_{\text{test}})
```

This framework ensures that the Model is trained, validated, and tested using separate groups of observations.

## Taking a Part of the DataFrame

After calculating the sizes of the Training, Validation, and Test Sets, the next step is to select the corresponding Rows from the DataFrame.

Suppose the Dataset contains $n$ observations:

```math
n=\text{Total Number of Rows}
```

For a $60\%/20\%/20\%$ Split, the size of each partition is:

```math
n_{\text{val}}=\left\lfloor0.20n\right\rfloor
```

```math
n_{\text{test}}=\left\lfloor0.20n\right\rfloor
```

```math
n_{\text{train}}
=
n-n_{\text{val}}-n_{\text{test}}
```

The three parts of the Dataset can be represented by the following Row positions:

```math
\text{Training Rows}
=
[0,n_{\text{train}})
```

```math
\text{Validation Rows}
=
[n_{\text{train}},n_{\text{train}}+n_{\text{val}})
```

```math
\text{Test Rows}
=
[n_{\text{train}}+n_{\text{val}},n)
```


This allows the Dataset to be divided according to the calculated partition sizes.

However, splitting the Dataset sequentially can create a problem if the observations already follow a particular order.

For example, the Dataset may be arranged by:

- Car manufacturer
- Production year
- Price
- Vehicle type
- Another important characteristic

If the Data is split without changing this order, some groups may appear mainly in one partition and may not be properly represented in the others.

The sequential Split can be represented as:

```math
\text{Ordered Dataset}
\rightarrow
\begin{cases}
\text{First }60\% &\rightarrow \text{Training Set}\\
\text{Next }20\% &\rightarrow \text{Validation Set}\\
\text{Last }20\% &\rightarrow \text{Test Set}
\end{cases}
```

This may result in different Data distributions across the three partitions:

```math
P_{\text{train}}(X)
\neq
P_{\text{val}}(X)
\neq
P_{\text{test}}(X)
```

where $P(X)$ represents the distribution of the Features.

Therefore, before selecting the Rows, we should shuffle the observations so that each partition contains a mixture of records from different parts of the original Dataset:

```math
\text{Ordered Dataset}
\rightarrow
\text{Shuffle Indices}
\rightarrow
\text{Training, Validation, and Test Sets}
```

Shuffling helps make the three Dataset partitions more representative of the overall Data.

## Shuffling the Records

To avoid splitting an ordered Dataset sequentially, we shuffle the Records before dividing them into Training, Validation, and Test Sets.

Suppose the Dataset contains $n$ observations. First, we create a sequence of Row Indices:

```math
I=
\begin{bmatrix}
0&1&2&\cdots&n-1
\end{bmatrix}
```

These indices represent the original Row positions in the DataFrame.

Next, we randomly shuffle their order:

```math
I_{\text{original}}
\rightarrow
I_{\text{shuffled}}
```

For example:

```math
\begin{bmatrix}
0&1&2&3&4&5
\end{bmatrix}
\rightarrow
\begin{bmatrix}
3&1&5&0&4&2
\end{bmatrix}
```

The shuffled indices are then used to select Records from the DataFrame. This changes the order of the observations without changing their values:

```math
df_{\text{shuffled}}
=
df[I_{\text{shuffled}}]
```

After shuffling, the indices are divided according to the previously calculated sizes:

```math
I_{\text{train}}
=
I_{\text{shuffled}}
[0:n_{\text{train}}]
```

```math
I_{\text{val}}
=
I_{\text{shuffled}}
[n_{\text{train}}:n_{\text{train}}+n_{\text{val}}]
```

```math
I_{\text{test}}
=
I_{\text{shuffled}}
[n_{\text{train}}+n_{\text{val}}:n]
```

The three Dataset partitions are then created using their corresponding indices:

```math
I_{\text{train}}
\rightarrow
df_{\text{train}}
```

```math
I_{\text{val}}
\rightarrow
df_{\text{val}}
```

```math
I_{\text{test}}
\rightarrow
df_{\text{test}}
```

The first group of shuffled indices is used for Training, the second group for Validation, and the remaining group for Testing.

### Random Seed and Reproducibility

Random shuffling can produce a different Row order every time it is performed:

```math
I_{\text{shuffled}}^{(1)}
\neq
I_{\text{shuffled}}^{(2)}
```

Different shuffled indices produce different Training, Validation, and Test Sets. This may also lead to different Model results.

To make the random process reproducible, we set a **Random Seed** before shuffling:

```math
\text{Same Random Seed}
\rightarrow
\text{Same Shuffled Indices}
\rightarrow
\text{Same Data Split}
```

For example, if the Random Seed is fixed at $2$:

```math
\text{Seed}=2
```

the same shuffled order can be generated again whenever the process is repeated.

The complete process is:

```math
\text{Create Indices}
\rightarrow
\text{Set Random Seed}
\rightarrow
\text{Shuffle Indices}
\rightarrow
\text{Split Indices}
\rightarrow
\begin{cases}
df_{\text{train}}\\
df_{\text{val}}\\
df_{\text{test}}
\end{cases}
```

Shuffling helps distribute the observations across the three partitions, while the Random Seed makes the Split reproducible.

## Resetting the Index

After shuffling and splitting the Dataset, the Training, Validation, and Test Sets still retain their original Row Indices.

For example, the Training Set may have Indices in a random order:

```math
I_{\text{train}}
=
\begin{bmatrix}
731&245&1089&56&\cdots
\end{bmatrix}
```

These Indices refer to the Row positions in the original Dataset. However, they are no longer needed after the new Dataset partitions have been created.

We therefore reset the Index of each Dataset:

```math
I_{\text{random}}
\rightarrow
I_{\text{reset}}
```

The new Index becomes sequential and starts from $0$:

```math
I_{\text{reset}}
=
\begin{bmatrix}
0&1&2&3&\cdots&m-1
\end{bmatrix}
```

where $m$ is the number of observations in that Dataset partition.

The Index is reset separately for all three Datasets:

```math
df_{\text{train}}
\rightarrow
\text{Reset Index}
\rightarrow
df_{\text{train}}^{*}
```

```math
df_{\text{val}}
\rightarrow
\text{Reset Index}
\rightarrow
df_{\text{val}}^{*}
```

```math
df_{\text{test}}
\rightarrow
\text{Reset Index}
\rightarrow
df_{\text{test}}^{*}
```

When resetting the Index, the old Index should be removed rather than added as a new Column.

Conceptually:

```math
\text{Reset Index}
=
\text{Create New Sequential Index}
+
\text{Remove Old Index}
```

Resetting the Index does not change the observations or their shuffled order. It changes only their Row labels:

```math
\text{Data Before Reset}
=
\text{Data After Reset}
```

but:

```math
\text{Old Index}
\neq
\text{New Index}
```

The complete process is:

```math
\text{Shuffle Records}
\rightarrow
\text{Split Dataset}
\rightarrow
\text{Reset Indices}
\rightarrow
\begin{cases}
df_{\text{train}}\\
df_{\text{val}}\\
df_{\text{test}}
\end{cases}
```

This makes each Dataset partition cleaner and easier to use in the next steps of the Machine Learning process.

## Preparing the Target and Removing It from the Features

After splitting the Dataset into Training, Validation, and Test Sets, we separate the **Target Vector** $y$ from the Features used to create the **Feature Matrix** $X$.

For a Supervised Machine Learning Dataset:

```math
D=(X,y)
```

where:

- $X$ contains the Features used as input.
- $y$ contains the Target values that the Model needs to predict.

The three Dataset partitions are separated into:

```math
D_{\text{train}}
=
(X_{\text{train}},y_{\text{train}})
```

```math
D_{\text{val}}
=
(X_{\text{val}},y_{\text{val}})
```

```math
D_{\text{test}}
=
(X_{\text{test}},y_{\text{test}})
```

### Applying a Log Transformation

The Target variable `price` has a **Long-tail Distribution**, meaning that most cars have moderate prices while a small number of cars have extremely high prices.

To reduce the influence of these very large values, we apply a Log Transformation:

```math
y=\log(1+\text{price})
```

The addition of $1$ allows the transformation to handle a price of $0$:

```math
\log(1+0)=0
```

The transformed Targets are prepared separately for each Dataset:

```math
y_{\text{train}}
=
\log(1+\text{price}_{\text{train}})
```

```math
y_{\text{val}}
=
\log(1+\text{price}_{\text{val}})
```

```math
y_{\text{test}}
=
\log(1+\text{price}_{\text{test}})
```

The transformation compresses large Target values:

```math
\text{Long-tail Target}
\rightarrow
\log(1+\text{Target})
\rightarrow
\text{Less Skewed Target}
```

The Model will therefore learn to predict the transformed Target rather than the original price:

```math
X\rightarrow\widehat{y}_{\log}
```

When the Prediction needs to be interpreted as the original car price, the transformation is reversed:

```math
\widehat{\text{price}}
=
\exp(\widehat{y}_{\log})-1
```

### Removing the Target from the Features

After creating the Target Vectors, the `price` Column must be removed from the Training, Validation, and Test DataFrames.

This ensures that the Feature Matrices contain only input variables:

```math
X=
D-\{\text{Target}\}
```

Therefore:

```math
X_{\text{train}}
=
D_{\text{train}}-\{\text{price}\}
```

```math
X_{\text{val}}
=
D_{\text{val}}-\{\text{price}\}
```

```math
X_{\text{test}}
=
D_{\text{test}}-\{\text{price}\}
```

If the Target is accidentally included as a Feature, the Model receives the value that it is supposed to predict:

```math
X=
\begin{bmatrix}
\text{Features}&\text{Target}
\end{bmatrix}
```

This creates **Data Leakage**:

```math
\text{Target Included in }X
\rightarrow
\text{Data Leakage}
\rightarrow
\text{Unrealistically Good Performance}
```

The Model would appear highly accurate, but the evaluation would not represent its ability to predict prices from genuinely unseen Features.

The correct separation is:

```math
\boxed{
\text{Dataset}
=
\text{Features }X
+
\text{Target }y
}
```

The completed Validation Framework is:

```math
\boxed{
(X_{\text{train}},y_{\text{train}}),
\quad
(X_{\text{val}},y_{\text{val}}),
\quad
(X_{\text{test}},y_{\text{test}})
}
```

At this point, the Training, Validation, and Test Sets are ready for the next stage: **Training the Linear Regression Model**.

# Linear Regression

Linear Regression is a Machine Learning Model used for solving **Regression Problems**, where the Target is a numerical value.

The overall objective is to use the Features $X$ to generate Predictions that are as close as possible to the Actual Target $y$:

```math
g(X)\approx y
```

The Linear Regression process can be understood through five connected parts:

```math
\text{One Prediction}
\rightarrow
\text{Vector and Matrix Form}
\rightarrow
\text{Training}
\rightarrow
\text{Trained Model}
\rightarrow
\text{Evaluation}
```

These five parts explain:

1. How the Model produces a Prediction for one observation.
2. How the same calculation is applied to multiple observations.
3. How the Model learns its Weights from the Training Data.
4. How the learned Weights become a Trained Model.
5. How the quality of the Predictions is evaluated.

---

## 1. Simple Linear Regression: How One Prediction Is Created

The first step is to understand how Linear Regression produces a Prediction for one observation.

Instead of using the entire Feature Matrix $X$, we begin with one observation:

```math
x_i
```

If the observation contains $n$ Features, its Feature Vector is:

```math
x_i=
\begin{bmatrix}
x_{i1}\\
x_{i2}\\
\vdots\\
x_{in}
\end{bmatrix}
```

where:

- $i$ identifies the observation.
- $j$ identifies the Feature.
- $x_{ij}$ is the value of Feature $j$ for observation $i$.

Our objective for this observation is:

```math
g(x_i)\approx y_i
```

where $y_i$ is the Actual Target for observation $i$.

Linear Regression does not simply add the Features together. Each Feature is associated with a Weight:

```math
\begin{aligned}
x_{i1}&\rightarrow w_1\\
x_{i2}&\rightarrow w_2\\
&\ \vdots\\
x_{in}&\rightarrow w_n
\end{aligned}
```

Each Feature is multiplied by its corresponding Weight:

```math
x_{ij}w_j
```

This value represents the **Contribution** of Feature $j$ to the Prediction.

The Model combines all Feature Contributions and adds the Bias $w_0$:

```math
g(x_i)
=
w_0+\sum_{j=1}^{n}x_{ij}w_j
```

Expanded:

```math
g(x_i)
=
w_0
+x_{i1}w_1
+x_{i2}w_2
+\cdots
+x_{in}w_n
```

Therefore:

```math
\text{Prediction}
=
\text{Bias}
+
\text{Feature Contributions}
```

More specifically:

```math
\widehat{y}_i
=
w_0
+x_{i1}w_1
+x_{i2}w_2
+\cdots
+x_{in}w_n
```

where:

- $\widehat{y}_i$ is the Predicted Target.
- $w_0$ is the Bias.
- $w_1,w_2,\ldots,w_n$ are the Feature Weights.
- $x_{i1},x_{i2},\ldots,x_{in}$ are the Feature values.

At this stage, we assume that the Weights are already available:

```math
w_0,w_1,w_2,\ldots,w_n
```

We are not yet asking how the Model learned them. We are asking how Linear Regression uses the given Weights and Features to produce a Prediction.

The first part can therefore be summarized as:

```math
x_i,w
\rightarrow
g(x_i)
\rightarrow
\widehat{y}_i
```

This is the basic calculation on which the remaining parts are built.

---

## 2. Vector Form: From One Prediction to Multiple Predictions

Writing each Feature and Weight separately becomes inconvenient when the Dataset contains many Features and observations.

The same Linear Regression calculation can therefore be represented more compactly using Linear Algebra.

We begin with:

```math
g(x_i)
=
w_0+\sum_{j=1}^{n}x_{ij}w_j
```

The summation:

```math
x_{i1}w_1+x_{i2}w_2+\cdots+x_{in}w_n
```

is the Dot Product between the Feature Vector and the Weight Vector:

```math
x_i^Tw
```

Therefore:

```math
g(x_i)=w_0+x_i^Tw
```

### Including the Bias in the Vector

We can include the Bias in the Dot Product by introducing a Fictional Feature:

```math
x_{i0}=1
```

The extended Feature Vector becomes:

```math
x_i=
\begin{bmatrix}
1\\
x_{i1}\\
x_{i2}\\
\vdots\\
x_{in}
\end{bmatrix}
```

The complete Weight Vector becomes:

```math
w=
\begin{bmatrix}
w_0\\
w_1\\
w_2\\
\vdots\\
w_n
\end{bmatrix}
```

Because:

```math
1\times w_0=w_0
```

the Bias becomes part of the same Dot Product:

```math
g(x_i)=x_i^Tw
```

Therefore:

```math
\widehat{y}_i=x_i^Tw
```

This is still the same Linear Regression Model. Only the mathematical representation has changed.

### From One Observation to the Entire Dataset

A Dataset with $m$ observations and $n$ Features can be represented as a Feature Matrix:

```math
X=
\begin{bmatrix}
1&x_{11}&x_{12}&\cdots&x_{1n}\\
1&x_{21}&x_{22}&\cdots&x_{2n}\\
1&x_{31}&x_{32}&\cdots&x_{3n}\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&x_{m1}&x_{m2}&\cdots&x_{mn}
\end{bmatrix}
```

where:

```math
\text{Rows}=\text{Observations}
```

and:

```math
\text{Columns}=\text{Features}
```

The first Column contains ones and represents the Bias.

The dimensions of the Feature Matrix are:

```math
X\in\mathbb{R}^{m\times(n+1)}
```

The same Weight Vector is applied to every Row:

```math
w=
\begin{bmatrix}
w_0\\
w_1\\
w_2\\
\vdots\\
w_n
\end{bmatrix}
\in\mathbb{R}^{(n+1)\times1}
```

The Matrix–Vector multiplication is valid because:

```math
\underbrace{X}_{m\times(n+1)}
\underbrace{w}_{(n+1)\times1}
=
\underbrace{\widehat{y}}_{m\times1}
```

We can now generate Predictions for all observations simultaneously:

```math
g(X)=Xw
```

This produces the Prediction Vector:

```math
\widehat{y}
=
\begin{bmatrix}
\widehat{y}_1\\
\widehat{y}_2\\
\vdots\\
\widehat{y}_m
\end{bmatrix}
```

Therefore:

```math
\boxed{\widehat{y}=Xw}
```

The calculation for one observation:

```math
\text{One Observation}
\rightarrow
g(x_i)=x_i^Tw
```

becomes:

```math
\text{Multiple Observations}
\rightarrow
g(X)=Xw
```

### Different Representations of the Same Calculation

The Model itself has not changed.

The Simple Form is:

```math
w_0+\sum_{j=1}^{n}x_{ij}w_j
```

The Vector Form is:

```math
x_i^Tw
```

The Matrix Form is:

```math
Xw
```

These are different representations of the same Linear Regression calculation:

```math
w_0+\sum_{j=1}^{n}x_{ij}w_j
\equiv
x_i^Tw
\equiv
Xw
```

The appropriate representation depends on whether we are calculating a Prediction for one observation or multiple observations.

---

## 3. Training Linear Regression: Where Do the Weights Come From?

So far, we have assumed that the Weight Vector $w$ already exists.

We know that:

```math
\widehat{y}=Xw
```

However, in a Machine Learning Project, we do not normally assign the Weights manually. The Model must learn them from the Training Data.

This changes the direction of the problem.

During Prediction:

```math
\boxed{X,w\rightarrow\widehat{y}}
```

During Training:

```math
\boxed{X,y\rightarrow w}
```

We know the Features:

```math
X
```

and the Actual Target:

```math
y
```

We want to find the Weight Vector:

```math
w
```

that makes the Predictions:

```math
Xw
```

as close as possible to the Actual Target:

```math
y
```

Therefore, the Training objective is:

```math
\boxed{Xw\approx y}
```

### Why We Need the Normal Equation

If the relationship were exactly:

```math
Xw=y
```

we might initially try to isolate $w$ using:

```math
w=X^{-1}y
```

However, the Feature Matrix usually contains many observations and fewer Features:

```math
X\in\mathbb{R}^{m\times(n+1)}
```

where:

- $m$ is the Number of Observations.
- $n$ is the Number of Features.
- The additional Column represents the Bias.

Usually:

```math
m>n+1
```

Therefore, $X$ is generally a Rectangular Matrix rather than a Square Matrix:

```math
m\times(n+1)
```

An ordinary Inverse $X^{-1}$ exists only for an invertible Square Matrix. Therefore:

```math
X^{-1}
```

does not normally exist.

To solve this problem, we multiply both sides by the Transpose of $X$:

```math
X^TXw=X^Ty
```

The Matrix:

```math
X^TX
```

has the dimensions:

```math
\underbrace{X^T}_{(n+1)\times m}
\underbrace{X}_{m\times(n+1)}
=
\underbrace{X^TX}_{(n+1)\times(n+1)}
```

Therefore, $X^TX$ is a Square Matrix.

If $X^TX$ is invertible, the Weight Vector can be calculated as:

```math
\boxed{
w=(X^TX)^{-1}X^Ty
}
```

This is the **Normal Equation**.

The Training process can be summarized as:

```math
\text{Training Data }(X,y)
\rightarrow
\text{Normal Equation}
\rightarrow
\text{Learned Weights }w
```

The Weight Vector that was assumed to exist during Prediction now has an origin:

```math
X_{\text{train}},y_{\text{train}}
\rightarrow
w
```

---

## 4. From Weights to the Trained Model

After Training, we obtain the learned Weight Vector:

```math
w=
\begin{bmatrix}
w_0\\
w_1\\
w_2\\
\vdots\\
w_n
\end{bmatrix}
```

These values are the learned Parameters of the Linear Regression Model.

For one observation:

```math
\widehat{y}_i=x_i^Tw
```

For all observations:

```math
\widehat{y}=Xw
```

The relationship between Training and Prediction is:

```math
X_{\text{train}},y_{\text{train}}
\rightarrow
\text{Training}
\rightarrow
w
\rightarrow
\text{Trained Linear Regression Model}
```

The Trained Model can then be applied to a Feature Matrix:

```math
X
\rightarrow
\text{Model with learned }w
\rightarrow
\widehat{y}
```

This distinction is fundamental:

- **Training learns $w$.**
- **Prediction uses $w$.**

Therefore:

```math
\boxed{
\text{Training: }X_{\text{train}},y_{\text{train}}\rightarrow w
}
```

and:

```math
\boxed{
\text{Prediction: }X,w\rightarrow\widehat{y}
}
```

---

## 5. RMSE: How Good Are the Predictions?

After Training the Model and generating Predictions, we have two vectors.

The Actual Target Vector is:

```math
y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_m
\end{bmatrix}
```

The Prediction Vector is:

```math
\widehat{y}
=
\begin{bmatrix}
\widehat{y}_1\\
\widehat{y}_2\\
\vdots\\
\widehat{y}_m
\end{bmatrix}
```

The next question is:

> How different are the Predictions from the Actual Values?

This is where Model Evaluation begins.

### Calculating the Errors

For each observation, the Prediction Error is:

```math
e_i=\widehat{y}_i-y_i
```

For all observations, the Error Vector is:

```math
e=
\begin{bmatrix}
\widehat{y}_1-y_1\\
\widehat{y}_2-y_2\\
\vdots\\
\widehat{y}_m-y_m
\end{bmatrix}
```

Simply averaging the Errors is problematic because positive and negative values can cancel each other:

```math
(+e)+(-e)=0
```

A Mean Error close to zero would not necessarily mean that the Predictions are accurate.

### Squaring the Errors

To prevent positive and negative Errors from cancelling, each Error is squared:

```math
e_i^2
=
(\widehat{y}_i-y_i)^2
```

The Mean Squared Error is:

```math
MSE
=
\frac{1}{m}
\sum_{i=1}^{m}
(\widehat{y}_i-y_i)^2
```

Finally, we take the Square Root:

```math
\boxed{
RMSE
=
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
(\widehat{y}_i-y_i)^2
}
}
```

RMSE combines all Prediction Errors into one value that represents the overall Prediction Error according to this metric.

The process is:

```math
y,\widehat{y}
\rightarrow
\text{Errors}
\rightarrow
\text{Squared Errors}
\rightarrow
\text{Mean}
\rightarrow
\text{Square Root}
\rightarrow
RMSE
```

For the same Dataset and Target representation, a lower RMSE means that the Predictions are closer to the Actual Values:

```math
RMSE\downarrow
\quad\Rightarrow\quad
\text{Prediction Error}\downarrow
```

---

## Complete Linear Regression Process

The complete Linear Regression process is:

```math
\underbrace{X,y}_{\text{Training Data}}
\rightarrow
\underbrace{w}_{\text{Learned Weights}}
\rightarrow
\underbrace{g(X)=Xw}_{\text{Linear Regression Model}}
\rightarrow
\underbrace{\widehat{y}}_{\text{Predictions}}
\rightarrow
\underbrace{RMSE(y,\widehat{y})}_{\text{Evaluation}}
```

The complete mathematical relationship is:

```math
X,y
\rightarrow
w=(X^TX)^{-1}X^Ty
\rightarrow
\widehat{y}=Xw
\rightarrow
RMSE
=
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
(\widehat{y}_i-y_i)^2
}
```

Therefore, the entire process can be summarized as:

```math
\boxed{
\text{Training Data}
\rightarrow
\text{Learn Weights}
\rightarrow
\text{Make Predictions}
\rightarrow
\text{Evaluate Predictions}
}
```

# Feature Engineering

After training and evaluating the first Linear Regression Model, we can use RMSE to determine how well the Model predicts the Target.

If the performance is not satisfactory, one way to improve the Model is to improve the information contained in the Feature Matrix $X$. This process is called **Feature Engineering**.

Feature Engineering is the process of creating or transforming Features so that they represent the underlying problem more effectively.

The general process is:

```math
\text{Existing Features}
\rightarrow
\text{Feature Engineering}
\rightarrow
\text{Improved Feature Matrix}
\rightarrow
\text{Model Training}
```

---

## Why Feature Engineering Is Important

A Machine Learning Model can learn only from the information provided through its Features.

The relationship between the Features and Predictions is:

```math
X
\rightarrow
\text{Model}
\rightarrow
\widehat{y}
```

If the Feature Matrix does not represent an important relationship clearly, the Model may not be able to learn that relationship effectively.

Feature Engineering changes the representation of the input:

```math
X_{\text{original}}
\rightarrow
X_{\text{engineered}}
```

The Model itself does not necessarily change. Instead, it receives Features that describe the problem more meaningfully.

Therefore, Model performance depends on both:

```math
\text{Model Performance}
=
f(
\text{Model},
\text{Feature Representation}
)
```

In other words:

```math
\text{Model}
+
\text{Feature Representation}
\rightarrow
\text{Predictions}
```

---

## Creating a New Feature

The car price Dataset contains a Feature called `year`, which represents the year in which each car was produced.

The year is related to the price because newer and older cars generally have different values. However, the production year does not directly express how old the car was when the Data was collected.

Because the Dataset was collected in $2017$, a new Feature called `age` can be created:

```math
\text{age}
=
2017-\text{year}
```

This transformation changes the representation from the production year to the age of the car:

```math
\text{year}
\rightarrow
\text{age}
```

The new Feature does not come from an external source. It is derived from information already available in the Dataset.

This illustrates the main idea of Feature Engineering:

```math
\text{Existing Information}
\rightarrow
\text{More Meaningful Representation}
```

---

## Updating the Feature Matrix

Before Feature Engineering, the Model uses a Feature Matrix containing the original Features:

```math
X_{\text{base}}
=
\begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1n}\\
x_{21}&x_{22}&\cdots&x_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
x_{m1}&x_{m2}&\cdots&x_{mn}
\end{bmatrix}
```

where:

```math
m=\text{Number of Observations}
```

and:

```math
n=\text{Number of Features}
```

The dimensions of the Base Feature Matrix are:

```math
X_{\text{base}}
\in
\mathbb{R}^{m\times n}
```

After creating `age`, it is added as another Column:

```math
X_{\text{engineered}}
=
\begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1n}&age_1\\
x_{21}&x_{22}&\cdots&x_{2n}&age_2\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
x_{m1}&x_{m2}&\cdots&x_{mn}&age_m
\end{bmatrix}
```

The number of observations remains the same, but the number of Features increases:

```math
X_{\text{base}}
\in
\mathbb{R}^{m\times n}
```

```math
X_{\text{engineered}}
\in
\mathbb{R}^{m\times(n+1)}
```

Therefore:

```math
m\times n
\rightarrow
m\times(n+1)
```

Feature Engineering changes both the structure and the information contained in $X$.

---

## How the New Feature Enters the Model

Before adding the new Feature, Linear Regression produces a Prediction using the Base Features:

```math
\widehat{y}_i
=
w_0+\sum_{j=1}^{n}x_{ij}w_j
```

After adding `age`, the Model includes an additional Feature Contribution:

```math
\widehat{y}_i
=
w_0
+
\sum_{j=1}^{n}x_{ij}w_j
+
age_iw_{\text{age}}
```

The new Feature receives its own Weight:

```math
age_i
\rightarrow
w_{\text{age}}
```

Its contribution to the Prediction is:

```math
age_iw_{\text{age}}
```

Feature Engineering determines what information enters the Model, while Model Training determines how much Weight that information receives.

Therefore:

```math
\text{Feature Engineering}
\rightarrow
\text{Creates the Inputs}
```

```math
\text{Training}
\rightarrow
\text{Learns the Weights}
```

---

## Retraining the Model

After changing the Feature Matrix, the Model must be trained again.

The original Model was trained using:

```math
X_{\text{base}},y
\rightarrow
\text{Training}
\rightarrow
w_{\text{base}}
```

After Feature Engineering:

```math
X_{\text{engineered}},y
\rightarrow
\text{Training}
\rightarrow
w_{\text{engineered}}
```

The original Weights cannot simply be reused because the new Feature Matrix contains an additional Column.

The new Weight Vector must include a Weight for `age`:

```math
w_{\text{engineered}}
=
\begin{bmatrix}
w_0\\
w_1\\
w_2\\
\vdots\\
w_n\\
w_{\text{age}}
\end{bmatrix}
```

The process becomes:

```math
X_{\text{engineered}},y
\rightarrow
w_{\text{engineered}}
\rightarrow
\widehat{y}
```

---

## Consistency Across the Datasets

The same Feature Engineering process must be applied to the Training, Validation, and Test Sets.

```math
df_{\text{train}}
\rightarrow
\text{Feature Engineering}
\rightarrow
X_{\text{train}}
```

```math
df_{\text{val}}
\rightarrow
\text{Feature Engineering}
\rightarrow
X_{\text{val}}
```

```math
df_{\text{test}}
\rightarrow
\text{Feature Engineering}
\rightarrow
X_{\text{test}}
```

All Feature Matrices must contain the same Features in the same order:

```math
\text{Columns of }X_{\text{train}}
=
\text{Columns of }X_{\text{val}}
=
\text{Columns of }X_{\text{test}}
```

Their number of Rows may differ, but they must have the same number of Columns:

```math
X_{\text{train}}
\in
\mathbb{R}^{m_{\text{train}}\times p}
```

```math
X_{\text{val}}
\in
\mathbb{R}^{m_{\text{val}}\times p}
```

```math
X_{\text{test}}
\in
\mathbb{R}^{m_{\text{test}}\times p}
```

where $p$ is the total number of Features after Feature Engineering.

If a Feature is created only for the Training Set, the Model will expect an input that is not available during Validation or Testing.

Therefore, Feature Engineering should be treated as a consistent Data Preparation process rather than as a change made only to the Training Data.

---

## Preserving the Original Data

The Feature Preparation process should produce the required Feature Matrix without unexpectedly modifying the original DataFrame.

Conceptually:

```math
df_{\text{original}}
\rightarrow
df_{\text{copy}}
\rightarrow
\text{Feature Engineering}
\rightarrow
X
```

The transformation is applied to a copy of the DataFrame, while the original Dataset remains unchanged.

This separates the original Data from the temporary representation created for Model Training:

```math
\text{Original Data}
\neq
\text{Model Input Representation}
```

Preserving the original Data makes the preparation process safer, repeatable, and easier to apply consistently across different Datasets.

---

## Evaluating the New Feature

After adding `age`, the Model is trained again using the new Feature Matrix:

```math
X_{\text{train}},y_{\text{train}}
\rightarrow
\text{Training}
\rightarrow
w
```

The learned Weights are then applied to the Validation Set:

```math
X_{\text{val}},w
\rightarrow
\widehat{y}_{\text{val}}
```

Finally, the Predictions are compared with the Actual Validation Targets:

```math
y_{\text{val}},
\widehat{y}_{\text{val}}
\rightarrow
RMSE
```

Before adding `age`, the RMSE was approximately:

```math
RMSE_{\text{base}}
\approx
0.76
```

After adding `age`, it decreased to approximately:

```math
RMSE_{\text{age}}
\approx
0.51
```

Therefore:

```math
RMSE_{\text{age}}
<
RMSE_{\text{base}}
```

Because a lower RMSE means that the Predictions are closer to the Actual Values, the result indicates that `age` is a useful Feature for predicting car prices.

---

## Feature Engineering Is an Evaluation Process

Creating a new Feature does not automatically mean that the Model will improve.

Every new Feature must be evaluated using the Validation Set:

```math
\text{Create Feature}
\rightarrow
\text{Retrain Model}
\rightarrow
\text{Generate Validation Predictions}
\rightarrow
\text{Calculate RMSE}
```

The new Feature may be considered useful when it reduces the Validation RMSE:

```math
RMSE_{\text{new}}
<
RMSE_{\text{previous}}
```

If the RMSE remains unchanged or becomes higher, the Feature may not provide useful predictive information, or it may introduce additional problems into the Model.

Therefore:

```math
\text{New Feature}
\nRightarrow
\text{Better Model}
```

The usefulness of a Feature must be supported by Validation results.

---

## Complete Feature Engineering Process

The complete process can be summarized as:

```math
\text{Existing Data}
\rightarrow
\text{Create or Transform Features}
\rightarrow
X_{\text{engineered}}
```

The engineered Feature Matrix is then used in the existing Linear Regression process:

```math
X_{\text{engineered}},y
\rightarrow
\text{Training}
\rightarrow
w_{\text{engineered}}
\rightarrow
\widehat{y}
\rightarrow
RMSE
```

For this Dataset:

```math
\text{year}
\rightarrow
\text{age}
\rightarrow
\text{Retrain Model}
\rightarrow
\text{Lower RMSE}
```

The main concept is that Feature Engineering improves how the Data is represented to the Model. It does not replace Model Training or Model Evaluation.

Instead, it becomes part of the complete Machine Learning process:

```math
\boxed{
\text{Data Preparation}
\rightarrow
\text{Feature Engineering}
\rightarrow
\text{Model Training}
\rightarrow
\text{Prediction}
\rightarrow
\text{Evaluation}
}
```

# Categorical Variables

After creating Numerical Features, the next stage of Feature Engineering is to include **Categorical Variables** in the Model.

A Categorical Variable represents a group, type, or class rather than a measurable numerical quantity. In the car price Dataset, these variables describe characteristics such as the manufacturer, transmission type, fuel type, and vehicle style.

Categorical Variables may contain important information for predicting the Target. However, Linear Regression cannot work directly with category names. They must first be transformed into numerical Features.

The general process is:

```math
\text{Categorical Variables}
\rightarrow
\text{Numerical Encoding}
\rightarrow
\text{Feature Matrix }X
\rightarrow
\text{Model Training}
```

---

## Identifying Categorical Variables

Categorical Variables are commonly stored as strings. Examples in the car price Dataset include:

- `make`
- `model`
- `engine_fuel_type`
- `transmission_type`
- `driven_wheels`
- `market_category`
- `vehicle_size`
- `vehicle_style`

Pandas normally identifies these Columns using the `object` Data Type.

However, the Data Type alone does not determine whether a variable is Numerical or Categorical. We also need to consider what its values represent.

For example, `number_of_doors` contains numerical values, but these values identify different categories of cars rather than a continuous quantity. It can therefore be treated as a Categorical Variable.

This distinction can be expressed as:

```math
\text{Data Type}
\neq
\text{Meaning of the Variable}
```

Identifying Categorical Variables therefore requires an understanding of both the Dataset structure and the meaning of each Feature.

---

## Why Categorical Variables Must Be Encoded

Linear Regression produces Predictions by multiplying each Feature by its corresponding Weight:

```math
\widehat{y}_i
=
w_0+\sum_{j=1}^{n}x_{ij}w_j
```

Every Feature must therefore have a numerical representation.

Category names cannot be used directly in this calculation:

```math
\text{Category Name}
\times
\text{Weight}
```

Assigning arbitrary numbers to categories is also inappropriate because it may create an order or distance that does not actually exist.

For example:

```math
\text{Category A}=1,\qquad
\text{Category B}=2,\qquad
\text{Category C}=3
```

This representation would suggest an ordered relationship:

```math
\text{Category A}
<
\text{Category B}
<
\text{Category C}
```

It would also imply that a measurable distance exists between the categories.

For categories such as manufacturers or transmission types, these numerical relationships have no meaningful interpretation. A different representation is therefore required.

---

## One-Hot Encoding

**One-Hot Encoding** transforms each category into a separate Binary Feature.

Each Binary Feature contains:

```math
1
=
\text{The observation belongs to the category}
```

```math
0
=
\text{The observation does not belong to the category}
```

If a Categorical Variable contains $k$ selected categories, it can be transformed into $k$ Binary Features:

```math
x_{\text{categorical}}
\rightarrow
\begin{bmatrix}
d_1&d_2&\cdots&d_k
\end{bmatrix}
```

For observation $i$, Binary Feature $d_{il}$ is defined as:

```math
d_{il}
=
\begin{cases}
1,&\text{if observation }i\text{ belongs to category }l\\
0,&\text{otherwise}
\end{cases}
```

The transformation changes one Categorical Column into multiple numerical Columns:

```math
\text{One Categorical Feature}
\rightarrow
\text{Multiple Binary Features}
```

These Binary Features can then be included in the Feature Matrix and used by Linear Regression.

---

## Adding Binary Features to the Feature Matrix

Before adding Categorical Variables, the Feature Matrix contains the Numerical and previously engineered Features:

```math
X_{\text{numerical}}
\in
\mathbb{R}^{m\times n}
```

where:

- $m$ is the Number of Observations.
- $n$ is the Number of existing Features.

After One-Hot Encoding, the Binary Features are represented by:

```math
D=
\begin{bmatrix}
d_{11}&d_{12}&\cdots&d_{1k}\\
d_{21}&d_{22}&\cdots&d_{2k}\\
\vdots&\vdots&\ddots&\vdots\\
d_{m1}&d_{m2}&\cdots&d_{mk}
\end{bmatrix}
```

where:

```math
D\in\mathbb{R}^{m\times k}
```

The Binary Features are added as new Columns:

```math
X_{\text{categorical}}
=
\begin{bmatrix}
X_{\text{numerical}}&D
\end{bmatrix}
```

The complete Feature Matrix has the dimensions:

```math
X_{\text{categorical}}
\in
\mathbb{R}^{m\times(n+k)}
```

Therefore:

```math
m\times n
\rightarrow
m\times(n+k)
```

The number of observations remains unchanged, but the number of Features increases.

---

## How Categories Enter Linear Regression

Each encoded category receives its own Weight:

```math
\begin{aligned}
d_1&\rightarrow w_{d_1}\\
d_2&\rightarrow w_{d_2}\\
&\ \vdots\\
d_k&\rightarrow w_{d_k}
\end{aligned}
```

The Prediction becomes:

```math
\widehat{y}_i
=
w_0
+
\sum_{j=1}^{n}x_{ij}w_j
+
\sum_{l=1}^{k}d_{il}w_{d_l}
```

The first summation represents the contributions of the Numerical Features:

```math
\sum_{j=1}^{n}x_{ij}w_j
```

The second summation represents the contributions of the encoded Categorical Features:

```math
\sum_{l=1}^{k}d_{il}w_{d_l}
```

The Model can therefore learn how each represented category contributes to the Prediction.

---

## Selecting Categories

Some Categorical Variables contain many distinct values. Encoding every value would create a large number of Binary Features.

If a variable contains $k$ unique categories:

```math
1\text{ Categorical Column}
\rightarrow
k\text{ Binary Columns}
```

A Categorical Variable containing many distinct values is described as having **High Cardinality**.

High Cardinality can cause the Feature Matrix to become unnecessarily large:

```math
\text{Many Categories}
\rightarrow
\text{Many Binary Features}
\rightarrow
\text{Large Feature Matrix}
```

For this reason, the lesson selects the most frequent categories rather than encoding every available value.

The process is:

```math
\text{Categorical Variable}
\rightarrow
\text{Identify Frequent Categories}
\rightarrow
\text{Create Selected Binary Features}
```

Categories outside the selected group are represented by zeros across the selected Binary Features.

This approach controls the number of Columns while retaining the categories that occur most frequently in the Training Data.

---

## Selecting Categories from the Training Set

The categories must be selected using only the Training Set:

```math
df_{\text{train}}
\rightarrow
\text{Select Categories}
\rightarrow
\text{Category Definitions}
```

The same Category Definitions are then applied to all Datasets:

```math
\text{Category Definitions}
\rightarrow
\begin{cases}
X_{\text{train}}\\
X_{\text{val}}\\
X_{\text{test}}
\end{cases}
```

The Validation and Test Sets should not determine which categories are selected. Otherwise, information outside the Training Set would influence the Feature Engineering process.

This could introduce Data Leakage:

```math
\text{Information from Validation or Test Data}
\rightarrow
\text{Feature Selection}
\rightarrow
\text{Data Leakage}
```

Therefore, the Training Data determines the Feature structure, while the Validation and Test Data follow that same structure.

---

## Consistency Across the Datasets

The same One-Hot Encoding process must be applied to the Training, Validation, and Test Sets.

All three Feature Matrices must contain the same Columns in the same order:

```math
\text{Columns of }X_{\text{train}}
=
\text{Columns of }X_{\text{val}}
=
\text{Columns of }X_{\text{test}}
```

The required dimensions are:

```math
X_{\text{train}}
\in
\mathbb{R}^{m_{\text{train}}\times p}
```

```math
X_{\text{val}}
\in
\mathbb{R}^{m_{\text{val}}\times p}
```

```math
X_{\text{test}}
\in
\mathbb{R}^{m_{\text{test}}\times p}
```

The number of Rows may differ, but all three Matrices must contain the same number of Features:

```math
p_{\text{train}}
=
p_{\text{val}}
=
p_{\text{test}}
```

The meaning and order of the Columns must also remain the same.

---

## Retraining After Categorical Encoding

Adding Binary Features changes the Feature Matrix. The Model must therefore be trained again.

Before adding Categorical Features:

```math
X_{\text{numerical}},y
\rightarrow
\text{Training}
\rightarrow
w_{\text{numerical}}
```

After adding Categorical Features:

```math
X_{\text{categorical}},y
\rightarrow
\text{Training}
\rightarrow
w_{\text{categorical}}
```

The new Weight Vector includes both Numerical and Categorical Weights:

```math
w_{\text{categorical}}
=
\begin{bmatrix}
w_0\\
w_1\\
\vdots\\
w_n\\
w_{d_1}\\
\vdots\\
w_{d_k}
\end{bmatrix}
```

The original Weight Vector cannot be reused because the structure of the Feature Matrix has changed.

---

## Evaluating Categorical Features

Categorical Features must be evaluated using the same Validation Framework as other engineered Features:

```math
\text{Encode Categories}
\rightarrow
\text{Retrain Model}
\rightarrow
\text{Generate Validation Predictions}
\rightarrow
\text{Calculate RMSE}
```

Adding `number_of_doors` produces only a small improvement:

```math
RMSE:
0.5172
\rightarrow
0.5158
```

Adding `make` produces a larger improvement:

```math
RMSE:
0.5158
\rightarrow
0.5077
```

These results show that different Categorical Variables contain different levels of predictive information.

Therefore:

```math
\text{More Features}
\nRightarrow
\text{Better Model}
```

The usefulness of each Feature must be determined through Validation rather than assumed from its presence in the Dataset.

---

## Adding Multiple Categorical Variables

The same One-Hot Encoding process can be applied to multiple Categorical Variables:

```math
C_1,C_2,\ldots,C_q
```

Each Categorical Variable produces a set of Binary Features:

```math
\begin{aligned}
C_1&\rightarrow D_1\\
C_2&\rightarrow D_2\\
&\ \vdots\\
C_q&\rightarrow D_q
\end{aligned}
```

The complete Feature Matrix becomes:

```math
X_{\text{engineered}}
=
\begin{bmatrix}
X_{\text{numerical}}&
D_1&
D_2&
\cdots&
D_q
\end{bmatrix}
```

This allows the Model to use more information from the Dataset. However, it also increases the number of Columns and the possibility that some Features contain duplicated or strongly related information.

---

## When Additional Features Make the Model Unstable

After adding several Categorical Variables, the Model produces extremely large Weights and the RMSE increases significantly:

```math
RMSE\approx41.45
```

Some learned Weights become approximately:

```math
|w_j|\approx10^{15}
```

This indicates that the Linear Regression calculation has become numerically unstable.

The problem can be represented as:

```math
\text{Many Binary Features}
\rightarrow
\text{Related or Redundant Columns}
\rightarrow
\text{Unstable Weights}
\rightarrow
\text{Poor Predictions}
```

The issue occurs during the calculation of the Normal Equation:

```math
w=(X^TX)^{-1}X^Ty
```

When the Feature Matrix contains strongly related Columns, $X^TX$ can become singular or nearly singular:

```math
\text{Related Columns in }X
\rightarrow
\text{Unstable }X^TX
```

The Inverse may then produce extremely large values:

```math
\text{Unstable }(X^TX)^{-1}
\rightarrow
\text{Extremely Large }w
```

The Categorical Variables may still contain useful information. The problem is not necessarily the information itself, but the numerical instability created during Model Training.

---

## Complete Categorical Variable Process

The complete process can be summarized as:

```math
\text{Identify Categorical Variables}
\rightarrow
\text{Select Categories}
\rightarrow
\text{One-Hot Encoding}
\rightarrow
\text{Binary Features}
```

The Binary Features are added to the Feature Matrix:

```math
X_{\text{numerical}}
\rightarrow
X_{\text{engineered}}
```

The Model is then trained and evaluated:

```math
X_{\text{engineered}},y
\rightarrow
\text{Training}
\rightarrow
w
\rightarrow
\widehat{y}
\rightarrow
RMSE
```

Categorical Variables can improve the Model when they provide useful predictive information:

```math
\text{Categorical Variables}
\rightarrow
\text{More Information}
\rightarrow
\text{Potential Improvement}
```

However, adding many related Binary Features can also create numerical instability:

```math
\text{Many Related Features}
\rightarrow
\text{Numerical Instability}
```

This problem leads to the next stage, **Regularization**, which stabilizes the Training calculation and controls the size of the learned Weights.

# Regularization

After adding multiple Categorical Features, the Linear Regression Model produced extremely large Weights and a significantly higher RMSE. This indicates that the Training calculation became numerically unstable.

**Regularization** is a technique used to stabilize the Model by controlling the size of its learned Weights.

The general problem can be summarized as:

```math
\text{Related Features}
\rightarrow
\text{Unstable Matrix}
\rightarrow
\text{Large Weights}
\rightarrow
\text{Unstable Predictions}
```

Regularization modifies the Training process to produce more controlled and stable Weights.

---

## The Problem of Related Features

Linear Regression learns the Weight Vector using the Normal Equation:

```math
w=(X^TX)^{-1}X^Ty
```

This calculation requires the Inverse of:

```math
X^TX
```

When the Feature Matrix contains duplicated or strongly related Features, some Columns provide the same or nearly the same information.

In Linear Algebra, these Columns are described as **linearly dependent** or nearly linearly dependent.

As a result, $X^TX$ can become singular or nearly singular:

```math
\text{Related Columns in }X
\rightarrow
\text{Unstable }X^TX
```

If $X^TX$ is singular, its Inverse does not exist:

```math
(X^TX)^{-1}
\quad\text{does not exist}
```

If it is nearly singular, the Inverse may technically exist but produce extremely large values.

These large values lead to unstable Weights:

```math
\text{Unstable }(X^TX)^{-1}
\rightarrow
\text{Unstable }w
```

This explains why the Model became unstable after adding many One-Hot Encoded Features.

---

## Why the Weights Become Unstable

When two Features contain nearly the same information, the Model cannot clearly determine how much Weight should be assigned to each Feature.

Their combined contribution may be represented as:

```math
x_1w_1+x_2w_2
```

If:

```math
x_1\approx x_2
```

the Model may assign a very large positive Weight to one Feature and a very large negative Weight to the other.

Although these Weights may partly cancel each other, the Model becomes highly sensitive to small changes in the Data.

Therefore:

```math
\text{Related Features}
\rightarrow
\text{Uncertain Feature Contributions}
\rightarrow
\text{Large Weights}
\rightarrow
\text{Unstable Model}
```

---

## Adding Regularization

Regularization modifies the Matrix before calculating its Inverse.

Instead of using:

```math
X^TX
```

we add a small positive value to its diagonal:

```math
X^TX+rI
```

where:

- $r$ is the Regularization Parameter.
- $I$ is the Identity Matrix.

The Identity Matrix contains $1$ on its diagonal and $0$ everywhere else:

```math
I=
\begin{bmatrix}
1&0&\cdots&0\\
0&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&1
\end{bmatrix}
```

Multiplying the Identity Matrix by $r$ produces:

```math
rI=
\begin{bmatrix}
r&0&\cdots&0\\
0&r&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&r
\end{bmatrix}
```

Therefore, adding $rI$ changes only the diagonal values of $X^TX$.

---

## The Regularized Normal Equation

After adding Regularization, the Normal Equation becomes:

```math
\boxed{
w=(X^TX+rI)^{-1}X^Ty
}
```

Without Regularization:

```math
w=(X^TX)^{-1}X^Ty
```

With Regularization:

```math
w=(X^TX+rI)^{-1}X^Ty
```

Adding $rI$ makes the Matrix more stable and easier to invert reliably.

The process becomes:

```math
X^TX
\rightarrow
X^TX+rI
\rightarrow
(X^TX+rI)^{-1}
\rightarrow
w_{\text{regularized}}
```

---

## Controlling the Weights

Regularization helps prevent the Weight values from becoming excessively large:

```math
|w_j|\downarrow
```

The Model still learns the relationship between the Features and Target, but it avoids relying on extreme Weight combinations.

Conceptually, Training now balances two objectives:

```math
\text{Fit the Training Data}
+
\text{Control the Weights}
```

This can be expressed as:

```math
\text{Regularized Loss}
=
\text{Prediction Error}
+
\text{Weight Penalty}
```

For L2 Regularization:

```math
\boxed{
\text{Regularized Loss}
=
\sum_{i=1}^{m}(y_i-\widehat{y}_i)^2
+
r\sum_{j=1}^{n}w_j^2
}
```

The first component measures the Prediction Error:

```math
\sum_{i=1}^{m}(y_i-\widehat{y}_i)^2
```

The second component penalizes large Weights:

```math
r\sum_{j=1}^{n}w_j^2
```

The Model therefore searches for Weights that produce accurate Predictions without becoming unnecessarily large.

---

## The Regularization Parameter

The parameter $r$ controls the strength of Regularization.

When:

```math
r=0
```

no Regularization is applied, and the equation returns to the original Normal Equation.

As $r$ increases:

```math
r\uparrow
\quad\Rightarrow\quad
\text{Regularization Strength}\uparrow
```

Stronger Regularization generally produces smaller Weights:

```math
r\uparrow
\quad\Rightarrow\quad
|w_j|\downarrow
```

However, the value of $r$ must be balanced.

If $r$ is too small, the Weights may remain unstable. If $r$ is too large, the Model may reduce the Weights too strongly and fail to learn important relationships.

Therefore:

```math
\text{Too Little Regularization}
\rightarrow
\text{Unstable Model}
```

```math
\text{Appropriate Regularization}
\rightarrow
\text{Stable Model}
```

```math
\text{Too Much Regularization}
\rightarrow
\text{Model Learns Too Little}
```

---

## Regularization and Feature Engineering

Feature Engineering and Regularization perform different but connected roles.

Feature Engineering determines what information is provided to the Model:

```math
\text{Raw Data}
\rightarrow
\text{Engineered Features}
\rightarrow
X
```

Regularization controls how strongly the Model uses that information:

```math
X,y
\rightarrow
\text{Regularized Training}
\rightarrow
w_{\text{regularized}}
```

Therefore:

```math
\text{Feature Engineering}
\rightarrow
\text{Improves the Representation of Data}
```

```math
\text{Regularization}
\rightarrow
\text{Improves the Stability of Training}
```

Regularization does not remove the Categorical Features. It allows the Model to use them while reducing the risk of unstable Weights.

---

## Evaluating the Regularized Model

After applying Regularization, the Model is trained again:

```math
X_{\text{train}},
y_{\text{train}},
r
\rightarrow
\text{Regularized Training}
\rightarrow
w_{\text{regularized}}
```

The learned Weights are applied to the Validation Set:

```math
X_{\text{val}},
w_{\text{regularized}}
\rightarrow
\widehat{y}_{\text{val}}
\rightarrow
RMSE
```

Before Regularization, the unstable Model produced approximately:

```math
RMSE\approx41.45
```

After Regularization:

```math
RMSE\approx0.4569
```

Therefore:

```math
41.45
\rightarrow
0.4569
```

This shows that Regularization stabilizes the Training calculation and allows the Model to use the additional Categorical Features effectively.

---

## Complete Regularization Process

The complete relationship is:

```math
\text{Many Engineered Features}
\rightarrow
\text{Related Columns}
\rightarrow
\text{Unstable }X^TX
```

Regularization modifies the Matrix:

```math
X^TX
\rightarrow
X^TX+rI
```

The regularized Matrix is used to learn controlled Weights:

```math
w_{\text{regularized}}
=
(X^TX+rI)^{-1}X^Ty
```

The complete Model process becomes:

```math
X_{\text{engineered}},y
\rightarrow
\text{Regularized Training}
\rightarrow
w_{\text{regularized}}
\rightarrow
\widehat{y}
\rightarrow
RMSE
```

The main concept is that Regularization stabilizes Linear Regression by controlling the size of the Weights. It allows the Model to use a larger set of Features without becoming excessively sensitive to duplicated or strongly related information.

---

# Tuning the Model

Regularization stabilizes the Linear Regression Model and prevents the Weights from becoming excessively large.

The strength of Regularization is controlled by the parameter $r$. Different values of $r$ produce different Weight Vectors and may affect the predictive performance of the Model.

**Model Tuning** is the process of testing different values of a Hyperparameter and selecting the value that produces the best performance on the Validation Set.

The general process is:

```math
\text{Candidate Values of }r
\rightarrow
\text{Train Multiple Models}
\rightarrow
\text{Evaluate on Validation Set}
\rightarrow
\text{Select Best }r
```

---

## Parameters and Hyperparameters

It is important to distinguish between **Model Parameters** and **Hyperparameters**.

Model Parameters are learned directly from the Training Data:

```math
X_{\text{train}},
y_{\text{train}}
\rightarrow
\text{Training}
\rightarrow
w
```

For Linear Regression, the Model Parameters are:

```math
w_0,w_1,w_2,\ldots,w_n
```

These Parameters determine how the Features contribute to the Prediction:

```math
\widehat{y}
=
w_0+\sum_{j=1}^{n}x_jw_j
```

A Hyperparameter is not learned directly by the Model. It is selected before Training and controls how the Training process works.

For Regularized Linear Regression:

```math
r=\text{Regularization Strength}
```

The relationship is:

```math
r
\rightarrow
\text{Regularized Training}
\rightarrow
w_{\text{regularized}}
```

Therefore:

```math
\text{Model Parameters}
=
\text{Values Learned During Training}
```

```math
\text{Hyperparameters}
=
\text{Values Selected to Control Training}
```

---

## The Effect of $r$ on the Model

The Regularized Normal Equation is:

```math
w=(X^TX+rI)^{-1}X^Ty
```

The value of $r$ determines how much is added to the diagonal of $X^TX$:

```math
X^TX
\rightarrow
X^TX+rI
```

When:

```math
r=0
```

the equation becomes the original Normal Equation:

```math
w=(X^TX)^{-1}X^Ty
```

Therefore, no Regularization is applied.

When $r$ is a small positive value:

```math
r>0
\rightarrow
\text{More Stable }(X^TX+rI)^{-1}
```

As $r$ increases:

```math
r\uparrow
\quad\Rightarrow\quad
\text{Regularization Strength}\uparrow
```

Stronger Regularization generally produces smaller Weights:

```math
r\uparrow
\quad\Rightarrow\quad
|w_j|\downarrow
```

However, reducing the Weights too much may prevent the Model from learning important relationships between the Features and Target.

---

## Why We Need to Tune $r$

There is no single value of $r$ that is automatically suitable for every Dataset.

If $r$ is too small:

```math
\text{Small }r
\rightarrow
\text{Insufficient Regularization}
\rightarrow
\text{Large Weights}
```

If $r$ is too large:

```math
\text{Large }r
\rightarrow
\text{Weights Reduced Too Much}
\rightarrow
\text{Loss of Predictive Information}
```

The objective is to find a balanced value:

```math
\text{Appropriate }r
\rightarrow
\text{Stable Weights}
+
\text{Good Predictions}
```

Because the best value is not known in advance, several Candidate Values must be tested.

---

## Trying Different Values of $r$

The lesson evaluates the following Candidate Values:

```math
r\in
\left\{
0,\,
0.00001,\,
0.0001,\,
0.001,\,
0.1,\,
1,\,
10
\right\}
```

These values represent different levels of Regularization:

```math
r=0
\rightarrow
\text{No Regularization}
```

```math
r\approx0
\rightarrow
\text{Weak Regularization}
```

```math
r>0
\rightarrow
\text{Stronger Regularization}
```

For each Candidate Value, a separate Model is trained:

```math
X_{\text{train}},
y_{\text{train}},
r_k
\rightarrow
\text{Training}
\rightarrow
w_k
```

Each value of $r$ therefore produces a different Weight Vector:

```math
\begin{aligned}
r_1&\rightarrow w_1\\
r_2&\rightarrow w_2\\
&\ \vdots\\
r_k&\rightarrow w_k
\end{aligned}
```

The Models are then compared using the same Validation Set.

---

## Using the Validation Set for Tuning

For each trained Model, Predictions are generated from the Validation Features:

```math
X_{\text{val}},
w_k
\rightarrow
\widehat{y}_{\text{val},k}
```

The Predictions are compared with the Actual Validation Targets:

```math
y_{\text{val}},
\widehat{y}_{\text{val},k}
\rightarrow
RMSE_{\text{val},k}
```

The process is repeated for every Candidate Value:

```math
r_k
\rightarrow
w_k
\rightarrow
\widehat{y}_{\text{val},k}
\rightarrow
RMSE_{\text{val},k}
```

The best value is selected according to:

```math
\boxed{
r_{\text{best}}
=
\underset{r}{\arg\min}
\ RMSE_{\text{val}}(r)
}
```

This means selecting the value of $r$ that produces the lowest Validation RMSE.

---

## Reading the Results

When:

```math
r=0
```

the Model receives no Regularization. The duplicated or strongly related Features make the Training calculation unstable.

As a result:

```math
|w_0|
\rightarrow
\text{Very Large}
```

and:

```math
RMSE\approx266
```

When a small positive value is added:

```math
r>0
```

the Model immediately becomes more stable. The Weight values decrease, and the Validation RMSE returns to a reasonable level.

The relationship is:

```math
r=0
\rightarrow
\text{Unstable Weights}
\rightarrow
\text{High RMSE}
```

```math
r>0
\rightarrow
\text{Stable Weights}
\rightarrow
\text{Lower RMSE}
```

After the Model becomes stable, increasing $r$ slightly does not initially change RMSE very much.

However, when $r$ becomes too large:

```math
r\uparrow
\rightarrow
\text{Weights Become Smaller}
\rightarrow
\text{Model Becomes More Restricted}
```

When:

```math
r=10
```

the Model is regularized too strongly, and its Validation performance becomes worse.

The overall pattern is:

```math
\text{No Regularization}
\rightarrow
\text{Unstable Model}
```

```math
\text{Small Regularization}
\rightarrow
\text{Stable Model with Good Performance}
```

```math
\text{Strong Regularization}
\rightarrow
\text{Model Performance Degrades}
```

---

## Bias and Regularization

In the implementation used in this lesson, the Bias $w_0$ becomes smaller as the Regularization Parameter increases:

```math
r\uparrow
\quad\Rightarrow\quad
|w_0|\downarrow
```

This happens because the Identity Matrix is added to the entire diagonal of $X^TX$. Therefore, the Bias is regularized together with the other Weights.

However, a smaller Bias does not automatically mean a better Model:

```math
\text{Smaller Weights}
\nRightarrow
\text{Automatically Better Predictions}
```

Model quality must still be evaluated using the Validation RMSE.

---

## Selecting the Regularization Parameter

Several small values of $r$ produce similar RMSE results. Therefore, there may not be one clearly superior value.

The lesson selects:

```math
\boxed{r=0.001}
```

This value is appropriate because:

- It is large enough to stabilize the Model.
- It prevents the Weights from becoming excessively large.
- The Validation RMSE has not started to degrade.
- It provides performance similar to the best Candidate Values.

The selection is based on both stability and Validation performance:

```math
r_{\text{selected}}
=
\text{Stable Model}
+
\text{Low Validation RMSE}
```

---

## Training the Selected Model

After selecting $r$, the Linear Regression Model is trained again using:

```math
r=0.001
```

The Training process is:

```math
X_{\text{train}},
y_{\text{train}},
r=0.001
\rightarrow
\text{Regularized Training}
\rightarrow
w_{\text{selected}}
```

The learned Model is applied to the Validation Set:

```math
X_{\text{val}},
w_{\text{selected}}
\rightarrow
\widehat{y}_{\text{val}}
```

Finally, its performance is measured:

```math
y_{\text{val}},
\widehat{y}_{\text{val}}
\rightarrow
RMSE_{\text{val}}
```

The Validation RMSE is approximately:

```math
RMSE_{\text{val}}
\approx
0.4608
```

This confirms that the selected value produces a stable Model with good Validation performance.

---

## Why the Test Set Is Not Used for Tuning

The Test Set must remain separate during Model Tuning.

If the Test Set is used to select $r$, information from the Test Data influences Model Development:

```math
\text{Test Results}
\rightarrow
\text{Hyperparameter Selection}
```

The Test Set would no longer represent completely unseen Data.

The correct separation of responsibilities is:

```math
\text{Training Set}
\rightarrow
\text{Learn the Weights}
```

```math
\text{Validation Set}
\rightarrow
\text{Select the Hyperparameter}
```

```math
\text{Test Set}
\rightarrow
\text{Final Evaluation}
```

The Validation Set can be used repeatedly while comparing Candidate Values. The Test Set should be used only after the Model and Hyperparameters have been selected.

---

## Training and Tuning Are Different Processes

Training determines the Model Parameters for a fixed value of $r$:

```math
X_{\text{train}},
y_{\text{train}}
\xrightarrow{r\text{ fixed}}
w
```

Tuning compares the results obtained from different values of $r$:

```math
r_1,r_2,\ldots,r_k
\rightarrow
RMSE_1,RMSE_2,\ldots,RMSE_k
```

Therefore:

```math
\text{Training}
\rightarrow
\text{Learns }w
```

```math
\text{Tuning}
\rightarrow
\text{Selects }r
```

Training occurs multiple times during Tuning because every Candidate Value requires a newly trained Model.

---

## Complete Model Tuning Process

The complete process is:

```math
\text{Define Candidate Values of }r
```

```math
\downarrow
```

```math
\text{Train One Model for Each }r
```

```math
\downarrow
```

```math
\text{Generate Validation Predictions}
```

```math
\downarrow
```

```math
\text{Calculate Validation RMSE}
```

```math
\downarrow
```

```math
\text{Compare the Results}
```

```math
\downarrow
```

```math
\boxed{\text{Select }r=0.001}
```

The selected Model Development process is:

```math
X_{\text{train}},
y_{\text{train}},
r=0.001
\rightarrow
w_{\text{selected}}
\rightarrow
\widehat{y}_{\text{val}}
\rightarrow
RMSE_{\text{val}}\approx0.4608
```

The main concept is that Model Tuning uses the Validation Set to select the Hyperparameter that provides an appropriate balance between Model stability and predictive performance.

The next step is to evaluate and use the selected Model with the Test Set, which has remained untouched throughout Training and Tuning.

# Using the Model

After Feature Engineering, Regularization, and Model Tuning, we have selected the Model configuration that performs best on the Validation Set.

The selected Regularization Parameter is:

```math
\boxed{r=0.001}
```

At this stage, Model Development is complete. The next steps are:

1. Combine the Training and Validation Data.
2. Train the Final Model using the selected Hyperparameter.
3. Evaluate the Final Model on the Test Set.
4. Use the Final Model to predict the price of a new car.

The overall process is:

```math
\text{Training and Validation Data}
\rightarrow
\text{Final Training}
\rightarrow
\text{Test Evaluation}
\rightarrow
\text{New Prediction}
```

---

## From Model Development to Final Training

During Model Development, the Dataset was divided into three parts:

```math
\text{Training Set}
+
\text{Validation Set}
+
\text{Test Set}
```

Each part has a different responsibility:

```math
\text{Training Set}
\rightarrow
\text{Learn the Weights}
```

```math
\text{Validation Set}
\rightarrow
\text{Select Features and Hyperparameters}
```

```math
\text{Test Set}
\rightarrow
\text{Final Evaluation}
```

The Validation Set was required while comparing different Feature combinations and values of $r$.

Once the final Features and Regularization Parameter have been selected, the Validation Set is no longer needed for Model Tuning. It can now be combined with the Training Set to provide more Data for Final Training.

---

## Combining Training and Validation Data

The Training and Validation DataFrames are combined into one Dataset:

```math
df_{\text{full train}}
=
df_{\text{train}}
\cup
df_{\text{val}}
```

Their Target Vectors must also be combined:

```math
y_{\text{full train}}
=
\begin{bmatrix}
y_{\text{train}}\\
y_{\text{val}}
\end{bmatrix}
```

If the original Data Split was:

```math
60\%\text{ Training}
+
20\%\text{ Validation}
+
20\%\text{ Test}
```

the new structure becomes:

```math
80\%\text{ Full Training}
+
20\%\text{ Test}
```

Therefore:

```math
\boxed{
\text{Full Training Set}
=
\text{Training Set}
+
\text{Validation Set}
}
```

---

## Why Training and Validation Are Combined

During Model Tuning, the Validation Set must remain separate so that it can provide an independent comparison between Candidate Models.

After selecting:

- The Features
- The Feature Engineering Process
- The Regularization Method
- The Regularization Parameter $r=0.001$

there are no more Model Development decisions for the Validation Set to support.

Combining it with the Training Set allows the Final Model to learn from more observations:

```math
m_{\text{full train}}
=
m_{\text{train}}
+
m_{\text{val}}
```

The process changes from:

```math
X_{\text{train}},
y_{\text{train}}
\rightarrow
w_{\text{development}}
```

to:

```math
X_{\text{full train}},
y_{\text{full train}}
\rightarrow
w_{\text{final}}
```

The Final Weight Vector may differ from the previous Weight Vector because it is learned from a larger Dataset.

---

## Preparing the Full Training Data

The combined Data must pass through the same Feature Engineering process used during Model Development.

This includes:

- Selecting the Base Numerical Features
- Handling Missing Values
- Creating the `age` Feature
- Encoding the selected Categorical Variables
- Maintaining the same Feature order

Conceptually:

```math
df_{\text{full train}}
\rightarrow
\text{Feature Engineering}
\rightarrow
X_{\text{full train}}
```

The Final Training Matrix contains:

```math
X_{\text{full train}}
=
\begin{bmatrix}
\text{Numerical Features}&
\text{Engineered Features}&
\text{Categorical Features}
\end{bmatrix}
```

The Target remains in the transformed Log Scale:

```math
y_{\text{full train}}
=
\log(1+\text{price})
```

The Target must remain separate from the Feature Matrix to prevent Data Leakage:

```math
X_{\text{full train}}
\cap
y_{\text{full train}}
=
\varnothing
```

---

## Training the Final Model

The Final Model is trained using:

```math
X_{\text{full train}}
```

```math
y_{\text{full train}}
```

and the selected Regularization Parameter:

```math
r=0.001
```

The Final Training process is:

```math
X_{\text{full train}},
y_{\text{full train}},
r=0.001
\rightarrow
\text{Regularized Training}
\rightarrow
w_{\text{final}}
```

Using the Regularized Normal Equation:

```math
\boxed{
w_{\text{final}}
=
\left(
X_{\text{full train}}^TX_{\text{full train}}
+rI
\right)^{-1}
X_{\text{full train}}^T
y_{\text{full train}}
}
```

The Final Model is represented by:

```math
g_{\text{final}}(X)
=
w_0+Xw
```

The Final Weights are learned from both the Training and Validation observations.

---

## The Test Set Remains Separate

The Test Set must not be included in Final Training:

```math
df_{\text{test}}
\not\subset
df_{\text{full train}}
```

It represents unseen Data that has not influenced:

- Feature selection
- Hyperparameter selection
- Model comparison
- Final Weight estimation

Therefore:

```math
\text{Full Training Data}
\rightarrow
\text{Learn the Final Model}
```

```math
\text{Test Data}
\rightarrow
\text{Evaluate the Final Model}
```

If the Test Set were used during Training or Tuning, it would no longer provide an independent estimate of Model performance.

---

## Preparing the Test Data

The Test Data must pass through exactly the same Feature Engineering process as the Full Training Data:

```math
df_{\text{test}}
\rightarrow
\text{Feature Engineering}
\rightarrow
X_{\text{test}}
```

The Feature Matrices must contain the same Columns:

```math
\text{Columns of }X_{\text{full train}}
=
\text{Columns of }X_{\text{test}}
```

Their dimensions may be represented as:

```math
X_{\text{full train}}
\in
\mathbb{R}^{m_{\text{full train}}\times p}
```

```math
X_{\text{test}}
\in
\mathbb{R}^{m_{\text{test}}\times p}
```

The number of Rows may differ, but both Matrices must contain the same $p$ Features.

The meaning and order of the Columns must also remain identical:

```math
X_{\text{full train},\,\cdot j}
\equiv
X_{\text{test},\,\cdot j}
```

This ensures that every Final Weight is applied to the correct Test Feature.

---

## Evaluating the Final Model

The Final Weight Vector is applied to the Test Feature Matrix:

```math
\widehat{y}_{\text{test}}
=
w_0+X_{\text{test}}w
```

The Predictions are compared with the Actual Test Targets:

```math
y_{\text{test}},
\widehat{y}_{\text{test}}
\rightarrow
RMSE_{\text{test}}
```

The Test RMSE is:

```math
\boxed{
RMSE_{\text{test}}
=
\sqrt{
\frac{1}{m_{\text{test}}}
\sum_{i=1}^{m_{\text{test}}}
(\widehat{y}_i-y_i)^2
}
}
```

This Test RMSE provides the final estimate of how well the selected Model performs on unseen Data.

The Model demonstrates reasonable Generalization when:

```math
RMSE_{\text{test}}
\approx
RMSE_{\text{val}}
```

A large difference could indicate that the Model Development decisions were too closely adapted to the Validation Set.

---

## Validation Performance and Test Performance

The Validation RMSE was used to select the Model:

```math
RMSE_{\text{val}}
\rightarrow
\text{Model Selection}
```

The Test RMSE is used to evaluate the selected Final Model:

```math
RMSE_{\text{test}}
\rightarrow
\text{Final Performance Estimate}
```

The Validation result answers:

> Which Model configuration should we choose?

The Test result answers:

> How well does the selected Model perform on unseen Data?

Therefore:

```math
\text{Validation}
\neq
\text{Final Evaluation}
```

The Test Set should be evaluated only after the Model Development decisions have been completed.

---

## Using the Model for a New Observation

After the Final Model has been trained and evaluated, it can be used to predict the price of a new car.

The new observation contains the original car characteristics:

```math
x_{\text{new}}
=
\begin{bmatrix}
\text{year}&
\text{engine information}&
\text{fuel type}&
\text{transmission}&
\text{vehicle characteristics}
\end{bmatrix}
```

Before Prediction, the new observation must pass through the same Feature Engineering process:

```math
\text{New Car Data}
\rightarrow
\text{Feature Engineering}
\rightarrow
x_{\text{new prepared}}
```

This includes:

- Creating the `age` Feature
- Handling Missing Values
- Encoding the same Categorical Variables
- Maintaining the same Feature order

The prepared observation must contain the same $p$ Features as the Final Training Matrix:

```math
x_{\text{new prepared}}
\in
\mathbb{R}^{1\times p}
```

```math
X_{\text{full train}}
\in
\mathbb{R}^{m_{\text{full train}}\times p}
```

The Model generates one Prediction:

```math
\widehat{y}_{\text{new}}
=
w_0+x_{\text{new prepared}}w
```

The dimensions are:

```math
\underbrace{x_{\text{new prepared}}}_{1\times p}
\underbrace{w}_{p\times1}
=
\underbrace{\widehat{y}_{\text{new}}}_{1\times1}
```

This is the same Linear Regression calculation used throughout the project, but it is now applied to one new observation.

---

## Prediction in the Log Scale

Earlier in the project, the Target was transformed using:

```math
y=\log(1+\text{price})
```

The Model was trained to predict this transformed Target:

```math
\widehat{y}
=
\widehat{\log(1+\text{price})}
```

Therefore, the direct output of the Model is not yet the car price in its original monetary scale.

To convert the Prediction back, we apply the inverse transformation:

```math
\boxed{
\widehat{\text{price}}
=
\exp(\widehat{y})-1
}
```

This is equivalent to:

```math
\widehat{\text{price}}
=
\operatorname{expm1}(\widehat{y})
```

The complete Prediction process is:

```math
\text{New Car}
\rightarrow
x_{\text{new prepared}}
\rightarrow
\widehat{y}_{\log}
\rightarrow
\exp(\widehat{y}_{\log})-1
\rightarrow
\widehat{\text{price}}
```

Without the inverse transformation, the result would remain in the Log Scale and could not be interpreted directly as the predicted car price.

---

## Evaluation and Prediction Are Different

During Test Evaluation, the Model processes many observations:

```math
X_{\text{test}}
\rightarrow
\widehat{y}_{\text{test}}
\rightarrow
RMSE_{\text{test}}
```

When using the Model, it may process only one new observation:

```math
x_{\text{new}}
\rightarrow
\widehat{\text{price}}_{\text{new}}
```

Evaluation measures the overall quality of the Model across a Dataset:

```math
\text{Many Predictions}
\rightarrow
\text{One Performance Metric}
```

Using the Model produces a Prediction for a specific observation:

```math
\text{One Observation}
\rightarrow
\text{One Predicted Value}
```

The same Final Model and Feature Engineering process are used in both cases.

---

## Complete Final Model Process

The process begins with the selected Regularization Parameter:

```math
r_{\text{selected}}
=
0.001
```

The Training and Validation Sets are combined:

```math
df_{\text{train}}
+
df_{\text{val}}
\rightarrow
df_{\text{full train}}
```

The combined Data is prepared:

```math
df_{\text{full train}}
\rightarrow
X_{\text{full train}}
```

The Final Model is trained:

```math
X_{\text{full train}},
y_{\text{full train}},
r=0.001
\rightarrow
w_{\text{final}}
```

The Final Model is evaluated on the Test Set:

```math
X_{\text{test}},
w_{\text{final}}
\rightarrow
\widehat{y}_{\text{test}}
\rightarrow
RMSE_{\text{test}}
```

Finally, the Model can be applied to new Data:

```math
x_{\text{new}}
\rightarrow
\text{Feature Engineering}
\rightarrow
\widehat{y}_{\log}
\rightarrow
\widehat{\text{price}}
```

The complete process is:

```math
\boxed{
\text{Select Model}
\rightarrow
\text{Combine Train and Validation}
\rightarrow
\text{Train Final Model}
\rightarrow
\text{Evaluate on Test Data}
\rightarrow
\text{Predict New Values}
}
```

The main concept is that after the Features and Hyperparameters have been selected, the Training and Validation Sets are combined to train the Final Model. The untouched Test Set is then used for one final evaluation, and the completed Model can subsequently generate Predictions for new observations.
