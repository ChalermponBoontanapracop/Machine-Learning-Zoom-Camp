# Classification: Core Concepts Before Building a Model

**Classification** uses data to predict which category an observation belongs to. When there are two possible categories, the task is called **binary classification**. We represent the category of interest as $1$ and the other category as $0$.

This differs from Regression, which predicts a continuous numerical value. In Classification, $0$ and $1$ are **labels for categories**, not quantities.

### 1. Define the Prediction Task

Before building a model, we identify two parts of the data:

- **Features ($X$)** are the information used to make a prediction.

- **Target ($y$)** is the actual outcome we want to predict.

For observation $i$, we can express the goal as:

$$
g\left( x_{i} \right) \approx y_{i}
$$

Here, $x_{i}$ contains the features, $g$ is the model, and $y_{i}$ is the actual category. In binary classification, $y_{i}$ is either $0$ or $1$.

A classification model often produces a **score or estimated probability for class 1** between 0 and 1. We can then apply a *threshold* to turn that score into a predicted class. The score and the final class prediction are therefore two different things.

### 2. Prepare the Data

Raw data may contain inconsistent text, missing values, or columns stored in the wrong data type. Data preparation checks and corrects these issues so each feature has a clear and consistent meaning.

We also encode the target as $0$ and $1$, making it clear what each value means. This definition matters because we will use it to interpret predictions and evaluate the model.

Another important question is whether each feature would be **available at the time of prediction**. Information that only becomes known after the outcome occurs should not be used to train a model intended to predict that outcome in advance.

### 3. Split the Data

As in the Regression unit, we divide the data into three sets:

| **Dataset**        | **Purpose**                                                      |
|--------------------|------------------------------------------------------------------|
| **Train set**      | Teach the model the relationship between features and the target |
| **Validation set** | Check performance and compare choices during development         |
| **Test set**       | Evaluate the selected model at the end                           |

The aim is to find out whether the model can predict **new observations**, rather than merely remember the data used for training. We use the validation set while developing the model and reserve the test set for the final evaluation.

### 4. Explore the Data with EDA

**Exploratory Data Analysis (EDA)** helps us understand the dataset before training a model. We examine the number of observations, data types, missing or unusual values, and the distribution of the target.

In Classification, the **proportion of observations in each class** is particularly important. If one class is much more common than the other, we will need to interpret the model’s evaluation results carefully.

We can also explore how features relate to the target. These patterns may suggest which features could be useful for prediction, but an observed relationship alone does not establish cause and effect.

### Putting the Steps Together

$$
\begin{aligned}
&\text{Define the target} \rightarrow \text{Prepare the data} \\
&\rightarrow \text{Split the data} \rightarrow \text{Explore the data} \\
&\rightarrow \text{Train and evaluate the model}
\end{aligned}
$$

The key idea is to define **what we want to predict, which information we can use, and how we will check whether the model works on new data**. Once these decisions are clear, we can study how a classification algorithm learns and produces predictions.

## Feature Importance and Preparing Data for Classification

After preparing and exploring a dataset, the next step is to understand how its features relate to the target. Categorical and numerical features require different ways of examining those relationships. Categorical features may also need to be converted into numbers before a model can use them.

## Risk: Comparing Groups with the Overall Rate

**Risk** is the proportion of observations where the target equals $1$. It can be calculated for the entire dataset or for one group within a categorical feature.

$$
\text{Global~risk} = \frac{\text{Number~of~observations~with~}y = 1}{\text{Total~number~of~observations}}
$$

$$
\text{Group~risk} = \frac{\text{Number~of~observations~with~}y = 1\text{~in~the~group}}{\text{Total~number~of~observations~in~the~group}}
$$

Comparing group risk with global risk shows whether the target outcome is more or less common in that group. There are two ways to express this comparison:

$$
\text{Risk~difference} = \text{Group~risk} - \text{Global~risk}
$$

$$
\text{Risk~ratio} = \frac{\text{Group~risk}}{\text{Global~risk}}
$$

A positive risk difference, or a risk ratio greater than $1$, means the outcome is more common in the group than it is overall. The number of observations in the group matters too: a rate calculated from only a few observations may be unstable.

### Mutual Information: Comparing Categorical Features

Risk helps us inspect individual categories. **Mutual information** helps us assess a categorical feature as a whole.

It measures how much knowing a feature reduces uncertainty about the target. If a feature has no relationship with the target, its mutual information is $0$. Higher values indicate that the feature provides more information about the target.

Mutual information is useful for **ranking categorical features**, but it does not tell us which specific categories have higher or lower risk. To understand that direction, we return to the group risk calculations.

### Correlation: Examining Numerical Features

**Correlation** measures the direction and strength of a *linear* relationship. Its value ranges from $- 1$ to $1$.

| **Value**  | **Interpretation**                                             |
|------------|----------------------------------------------------------------|
| Near $1$   | Larger feature values tend to occur with larger target values  |
| Near $- 1$ | Larger feature values tend to occur with smaller target values |
| Near $0$   | Little linear relationship is visible                          |

With a target encoded as $0$ and $1$, a positive correlation means larger values of a numerical feature tend to occur more often with class $1$. A negative correlation suggests the reverse.

Correlation describes a relationship in the data; it does not prove cause and effect. A value near zero also does not necessarily mean a feature is useless—the relationship may be nonlinear.

## One-Hot Encoding: Preparing Categorical Features

A categorical feature contains labels, while many machine learning models require numerical input. **One-hot encoding** creates a separate binary column for each category. Each new column contains $1$ when an observation belongs to that category and $0$ otherwise.

**Before encoding — one categorical column**

| **Observation** | **Original feature** |
|-----------------|----------------------|
| 1               | value1               |
| 2               | value2               |
| 3               | value3               |
| 4               | value1               |

**After encoding — three binary columns**

| **Observation** | **value1** | **value2** | **value3** |
|-----------------|------------|------------|------------|
| 1               | 1          | 0          | 0          |
| 2               | 0          | 1          | 0          |
| 3               | 0          | 0          | 1          |
| 4               | 1          | 0          | 0          |

The observation numbers connect the rows in the two tables. For instance, observation 2 originally contains value2, so its encoded values are $\left\lbrack 0,1,0 \right\rbrack$.

This avoids giving categories an artificial order. If category names were replaced with $1$, $2$, and $3$, a model might treat those numbers as quantities. One-hot encoding represents each category without implying that one is greater than another.

The same encoding must be applied consistently to the data used for training, validation, testing, and future predictions.

### Connecting the Ideas

**Risk** shows how the target rate differs across groups. **Mutual information** compares categorical features as a whole. **Correlation** examines linear relationships involving numerical features. **One-hot encoding** then turns categorical values into numerical inputs.

Together, these steps help us understand the features and prepare a **feature matrix** $X$ for classification.

## Logistic Regression: From Features to Classification Probabilities

The feature matrix $X$ contains the information a classification model uses to make predictions. **Logistic Regression** combines those features to estimate the probability that an observation belongs to class $1$.

Despite the word *regression* in its name, its purpose here is **classification**. It produces a probability between $0$ and $1$, which can then be converted into a class prediction.

### 1. Start with a Linear Score

The first calculation should look familiar from Linear Regression. For one observation with features $x_{1},x_{2},\ldots,x_{m}$, the model combines the features with learned weights:

$$
z = w_{0} + w_{1}x_{1} + w_{2}x_{2} + \cdots + w_{m}x_{m}
$$

Here:

- $w_{0}$ is the **bias** or **intercept**.

- $w_{1},\ldots,w_{m}$ are the **weights** associated with the features.

- $z$ is the resulting **linear score**.

The same calculation can be written in vector form:

$$
z = w_{0} + x^{T}w
$$

The linear score can be any real number: negative, zero, or positive. It is **not yet a probability**, because probabilities must be between $0$ and $1$.

### 2. Convert the Score into a Probability

To convert the linear score into a probability, Logistic Regression uses the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{- z}}
$$

The complete prediction function is:

$$
g(x) = \sigma\,\left( w_{0} + x^{T}w \right) = \frac{1}{1 + e^{- \left( w_{0} + x^{T}w \right)}}
$$

The graph shows how a score $z$ becomes an estimated probability. Different values of the score correspond to different points on the curve.

**From linear score to probability**

![Sigmoid function mapping a linear score to an estimated probability](media/image1.png)

When $z = 0$, the sigmoid output is $0.5$. As $z$ becomes more positive, the output approaches $1$. As $z$ becomes more negative, it approaches $0$.

We interpret the output as the model’s **estimated probability of class** $1$:

$$
\widehat{p} = P(y = 1 \mid x)
$$

The estimated probability of class $0$ is:

$$
P(y = 0 \mid x) = 1 - \widehat{p}
$$

For example, if $\widehat{p} = 0.8$, the model estimates a probability of $0.8$ for class $1$ and $0.2$ for class $0$. This is an estimate based on learned patterns, not a guarantee about the outcome.

### 3. Follow One Prediction from Start to Finish

Suppose the weights and bias have already been learned:

$$
w_{0} = - 1,\ w_{1} = 0.8,\ w_{2} = 0.4
$$

For an observation with $x_{1} = 2$ and $x_{2} = 1$, the linear score is:

$$
\begin{aligned}
z & = w_{0} + w_{1}x_{1} + w_{2}x_{2} \\
 & = - 1 + (0.8)(2) + (0.4)(1) \\
 & = 1
\end{aligned}
$$

Next, apply the sigmoid function:

$$
\widehat{p} = \sigma(1) = \frac{1}{1 + e^{- 1}} \approx 0.731
$$

The model assigns this observation an estimated probability of approximately **0.731 for class** $1$. On the first graph, this result corresponds to $z = 1$.

Notice that these are two separate calculations:

$$
\boxed{\text{Features} \longrightarrow \text{Linear~score~}z \longrightarrow \text{Estimated~probability~}\widehat{p}}
$$

The weights are treated as already known in this calculation. Learning those weights is the job of the training process.

### 4. Understand What the Weights Do

Each weight changes the linear score before that score enters the sigmoid function.

- A **positive weight** increases $z$ when its feature increases. This increases the estimated probability of class $1$.

- A **negative weight** decreases $z$ when its feature increases. This decreases the estimated probability of class $1$.

- A weight close to **zero** makes a relatively small contribution for the same change in its feature.

The bias $w_{0}$ contributes to every prediction. It sets the starting point of the linear score before the features make their contributions.

For a one-hot encoded feature, the value of a category column is either $0$ or $1$. Its contribution is:

$$
w_{j}x_{j} = \begin{cases}
w_{j} & \text{when~the~category~is~present} \\
0 & \text{when~the~category~is~absent}
\end{cases}
$$

This connects One-Hot Encoding to Logistic Regression. Encoding gives each category its own numerical column; training allows the model to learn a weight for that column.

A weight describes a relationship **within the model**. It does not prove that a feature causes the outcome. We should also be careful when comparing the sizes of weights for numerical features measured in different units.

### 5. Turn a Probability into a Class Prediction

The estimated probability $\widehat{p}$ is not yet a predicted class. To assign a class, we choose a **threshold** $t$:

$$
\widehat{y} = \begin{cases}
1 & \text{if~}\widehat{p} \geq t \\
0 & \text{if~}\widehat{p} < t
\end{cases}
$$

The graph shows the decision rule. Probabilities on either side of the threshold are assigned to class $0$ or class $1$.

**How a threshold changes the predicted class**

![Class prediction as a function of estimated probability and threshold](media/image2.png)

With a threshold of $0.5$, the probability $0.731$ from the earlier calculation becomes a prediction of class $1$.

The threshold is a **decision rule**. Changing it does not change the model’s estimated probability for an observation; it changes how we turn that probability into a class.

| **Output**                          | **What it tells us**                     |
|-------------------------------------|------------------------------------------|
| Estimated probability $\widehat{p}$ | How likely the model considers class $1$ |
| Predicted class $\widehat{y}$       | Which class the chosen threshold assigns |

### 6. Learn the Weights from Data

During training, the model receives observations with their known target values. It adjusts the weights and bias so that observations labeled $1$ tend to receive higher estimated probabilities for class $1$, while observations labeled $0$ tend to receive lower ones.

The model needs a way to measure how well its probabilities match the actual targets. A training objective does this by giving a larger penalty to predictions that are far from the correct outcome. In particular, a confident but incorrect prediction should be penalized more heavily than an uncertain prediction.

Training repeatedly adjusts the weights to reduce the overall penalty. Once training is complete, the weights and bias are used to make predictions for observations whose outcomes are not yet known.

The distinction is:

- **During training:** the model learns $w_{0}$ and $w$ from features and known targets.

- **During prediction:** the model uses those learned values to calculate $z$ and then $\widehat{p}$ for a new observation.

### 7. Check Whether the Model Works on New Data

A model can learn patterns from the Train set without necessarily making good predictions on new observations. We therefore evaluate it using data that was not used to learn its weights.

The **Validation set** helps us examine the model during development. We calculate probabilities for its observations and compare them with their known targets. This tells us whether the relationships learned from the Train set also appear useful for data the model has not seen during training.

After selecting the model and the decisions involved in its use, we evaluate it on the **Test set**. This provides a final check on data kept separate throughout development.

When assessing predictions, we should keep two questions distinct:

1.  **Are the estimated probabilities useful?**

2.  **After applying a threshold, are the assigned classes useful?**

These questions concern related but different outputs. Choosing how to evaluate them is the next step after understanding how Logistic Regression produces a prediction.

### The Complete Prediction Process

For one observation, Logistic Regression follows this sequence:

$$
x\,\overset{\,w_{0} + x^{T}w\,}{\rightarrow}z\,\overset{\,\sigma(z)\,}{\rightarrow}\widehat{p}\,\overset{\,\text{threshold}\,}{\rightarrow}\widehat{y}
$$

**Logistic Regression begins with a weighted combination of features, uses the sigmoid function to turn that score into an estimated probability, and applies a threshold when a class prediction is needed.**

## Training a Logistic Regression Model

Previously, we treated the weights and bias as if they were already known. **Training** is the process of learning those values from observations whose outcomes are known.

For each observation $i$, Logistic Regression calculates a linear score and converts it into an estimated probability of class $1$:

$$
z_{i} = w_{0} + x_{i}^{T}w
$$

$$
{\widehat{p}}_{i} = \sigma\left( z_{i} \right) = \frac{1}{1 + e^{- z_{i}}}
$$

Here, $x_{i}$ contains the features, $y_{i}$ is the actual target ($0$ or $1$), and ${\widehat{p}}_{i}$ is the model’s estimated probability that $y_{i} = 1$. Training finds values for $w_{0}$ and $w$ that make those probabilities fit the known targets as well as possible.

### 1. Prepare the Feature Matrix and Target

Training needs two aligned pieces of information:

| **Input**            | **Meaning**                                                                                 |
|----------------------|---------------------------------------------------------------------------------------------|
| $X_{\text{train}}$ | The feature matrix: one row per observation and one column per numerical or encoded feature |
| $y_{\text{train}}$ | The known target for each corresponding row                                                 |

The order of the rows matters. The target in position $i$ must describe the same observation as row $i$ of the feature matrix.

Categorical features must be converted into numerical columns before they enter the model. When preparing validation data, the same feature definitions and column order must be used. Otherwise, a weight learned for one feature could be applied to a different feature.

The target itself is **not included among the features**. It supplies the answer the model learns from; it must not supply information when the model makes a prediction.

### 2. Make Predictions with the Current Weights

Training starts with a set of weights and a bias. Using their current values, the model calculates ${\widehat{p}}_{i}$ for every observation in the Train set.

At this point, the predictions may be poor. Some observations whose actual target is $1$ may receive a low probability for class $1$, while observations whose target is $0$ may receive a high one. The model needs a way to measure these errors so it can improve the weights.

### 3. Measure the Error with Log Loss

Logistic Regression is trained using a loss based on the **predicted probabilities** and the actual targets. For one observation, binary log loss can be written as:

$$
L_{i} = - \left\lbrack y_{i}\log\left( {\widehat{p}}_{i} \right) + \left( 1 - y_{i} \right)\log\left( 1 - {\widehat{p}}_{i} \right) \right\rbrack
$$

Because $y_{i}$ can only be $0$ or $1$, one part of the expression becomes zero:

$$
L_{i} = \begin{cases}
 - \log\left( {\widehat{p}}_{i} \right) & \text{if~}y_{i} = 1 \\
 - \log\left( 1 - {\widehat{p}}_{i} \right) & \text{if~}y_{i} = 0
\end{cases}
$$

This gives the loss an important property: **a confident prediction that is wrong receives a large penalty**. A probability close to $1$ fits an actual target of $1$ well, but fits an actual target of $0$ poorly. The reverse is true for a probability close to $0$.

The training objective combines the losses across observations, commonly by taking their average:

$$
J\left( w,w_{0} \right) = \frac{1}{n}\sum_{i = 1}^{n}L_{i}
$$

The model seeks weights and a bias that reduce this overall loss. Some formulations also add a **regularization term** to control the size of the weights.

### 4. Adjust the Weights

The weights determine how each feature changes the linear score $z_{i}$. Changing the score also changes the probability produced by the sigmoid function.

Training therefore follows a repeated pattern:

1.  Calculate probabilities using the current weights.

2.  Compare those probabilities with the known targets.

3.  Measure the loss.

4.  Adjust the weights and bias to reduce the loss.

The aim is to find a set of weights that captures useful patterns across the **Train set**. It is not to assign a separate rule or weight to every individual observation.

The result of training is a fitted model: its weights and bias are now available for making predictions on data whose targets are unknown.

### 5. Apply the Fitted Model to Validation Data

After training, the model receives $X_{\text{val}}$. It uses the **same learned weights and bias** to calculate a probability for each validation observation:

$$
{\widehat{p}}_{i}^{(val)} = \sigma\,\left( w_{0} + x_{i}^{(val)T}w \right)
$$

The validation targets are then used to assess the predictions. They are **not used to learn the weights** in this training run.

This separation answers a question that Train set performance cannot answer on its own: *Do the learned relationships remain useful for observations the model did not train on?*

### 6. Convert Probabilities into Classes When Needed

Training and probability prediction do not require a threshold. A threshold becomes relevant when we want a **class label**:

$$
{\widehat{y}}_{i} = \begin{cases}
1 & \text{if~}{\widehat{p}}_{i} \geq t \\
0 & \text{if~}{\widehat{p}}_{i} < t
\end{cases}
$$

For a threshold $t = 0.5$, probabilities of at least $0.5$ are assigned to class $1$. The distinction remains:

- **Log loss** examines the probability assigned to the actual outcome.

- A **class-based evaluation** examines the label assigned after applying a threshold.

Changing the threshold changes the class predictions, but it does **not** retrain the model or change its probabilities.

### 7. Evaluate the Validation Predictions

One initial measure is **accuracy**, the proportion of validation observations assigned the correct class:

$$
\text{Accuracy} = \frac{\text{Number~of~correct~class~predictions}}{\text{Total~number~of~predictions}}
$$

Accuracy is straightforward to interpret, but it should be read alongside the distribution of the target. If one class is much more common, a model can achieve a seemingly high accuracy while doing a poor job of finding the less common class. This is why exploring the class proportions earlier was important.

The Validation set helps us decide whether the fitted model is useful and whether development choices need adjustment. The **Test set** remains separate until those choices have been made.

### Connecting Training to Prediction

The complete process has two distinct phases:

$$
\begin{aligned}
\text{Training:}\  & \left( X_{\text{train}},y_{\text{train}} \right) \longrightarrow \text{learn~}w_{0},w \\
\text{Prediction:}\  & X_{\text{val}} \longrightarrow z \longrightarrow \widehat{p} \longrightarrow \widehat{y}\text{~if~a~threshold~is~applied}
\end{aligned}
$$

**Training learns the weights from known outcomes. Prediction holds those weights fixed and applies them to new feature rows. Validation then checks how well the resulting predictions match outcomes the model did not learn from.**

## Interpreting a Logistic Regression Model

Training produces a **bias** and a **weight for each feature**. These values help us understand how the model combines information to produce a prediction.

The key is to interpret the weights at the correct stage of the calculation. Logistic Regression first creates a **linear score**, then passes that score through the sigmoid function:

$$
z = w_{0} + w_{1}x_{1} + \cdots + w_{m}x_{m}
$$

$$
\widehat{p} = \sigma(z) = \frac{1}{1 + e^{- z}}
$$

A weight contributes to $z$. It is **not itself a probability**, and adding a weight of $0.2$ does not necessarily increase the probability by 20 percentage points. The sigmoid function determines how a change in $z$ affects the final probability.

### 1. Interpret the Bias

The bias $w_{0}$, also called the **intercept**, is the linear score when every feature value is zero:

$$
z = w_{0}\ \text{when}\ x_{1} = x_{2} = \cdots = x_{m} = 0
$$

Passing the bias through the sigmoid function gives the probability for that zero-feature configuration:

$$
\widehat{p} = \sigma\left( w_{0} \right)
$$

For example, if $w_{0} = - 1$:

$$
\widehat{p} = \sigma( - 1) = \frac{1}{1 + e^{1}} \approx 0.269
$$

The model therefore assigns a probability of approximately $0.269$ to the configuration where all its feature values are zero.

**This value is not automatically the overall rate of class** $1$ **in the dataset.** Its meaning depends on how the features were prepared. After one-hot encoding, an all-zero row may represent a reference category, or it may be a combination that does not occur in the data. A numerical value of zero may also be far from a typical observation. The bias is a starting point for the model’s calculation, but we must know what “all features equal zero” means before giving it a practical interpretation.

### 2. Interpret Positive and Negative Weights

For a feature $x_{j}$, its contribution to the linear score is:

$$
w_{j}x_{j}
$$

If $x_{j}$ increases while the other features stay the same:

- A **positive** $w_{j}$ raises $z$, so the estimated probability of class $1$ increases.

- A **negative** $w_{j}$ lowers $z$, so the estimated probability of class $1$ decreases.

- A **zero** $w_{j}$ makes no change to $z$.

The phrase **“while the other features stay the same”** matters. A weight describes the feature’s relationship with the prediction *within the fitted model*, alongside the other features included in that model.

A positive weight does not mean the feature guarantees class $1$. The final probability depends on the bias and **all** feature contributions added together.

### 3. Follow the Contributions for One Observation

Suppose a fitted model has:

$$
w_{0} = - 1,\ w_{1} = 0.8,\ w_{2} = - 0.4
$$

For an observation with $x_{1} = 1$ and $x_{2} = 1$, each part contributes to the score:

| **Part**        | **Calculation**           | **Contribution to** $z$ |
|-----------------|---------------------------|-------------------------|
| Bias            | $w_{0}$                 | $- 1.0$               |
| Feature 1       | $w_{1}x_{1} = 0.8(1)$   | $+ 0.8$               |
| Feature 2       | $w_{2}x_{2} = - 0.4(1)$ | $- 0.4$               |
| **Total score** | $z = - 1 + 0.8 - 0.4$   | $- 0.6$               |

The estimated probability is then:

$$
\widehat{p} = \sigma( - 0.6) = \frac{1}{1 + e^{0.6}} \approx 0.354
$$

Feature 1 pushes the score upward, while Feature 2 pushes it downward. The combined score is still negative, producing a probability below $0.5$. This is why reading one weight in isolation cannot tell us the complete prediction for an observation.

### 4. Interpret One-Hot Encoded Categories

A one-hot encoded category has a value of either $0$ or $1$. Its contribution is:

$$
w_{j}x_{j} = \begin{cases}
w_{j} & \text{if~the~category~is~present} \\
0 & \text{if~the~category~is~absent}
\end{cases}
$$

Suppose a category has a weight of $+ 0.8$. When that category is present, it adds $0.8$ to the linear score. When it is absent, its column adds nothing.

The **comparison behind a category weight depends on the encoding**. If one category is omitted as a reference, the weights of the included categories can be read relative to that reference, with other features held fixed. If the encoding keeps every category, the weights and bias should be interpreted as parts of the complete model calculation rather than as simple stand-alone comparisons with a single reference category.

### 5. Why a Weight Is Not a Fixed Probability Change

The sigmoid function is curved. The same increase in the linear score can produce different changes in probability, depending on where the score started.

For example, adding $1$ to the score gives:

| **Starting score** | **Probability before**         | **Probability after adding** $1$ |
|--------------------|--------------------------------|----------------------------------|
| $- 3$            | $\sigma( - 3) \approx 0.047$ | $\sigma( - 2) \approx 0.119$   |
| $0$              | $\sigma(0) = 0.500$          | $\sigma(1) \approx 0.731$      |
| $3$              | $\sigma(3) \approx 0.953$    | $\sigma(4) \approx 0.982$      |

In every row, the score increases by exactly $1$. The probability change is different. The sigmoid curve is most sensitive around its middle and flattens as it approaches $0$ or $1$.

This is the central caution in interpreting Logistic Regression: **weights describe changes to the linear score, not fixed percentage-point changes to probability.**

### 6. Interpret Weights Through Odds

There is another precise way to describe a weight. If $\widehat{p}$ is the estimated probability of class $1$, its **odds** are:

$$
\text{Odds} = \frac{\widehat{p}}{1 - \widehat{p}}
$$

Logistic Regression models the **log of the odds** as a linear combination of features:

$$
\log\left( \frac{\widehat{p}}{1 - \widehat{p}} \right) = w_{0} + w_{1}x_{1} + \cdots + w_{m}x_{m}
$$

When a numerical feature increases by one unit and the other features remain fixed, its coefficient $w_{j}$ is the change in **log-odds**. Equivalently, the odds are multiplied by:

$$
e^{w_{j}}
$$

Thus:

- $w_{j} > 0$ means $e^{w_{j}} > 1$: the odds of class $1$ increase.

- $w_{j} < 0$ means $e^{w_{j}} < 1$: the odds decrease.

- $w_{j} = 0$ means $e^{w_{j}} = 1$: the odds do not change.

**Odds and probability are different quantities.** An odds multiplier should not be described as the same percentage change in probability.

### 7. Compare Weights Carefully

The size of a coefficient alone does not always tell us which feature is most important.

A weight for a numerical feature describes a **one-unit change**. If two features use very different units or scales, comparing their raw weights can be misleading. Features can also be related to one another, which may affect how their fitted weights are distributed.

Finally, a fitted weight describes an association used by the model. It does not prove that changing the feature would *cause* a change in the real-world outcome.

### Putting the Interpretation Together

For any observation, read the model in this order:

$$
\underbrace{w_0}_{\text{starting score}} + \underbrace{w_1x_1 + \cdots + w_mx_m}_{\text{feature contributions}} = \underbrace{z}_{\text{linear score}}
$$

$$
z \xrightarrow{\text{sigmoid}} \underbrace{\widehat{p}}_{\text{estimated probability of class }1}
$$

The bias and weights explain **how the model constructs its score**. The sigmoid function turns that score into a probability. To understand one prediction, examine the contributions together; to understand one weight, keep its feature’s scale, encoding, and the other features in mind.

## Using a Logistic Regression Model

A trained Logistic Regression model has learned a bias and a weight for each feature. The next step is to use those learned values to make predictions, assess the final model, and apply it to a new observation.

The process has two parts that must stay together:

1.  **Prepare the features** in the same way as during training.

2.  **Apply the trained model** to obtain an estimated probability.

A model cannot interpret a raw category label if it was trained on one-hot encoded columns. It expects the same feature structure it learned from.

### 1. Keep the Feature Structure Consistent

During training, the original data was transformed into a feature matrix $X_{\text{train}}$. This matrix contains numerical features and binary columns created from categorical features.

A new observation must go through the **same preparation steps**. The resulting row must have the same feature columns, in the same order, as the training matrix:

$$
x_{\text{new}} = \left\lbrack x_{1},x_{2},\ldots,x_{m} \right\rbrack
$$

Each position must retain its original meaning. If position 2 represented one category during training, it must represent that same category when predicting a new observation.

This also applies to any other transformations, such as filling missing values. Their rules should be learned or defined from the training data and then applied consistently to validation, test, and new data. We must not use the new observation’s unknown target to prepare its features.

### 2. Calculate a Probability for a New Observation

Once the new feature row is ready, the model combines it with its learned weights:

$$
z_{\text{new}} = w_{0} + x_{\text{new}}^{T}w
$$

It then applies the sigmoid function:

$$
{\widehat{p}}_{\text{new}} = \frac{1}{1 + e^{- z_{\text{new}}}}
$$

The output ${\widehat{p}}_{\text{new}}$ is the **estimated probability of class** $1$. A larger value indicates that the model assigns greater likelihood to class $1$ for this observation.

This calculation does **not** change the model’s weights. Training has already finished; the model is now using what it learned.

### 3. Decide Whether a Class Label Is Needed

An estimated probability and a class label serve different purposes.

If the aim is to **rank observations by estimated risk**, the probabilities themselves may be the useful output. If a decision requires a binary result, we apply a threshold $t$:

$$
{\widehat{y}}_{\text{new}} = \begin{cases}
1 & \text{if~}{\widehat{p}}_{\text{new}} \geq t \\
0 & \text{if~}{\widehat{p}}_{\text{new}} < t
\end{cases}
$$

The threshold is a rule for acting on the probability. It is separate from the model’s learned weights.

For example, lowering the threshold will generally assign class $1$ to more observations; raising it will assign class $1$ to fewer. This changes the pattern of correct and incorrect class predictions. The choice should therefore depend on what each type of mistake means for the task, rather than being treated as a property of Logistic Regression itself.

### 4. Assess the Model on the Test Set

The Validation set was used while developing and selecting the model. Once those decisions are complete, the **Test set** provides a final assessment using observations kept separate from that development process.

The test data must receive the same feature preparation as the training data. The trained model then generates predictions for $X_{\text{test}}$, which can be compared with the known targets $y_{\text{test}}$.

This gives an estimate of how the selected approach performs on previously unseen observations. A difference between validation and test results can happen because the two sets contain different observations. A large difference is a reason to examine the data and the development process more closely.

After inspecting the test result, repeatedly changing the model to improve that same result would turn the Test set into another development set. It would no longer provide the independent final check we intended.

### 5. Train the Final Model on Available Development Data

During model development, the Train set was used to learn weights and the Validation set was used to make choices. Once those choices are fixed, the Train and Validation sets can be combined to train a **final model**:

$$
X_{\text{full~train}} = \begin{bmatrix}
X_{\text{train}} \\
X_{\text{val}}
\end{bmatrix},\ y_{\text{full~train}} = \begin{bmatrix}
y_{\text{train}} \\
y_{\text{val}}
\end{bmatrix}
$$

The final model learns from more labeled observations. Its weights may differ from those of the earlier model trained on the Train set alone.

Any preparation step that learns from data must also be fitted again on the combined development data and then applied to the Test set and future observations. The **Test set remains separate**: it is used to assess the final model, not to teach it.

### 6. Apply the Complete Prediction Process

Using the model for a new observation requires more than the Logistic Regression equation. The complete sequence is:

$$
\text{Raw~observation} \rightarrow \text{Prepared~features} \rightarrow \text{Linear~score} \rightarrow \text{Estimated~probability} \rightarrow \text{Decision,~if~needed}
$$

Each stage has a distinct role:

| **Stage**                    | **Purpose**                                                      |
|------------------------------|------------------------------------------------------------------|
| Prepare features             | Give the model the same feature definitions used during training |
| Calculate the linear score   | Combine feature values with the learned weights and bias         |
| Apply sigmoid                | Convert the score into an estimated probability of class $1$     |
| Apply a threshold, if needed | Turn the probability into a class prediction                     |

The preparation rules and the fitted model therefore belong to **one prediction process**. If the preparation changes, the meaning of the model’s inputs changes as well.

### 7. Interpret the Output with Care

A probability is a model estimate based on patterns in the available data. It is useful for comparing observations and supporting decisions, but it does not explain *why an individual outcome will certainly happen*.

Likewise, a class prediction of $1$ means the estimated probability met the chosen threshold. It does not mean the actual outcome is guaranteed to be $1$.

The practical question is whether these predictions help with the original objective. That requires checking performance on unseen data and considering the consequences of assigning an observation to the wrong class. The course’s classification module builds a model that estimates customer-churn risk from structured data; more detailed evaluation measures follow in the evaluation module.

### Connecting the Full Workflow

$$
\begin{aligned}
 & \text{Prepare~and~explore~data} \\
 & \rightarrow \text{Split~Train,~Validation,~and~Test} \\
 & \rightarrow \text{Encode~features} \\
 & \rightarrow \text{Train~and~select~the~model} \\
 & \rightarrow \text{Train~the~final~model} \\
 & \rightarrow \text{Evaluate~on~Test} \\
 & \rightarrow \text{Prepare~and~predict~new~observations}
\end{aligned}
$$

The central principle is **consistency**: a new observation must be represented in the same way as the observations used to train the model. Only then can the learned weights produce a meaningful probability.
