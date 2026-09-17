## What is Machine Learning?

Machine Learning is a process of extracting patterns from data. The data usually consists of two main parts: **features** and the **target**. Features are the information or characteristics that describe an object, while the target is the value or outcome that we want to predict.

The result of the Machine Learning process is a **model**. The model learns patterns from the available data and uses those patterns to make predictions.

When we want to use the model with new data, we provide the features of a new object as input. The model then processes those features and returns a prediction of the target.

```math
\text{Features} + \text{Target}
\rightarrow
\text{Training}
\rightarrow
\text{Model}
\rightarrow
\text{Prediction}
```

## Features and Target

To understand the basic concept of Machine Learning, we need to distinguish three important terms: **Features, Target, and Model**.

### 1. Features

**Features** are the information or characteristics that we know about the object or data we are studying. In simple terms, features are **what we know about the data**.

Features can appear in different forms, such as numbers, text, or more complex structured information such as location data.

For example, if we want to predict house prices, the features may include the size of the house, number of rooms, age of the house, and location.

### 2. Target

**Target** is the value or property that we want the Model to predict for data that it has not seen before. In simple terms, the target is **what we want the Model to predict**.

In some courses or learning materials, the term **Label** may be used instead of Target.

During the Training process, we use a **Labeled Dataset**, which is a dataset where both the Features and the Target are already known. This allows the Model to learn the relationship between the two.

For example, if we want to predict house prices, the Target is the **house price**.

### 3. Model

A **Model** is the result of the Training process. It contains the **patterns** and **relationships** learned from the Features and Target in the Training data.

During Training, the Machine Learning system analyzes the data to discover patterns and relationships between the Features and the Target.

The process can be summarized as:

$$
\text{Features} + \text{Target}
\rightarrow
\text{Training}
\rightarrow
\text{Model}
$$

Once the Model has been created, we can use it with new data that contains Features but does not yet have a Target. The Model then uses the patterns it has learned to generate a Prediction.

$$
\text{New Features}
\rightarrow
\text{Model}
\rightarrow
\text{Prediction}
$$

## Train a Model

**Model Training** means taking the Features and the Target and giving them to a Machine Learning Algorithm so that the Algorithm can learn from the data and create a Model.

The Training process can be represented as:

$$
X + y
\rightarrow
\text{Machine Learning Algorithm}
\rightarrow
\text{Model}
$$

where:

- \(X\) represents the Features
- \(y\) represents the Target

The Model stores the **patterns** and **relationships** that the Algorithm learns from the Training data.

In simple terms, the **Algorithm** is the process used to learn from the data, while the **Model** is the result produced after the Training process is complete.

$$
\text{Algorithm} + \text{Training Data}
\rightarrow
\text{Trained Model}
$$

A trained Model can be considered an **artifact** that can be saved and reused later. This means that we do not need to train the Model again every time we want to make a Prediction. Instead, we can load the trained Model and use it with new data.

---

## Making Predictions

Once we have a **Trained Model**, we can use it to make Predictions on new data where the Target is still unknown.

During Prediction, we provide only the Features of the new data to the Model. We do not provide the Target because the Target is the value that we want the Model to predict.

The Prediction process can be represented as:

$$
X_{\text{new}}
\rightarrow
\text{Trained Model}
\rightarrow
\hat{y}
$$

where:

$$ 
\(X_{\text{new}}\) 
$$

represents the Features of new data

$$
\(\hat{y}\)
$$

represents the Predicted Target

A Model may not predict every Sample with 100% accuracy because real-world data can vary and may contain factors that the Model did not learn during Training.

However, if the Model performs well, its Predictions should generally be close to the actual values, even though some individual Predictions may be slightly higher or lower than the true values.

The complete Machine Learning process can therefore be summarized as:

$$
X, y
\rightarrow
\text{Training}
\rightarrow
\text{Trained Model}
\rightarrow
X_{\text{new}}
\rightarrow
\hat{y}
$$

# Machine Learning vs Rule-Based Systems

---

## The Rule-Based Approach

The **Rule-Based Approach** is a way of building systems in which humans analyze data to identify patterns, convert those patterns into rules or conditions, and then implement those rules in the system to support decision-making.

The basic idea can be represented as:

$$
\text{Data}
\rightarrow
\text{Human Analysis}
\rightarrow
\text{Pattern}
\rightarrow
\text{Rule}
\rightarrow
\text{Decision}
$$

Rules are often written in the form of conditions such as:

$$
\text{IF condition}
\rightarrow
\text{THEN action}
$$

When data patterns or situations change, the existing rules may no longer be able to handle new cases correctly. As a result, the problem needs to be analyzed again and the rules must be added, modified, or updated.

This process may repeat over time:

$$
\text{Analyze}
\rightarrow
\text{Create Rules}
\rightarrow
\text{Deploy}
\rightarrow
\text{New Pattern}
\rightarrow
\text{Update Rules}
$$

As the number of rules increases, the system may become more complex. Rules can overlap, interact with each other, or create unexpected effects. Changing one rule may also affect other parts of the system, making the system more difficult to maintain.

Therefore, one important limitation of the Rule-Based Approach is that the rules must be created and updated by humans continuously, especially when the patterns in the data change.

When a problem becomes more complex and changes frequently, **Machine Learning** can be used as another approach. Instead of manually defining all rules, a Machine Learning Algorithm can learn patterns directly from data.

---

## The Machine Learning Approach

The **Machine Learning Approach** solves problems by allowing an Algorithm to learn patterns from data and create a Model that can be used to make Predictions on new data.

Instead of manually writing rules for every possible condition, the Model learns patterns from examples in the data.

The overall process can be divided into three main steps:

$$
\text{Get Data}
\rightarrow
\text{Define and Calculate Features}
\rightarrow
\text{Train and Use the Model}
$$

### Step 1: Get Data

The first step is to collect data that is relevant to the problem.

For **Supervised Machine Learning**, the data should include a **Label** or **Target** that represents the correct answer for each Sample. This allows the Algorithm to learn the relationship between the input data and the Target.

After collecting the data, we have:

$$
\text{Sample} + \text{Target}
$$

This data can then be prepared for Model Training.

### Step 2: Define and Calculate Features

Each Sample needs to be described using **Features**, which are characteristics that the Model can use to learn patterns.

Features are usually transformed into numerical values so that they can be processed by a Machine Learning Algorithm.

Some Features may have only two possible values, such as **True** or **False**. These are called **Binary Features** and can be encoded as:

$$
\text{True} = 1,
\qquad
\text{False} = 0
$$

The Features of each Sample are combined into a **Feature Vector**. When the Feature Vectors from all Samples are combined, they form a **Feature Matrix**.

We usually represent them as:

$$
X = \text{Feature Matrix}
$$

and

$$
y = \text{Target}
$$

Some Features may also come from knowledge gained while developing a Rule-Based System. Therefore, starting with a Rule-Based Approach can sometimes help us understand the problem and identify useful Features for Machine Learning.

### Step 3: Train and Use the Model

The Features \(X\) and Target \(y\) are given to a Machine Learning Algorithm so that the Algorithm can learn patterns from the data and create a Model.

This process is called **Training** or **Fitting**.

$$
X, y
\rightarrow
\text{Machine Learning Algorithm}
\rightarrow
\text{Trained Model}
$$

After the Model has been trained, we can provide the Features of new data, \(X_{\text{new}}\), to generate a Prediction.

$$
X_{\text{new}}
\rightarrow
\text{Trained Model}
\rightarrow
\text{Prediction}
$$

For a **Binary Classification** problem, the Model may return a Probability between 0 and 1.

For example:

$$
P(y = 1) = 0.78
$$

A **Threshold** can then be used to convert the Probability into a final decision. For example, if the Threshold is set to 0.5:

$$
\hat{y} =
\begin{cases}
1, & \text{if } P(y=1) \geq 0.5 \\
0, & \text{if } P(y=1) < 0.5
\end{cases}
$$

Therefore, the overall Machine Learning Approach can be summarized as:

$$
\text{Data}
\rightarrow
\text{Features and Target}
\rightarrow
\text{Training}
\rightarrow
\text{Trained Model}
\rightarrow
\text{New Features}
\rightarrow
\text{Prediction}
$$

## Rules vs Machine Learning

The main difference between a **Rule-Based System** and **Machine Learning** is who creates the rules or patterns used for decision-making.

In a **Rule-Based System**, humans analyze the data, identify rules, and write those rules directly into the code. The system then uses the data together with the predefined rules to produce an outcome.

$$
\text{Data} + \text{Rules / Code}
\rightarrow
\text{Software}
\rightarrow
\text{Outcome}
$$

The rules are therefore **hard-coded**. If the data or underlying patterns change, humans need to manually update, modify, or add new rules.

As the number of rules increases, the system can become more complex, and maintaining the rules may become more difficult.

In **Machine Learning**, the process is different. Data and known outcomes, or **Targets**, are used during Training so that the Algorithm can learn patterns from the data and create a Model.

$$
X + y
\rightarrow
\text{Machine Learning Algorithm}
\rightarrow
\text{Model}
$$

where:

- \(X\) represents the Features
- \(y\) represents the known Target

Once the Model has been trained, it can be used with new data where the outcome is still unknown. The new Features are provided to the trained Model, which then generates a Prediction.

$$
X_{\text{new}} + \text{Trained Model}
\rightarrow
\hat{y}
$$

where:

$$
\(X_{\text{new}}\)
$$

represents the Features of new data

$$
\(\hat{y}\) 
$$

represents the Predicted Target

In simple terms, a Rule-Based System depends on humans to define the rules, while Machine Learning allows the Algorithm to learn patterns from data and use those learned patterns to make Predictions.

# Supervised Machine Learning

## What is Supervised Machine Learning?

**Supervised Machine Learning (SML)** is a type of Machine Learning in which the Model is trained using examples where both the **Features** and the **Target** are already known.

The word **Supervised** means that the learning process is guided by labeled examples.

## Why "Supervised"?

We can think of the process as teaching the Model.

We provide examples to the Model together with the correct Target values so that the Model can learn the relationship between the Features and the Target.

From these examples, the Model tries to learn the patterns in the data. The goal is not only to memorize the Training data, but also to **generalize** what it has learned to new data that it has never seen before.

The basic idea can be summarized as:

$$
\text{Examples with Known Targets}
\rightarrow
\text{Learn Patterns}
\rightarrow
\text{Generalize to New Data}
$$

## The Feature Matrix and the Target

In Supervised Machine Learning, the Model is trained using **Features** and **Targets**. This means that we provide both the input data and the correct output that we want the Model to learn.

The **Feature Matrix** is usually represented by \(X\).

In \(X\):

- Rows represent **Observations, Samples, or Objects**
- Columns represent **Features**

The Feature Matrix contains the information that we use as input for the Model.

The **Target Variable** is usually represented by \(y\).

The Target Variable is a vector that contains the Target values that we want the Model to learn and predict. Each row in \(X\) has a corresponding Target value in \(y\).

## The Model as a Function

A Model can be viewed as a function, usually written as \(g\), that takes the Features \(X\) as input and produces Predictions that should be as close as possible to the Target \(y\).

This can be written as:

$$
g(X) \approx y
$$

After the Model makes a Prediction, the predicted value is usually written as:

$$
\hat{y} = g(X)
$$

where:

$$
\(X\) 
$$

represents the Features

$$
\(g\)
$$

represents the Model or learned function

$$
\(y\)
$$

represents the actual Target

$$
\(\hat{y}\) 
$$

represents the Predicted Target

During Training, the Model learns from the relationship between \(X\) and \(y\) in order to find an appropriate function \(g\) for making Predictions.

Therefore, Training can be viewed as the process of finding a function that connects the Features to the Target.

$$
X + y
\rightarrow
\text{Training}
\rightarrow
g
$$

or:

$$
\text{Features} + \text{Targets}
\rightarrow
\text{Training}
\rightarrow
\text{Model}
$$

Once the Model has been trained, it can be used with new Features to generate Predictions.

$$
X_{\text{new}}
\rightarrow
g
\rightarrow
\hat{y}
$$

# Cross-Industry Standard Process for Data Mining: CRISP-DM

## Why a Process?

When working on a Machine Learning Project, we usually need to understand the problem, collect the data, train the Model, and use the Model.

A methodology such as **CRISP-DM** helps organize these activities into a structured process. It helps us understand what should be done at each stage and how the different stages connect to each other.

At a high level, the workflow can be summarized as:

- **Understand the Problem** → Understand the problem and what we are trying to solve
- **Collect the Data** → Gather the data required for solving the problem
- **Train the Model** → Use the data to train a Model so that it can learn patterns
- **Use the Model** → Apply the trained Model to new data to generate Predictions

---

## CRISP-DM

**CRISP-DM** stands for **Cross-Industry Standard Process for Data Mining**.

It is an **Open Standard Process Model** that describes the process used for Data Mining and Analytics. This concept is also commonly applied to managing Machine Learning Projects.

The concept of CRISP-DM began in 1996, and in 1997 it became a project under the European Union's **ESPRIT funding initiative**. The project was led by five organizations: **Integral Solutions Ltd (ISL), Teradata, Daimler AG, NCR Corporation, and OHRA**, an insurance company.

CRISP-DM divides the working process into six main stages:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

## 1. Business Understanding

**Business Understanding** *(An important question is: Do we need Machine Learning for the Project? The goal of the Project has to be measurable.)*

The goal of this stage is to understand and identify the problem that we want to solve. We also need to consider how important the problem is and whether it is worth investing time and resources to solve it.

Another important question is whether we actually need to use **Machine Learning**. Some problems may be solved with a simpler **Rule-Based System** or **Heuristic**, without needing to invest time and resources in building a Machine Learning System.

If we decide to use Machine Learning, we should define a **Measurable Goal** so that we can later evaluate whether the Project has been successful.

---

## 2. Data Understanding

**Data Understanding** *(Analyze available Data Sources and decide whether more Data is required.)*

Machine Learning requires **Data**. If there is no Data, Machine Learning cannot be performed.

In this stage, we need to understand what Data we currently have, whether the available Data is sufficient, and how we can obtain any Data that is still missing. We may obtain additional Data from external sources or begin collecting the Data ourselves.

The questions that we need to consider include the following:

### 2.1 Does the Data Source Actually Work?

We need to check whether the Data Source is actually working as expected.

For example, when a user clicks a **Spam** button, does the system track and record that action every time?

### 2.2 Is the Data Reliable?

We need to determine whether the Data is reliable.

A user may mark a message as Spam even though the message is not actually Spam.

If we use this Data to train a Model, the Model may learn this behavior and begin classifying non-Spam messages as Spam as well.

Therefore, we often need to manually inspect and analyze the Data to determine whether its quality is sufficient.

### 2.3 Is There Enough Data?

We also need to consider whether there is enough Data.

If we only have 10 Records, there may not be much that we can do with them.

One possible outcome of this stage may be:

> We are not ready yet.

In this case, we may need to collect several thousand additional Records before continuing.

During the Data Understanding stage, we may discover new information about the problem. This may change the understanding that we developed during **Step 1: Business Understanding**.

This is normal because the stages of CRISP-DM are connected to each other.

$$
\text{Business Understanding}
\leftrightarrow
\text{Data Understanding}
$$

We can return to Business Understanding, revise our understanding, and then come back to Data Understanding again.

---

## 3. Data Preparation

**Data Preparation** *(Clean the Data, remove noise by applying pipelines, and convert the Data into a tabular format so that it can be used in Machine Learning.)*

Data Preparation is the stage where we prepare the Data so that it can be used by a Machine Learning Algorithm.

At this stage, we already know that we have Data of sufficient quality and quantity.

The next step is to transform and organize the Data so that it is ready for Machine Learning.

In general, Data Preparation includes the following activities:

### 3.1 Feature Extraction

Extract or create the Features that we need from the Raw Data.

### 3.2 Data Cleaning

Clean the Data and remove Noise.

This may include correcting or removing incorrect information in the Data.

### 3.3 Building Data Pipelines

Build a sequence of steps that receives Raw Data, applies different transformations, and produces clean Data that is ready for use.

### 3.4 Converting to Tabular Format

Convert the Data into a table format so that it can be provided to a Machine Learning Model.

During Data Preparation, the Data that has already been collected and reviewed is organized and transformed into a format that is suitable for Machine Learning.

A **Data Pipeline** takes Raw Data through different processes such as Data Cleaning, Data Transformation, and conversion into a **Tabular Format**.

From the prepared Data, we select or create the Features that the Model will use for learning.

The Features of each Sample are used to create a **Feature Vector**.

When the Feature Vectors of all Samples are combined, we obtain the **Feature Matrix \(X\)**.

$$
X = \text{Feature Matrix}
$$

The value that we want the Model to learn or predict is stored in the **Target Vector \(y\)**.

$$
y = \text{Target Vector}
$$

Therefore, the main goal of Data Preparation is to transform Data that is not yet ready into the **Feature Matrix \(X\)** and **Target \(y\)** that can be used to train a Machine Learning Model.

---

## 4. Modeling

**Modeling** *(Train different Models and choose the best one. Based on the results of this step, we may decide whether new Features need to be added or Data issues need to be fixed.)*

Modeling is the stage where the Data prepared during Data Preparation is used to train a Machine Learning Model. This is the stage where the learning process of Machine Learning actually takes place.

In this stage, we experiment with different Models or Algorithms and compare their performance in order to select the most suitable Model, such as Logistic Regression, Decision Tree, Neural Network, and other Models.

$$
X, y
\rightarrow
\text{Train Model}
\rightarrow
\text{Evaluate / Compare}
\rightarrow
\text{Select Base Model}
$$

During Modeling, we may discover that the Features are not sufficient or that there are problems with the quality or structure of the Data.

In that case, we can return to the Data Preparation stage, improve the Data or Features, and then train the Model again.

$$
\text{Data Preparation}
\leftrightarrow
\text{Modeling}
$$

Therefore, Modeling is not simply the process of training a Model once. It is a process of experimenting, improving, and comparing Models in order to identify the Model that gives the most suitable results.

A more systematic method for selecting the **Best Model** will be discussed later under **Model Selection**.

---

## 5. Evaluation

**Evaluation** *(Measure how well the Model is performing and determine whether it solves the Business Problem.)*

Evaluation is the stage where we assess the selected Model to determine how well it performs and whether it can achieve the goals defined during Business Understanding.

At this stage, we return to the Business Goals and Metrics defined at the beginning and compare them with the results produced by the Model.

Important questions include:

- Can the Model achieve the defined goals?
- Have the Metrics improved?
- If the results are below the original target, are they still acceptable?

Therefore, Evaluation does not only assess **Model Performance**, but also considers whether the Model can actually solve the Project problem.

$$
\text{Model Performance}
\rightarrow
\text{Compare with Business Goals}
\rightarrow
\text{Accept / Improve}
$$

If the results do not meet expectations, we can analyze what happened through a **Retrospective**.

The goals that were set may have been too ambitious, or the Model and Data may still need improvement.

We can then begin a new Iteration and use what we have learned to improve the Project.

If we discover that the Project cannot realistically achieve the required goals, we may also decide to stop the Project.

---

## 6. Deployment

**Deployment** *(Roll out the Model to Production for all users. Evaluation and Deployment often happen together through Online Evaluation.)*

Deployment is the stage where the Model that has passed Evaluation is moved into a **Production Environment** so that it can work with real Data and real users.

$$
\text{Trained Model}
\rightarrow
\text{Production}
\rightarrow
\text{Real-World Usage}
$$

In practice, Evaluation and Deployment often happen together.

We can first deploy the Model to a small group of users and observe how it performs. This process is called **Online Evaluation**.

If the Model performs well, we can gradually expand its use to all users.

$$
\text{Deploy to Small Group}
\rightarrow
\text{Online Evaluation}
\rightarrow
\text{Rollout}
$$

When we reach the Deployment stage, the focus begins to shift from Machine Learning toward **Engineering**.

The Model becomes part of a real system that needs to work continuously.

Therefore, we need to make sure that the system is:

- **Monitored**
- **Maintainable**
- **Reliable**

After Deployment, the system must continue to operate effectively in a real-world environment.

---

## Iterate!

**Iterate** means taking what we learn from developing and using the Model and using that knowledge to continuously improve the Project.

The Project does not simply end after Deployment.

An important idea is to **Start Simple**. In the first Iteration, we should create a simple Model or Solution and move through the CRISP-DM stages quickly, from understanding the problem, preparing the Data, building the Model, evaluating the results, to deploying the Model.

After that, we use Feedback and what we learn from the actual results to improve the next Iteration.

$$
\text{Start Simple}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Deploy}
\rightarrow
\text{Learn}
\rightarrow
\text{Improve}
\rightarrow
\text{Repeat}
$$

After learning from the first Iteration, we can return to Business Understanding and develop a more complex or more effective Solution or Model.

Completing two or three quick Iterations can reduce the risk of spending too much time building a complex Model from the beginning.

It also helps us quickly determine whether the Solution we are developing is actually useful and can solve the problem.

Therefore, the main idea of Iterate is:

$$
\text{Start Simple}
\rightarrow
\text{Experiment}
\rightarrow
\text{Learn}
\rightarrow
\text{Improve}
\rightarrow
\text{Repeat}
$$

We do not need to build a perfect Model in the first Iteration.

This is why, in the real world, a Machine Learning Project does not end when we obtain a good Model.

Instead, it is a continuous process of **learning, improving, evaluating, and adapting based on real-world usage**.

# Model Selection Process

In this section, we will take a deeper look at the **Modeling** stage of CRISP-DM. We will learn how to experiment with multiple Models and select the best one.

We will also understand why using only one Validation Set may give us a misleading impression of Model Performance, and how splitting the Data into **Training, Validation, and Test Sets** helps reduce this problem.

---

## The Modeling Step

From the previous section, we learned that **Modeling** is the stage where Machine Learning actually takes place.

Once the Data has been prepared, we can experiment with different Models such as **Logistic Regression, Decision Tree, Neural Network**, and other Models.

At this stage, we do not need to worry yet about how each Model works because we will study them in more detail later.

Some Models may work well for our problem, while others may not perform well.

Therefore, the goal is to experiment with multiple Models and select the Model that performs best.

$$
\text{Prepare Data}
\rightarrow
\text{Try Different Models}
\rightarrow
\text{Select Best Model}
$$

---

## Simulating Future Data with a Validation Set

Consider how we will use the Model in the real world.

We use the **Feature Matrix \(X\)** and **Target \(y\)** to train a Model \(g\). After the Model is deployed, it must apply what it has learned to new Data that it has never seen before and generate Predictions.

Therefore, when evaluating the Model, we want to simulate this situation so that we can measure how well the Model performs on **Unseen Data**, or Data that was not used during Training.

However, we cannot use future Data because it does not exist yet.

A practical approach is to separate part of the Dataset, for example:

- **80% = Training Set**
- **20% = Validation Set**

We train the Model using only the 80% Training Set and keep the remaining 20% separate so that the Model does not see it during Training.

This separated portion is called the **Validation Set** and acts as a simulation of new future Data.

From the Training Set, we have:

$$
X, y
$$

which are used to train the Model:

$$
g
$$

The Validation Set contains:

$$
X_v, y_v
$$

which the Model has never seen during Training.

We then use the Model \(g\) to make Predictions on \(X_v\).

$$
X_v
\rightarrow
g
\rightarrow
\hat{y}_v
$$

If the Model produces a Probability, we convert the Probability into a final decision based on a defined criterion.

We then compare the Model's Predictions:

$$
\hat{y}_v
$$

with the actual values:

$$
y_v
$$

After that, we can calculate how accurately the Model performs using an evaluation metric such as **Accuracy**.

---

## Comparing Models

We can repeat the same process for every Candidate Model and compare the Performance of each Model.

For example:

- Logistic Regression → Accuracy 66%
- Decision Tree → Accuracy 60%
- Random Forest → Accuracy 67%
- Neural Network → Accuracy 80%

These numbers are hypothetical and are used only to explain the concept.

In this example, the Neural Network has the highest Accuracy and would therefore be selected as the Best Model.

The main idea is:

$$
\text{Train Model}
\rightarrow
\text{Evaluate on Validation Set}
\rightarrow
\text{Compare Performance}
\rightarrow
\text{Select the Best Model}
$$

---

## The Multiple Comparisons Problem

However, selecting a Model in this way can create a problem.

If we test many Models repeatedly on the same Validation Set, one of the Models may achieve very good results simply because of chance.

When many Models are evaluated using the same Validation Set, it is possible to find a Model that matches the Validation Data extremely well and appears to have very high Accuracy, even though the result may be partly due to coincidence.

The same problem can happen with real Machine Learning Models.

A Model may perform exceptionally well on one particular Validation Set simply because of chance, but if we evaluate it on another Dataset, its Performance may be significantly different.

In Statistics, this problem is known as the **Multiple Comparisons Problem (MCP)**.

When we perform many comparisons, such as testing a large number of Models on the same Validation Set, the probability increases that one Model will achieve unusually good results by chance.

This is especially relevant because many Machine Learning methods have a **Probabilistic** nature.

---

## Train, Validation, Test

To help reduce this problem, instead of dividing the Dataset into only two parts, we divide it into three separate Sets.

For example:

- **60% Training Set** → used to train the Model
- **20% Validation Set** → used to evaluate and compare Models
- **20% Test Set** → used to evaluate the Best Model one final time

The **60/20/20** split is not a fixed rule.

Other proportions can be used depending on the Dataset and the problem.

The Dataset is therefore divided into three non-overlapping Subsets:

$$
\text{Training Set}
\quad
\text{Validation Set}
\quad
\text{Test Set}
$$

During Model Selection, we keep the **Test Set** completely separate and do not use it.

We train Models using the Training Set and evaluate them using the Validation Set in order to select the Best Model.

Once the Best Model has been selected, we evaluate it again using the Test Set.

This allows us to check whether the Model performed well on the Validation Set because it genuinely generalizes well, rather than because of chance.

---

## Model Selection Process

The Model Selection process can be summarized in six steps:

1. **Split the Dataset** into Training, Validation, and Test Sets.
2. **Train the Model** using the Training Set.
3. **Evaluate the Model** using the Validation Set.
4. Repeat Steps 2–3 for all Models that we want to test.
5. **Select the Best Model**.
6. Evaluate the Best Model using the Test Set and check whether its Performance is similar to the Performance observed on the Validation Set.

The overall process can be represented as:

$$
\text{Dataset}
\rightarrow
\text{Train / Validation / Test}
\rightarrow
\text{Train Models}
\rightarrow
\text{Evaluate on Validation}
\rightarrow
\text{Select the Best Model}
\rightarrow
\text{Evaluate on Test Set}
$$

---

## Reusing the Validation Set

There is one additional consideration.

In the process above, the 20% Validation Set is used only for evaluating and comparing Models. It is not used during Training.

After we have selected the Best Model, we can combine:

$$
\text{Training Set} + \text{Validation Set}
$$

to create a larger Training Dataset.

If the original split was:

- 60% Training
- 20% Validation
- 20% Test

then combining the Training and Validation Sets gives us:

- 80% of the Dataset for final Training.

We then train the Model again using the same Model type and Settings that were selected during the Model Selection process.

Because the final Model is now trained using more Data, it may perform better.

Finally, we evaluate this newly trained Model on the Test Set again.

The final flow is:

$$
\text{Train + Validation}
\rightarrow
\text{Retrain Best Model}
\rightarrow
\text{Test}
$$

# Linear Algebra

Linear Algebra provides the mathematical foundation for working with **Vectors** and **Matrices**.

In Machine Learning, Linear Algebra is especially important because datasets are commonly represented as matrices, and many models perform calculations using vector and matrix operations.

In this section, we will cover:

- Vector Operations
- Vector × Vector Multiplication
- Matrix × Vector Multiplication
- Matrix × Matrix Multiplication
- Identity Matrix
- Inverse Matrix

These concepts will be used again in the **Regression** section to explain and build Linear Regression.

---

## Vector Operations

A **Vector** is an ordered collection of numbers.

In Linear Algebra, vectors are usually written as **column vectors**.

For example:

```math
u =
\begin{bmatrix}
2 \\
4 \\
5 \\
6
\end{bmatrix}
```

### Scalar Multiplication

Scalar Multiplication means multiplying a Vector by a single number called a **Scalar**.

The Scalar is multiplied by every element of the Vector.

For example:

```math
u =
\begin{bmatrix}
2 \\
4 \\
5 \\
6
\end{bmatrix}
```

If we multiply \(u\) by 2:

```math
2u =
2
\begin{bmatrix}
2 \\
4 \\
5 \\
6
\end{bmatrix}
=
\begin{bmatrix}
4 \\
8 \\
10 \\
12
\end{bmatrix}
```

Therefore:

```math
\text{Scalar} \times \text{Vector}
\rightarrow
\text{Vector}
```

---

### Vector Addition

Vector Addition is performed **element by element**.

The elements in the same positions are added together.

Let:

```math
u =
\begin{bmatrix}
2 \\
4 \\
5 \\
6
\end{bmatrix}
```

and:

```math
v =
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
```

Then:

```math
u + v =
\begin{bmatrix}
2 \\
4 \\
5 \\
6
\end{bmatrix}
+
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
=
\begin{bmatrix}
3 \\
4 \\
5 \\
8
\end{bmatrix}
```

Therefore:

```math
\text{Vector} + \text{Vector}
\rightarrow
\text{Vector}
```

The two Vectors must have the same number of elements.

---

## Multiplication

There are three important types of multiplication in this section:

1. Vector × Vector
2. Matrix × Vector
3. Matrix × Matrix

Each type is related to the **Dot Product**.

---

## Vector-Vector Multiplication

Vector-Vector Multiplication is called the **Dot Product** or sometimes the **Inner Product**.

The Dot Product is different from Element-wise Multiplication.

For the vectors:

```math
u =
\begin{bmatrix}
2 \\
4 \\
5 \\
6
\end{bmatrix}
```

and:

```math
v =
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
```

the Dot Product is written as:

```math
u^T v
```

where \(u^T\) means that the Column Vector \(u\) is transposed into a Row Vector.

Therefore:

```math
u^T v =
\begin{bmatrix}
2 & 4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
```

The elements in the same positions are multiplied, and then all results are added together:

```math
u^T v =
(2)(1)
+
(4)(0)
+
(5)(0)
+
(6)(2)
```

```math
u^T v =
2 + 0 + 0 + 12
=
14
```

Therefore:

```math
\text{Vector} \cdot \text{Vector}
\rightarrow
\text{Scalar}
```

The general formula for the Dot Product is:

```math
u^T v =
\sum_{i=1}^{n} u_i v_i
```

where:

- \(n\) is the number of elements in the Vectors
- \(u_i\) is the \(i\)-th element of \(u\)
- \(v_i\) is the \(i\)-th element of \(v\)

Both Vectors must have the same dimension.

---

## Matrix-Vector Multiplication

Matrix-Vector Multiplication can be understood as taking the **Dot Product of each Row of a Matrix with a Vector**.

Let:

```math
U =
\begin{bmatrix}
2 & 4 & 5 & 6 \\
1 & 2 & 1 & 2 \\
3 & 1 & 2 & 1
\end{bmatrix}
```

and:

```math
v =
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
```

The dimensions are:

```math
(3 \times 4)(4 \times 1)
=
(3 \times 1)
```

The number of Columns in \(U\) must be equal to the number of elements in \(v\).

Now calculate the Dot Product for each Row.

For the first Row:

```math
\begin{bmatrix}
2 & 4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
=
14
```

For the second Row:

```math
\begin{bmatrix}
1 & 2 & 1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
=
5
```

For the third Row:

```math
\begin{bmatrix}
3 & 1 & 2 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
=
5
```

Therefore:

```math
Uv =
\begin{bmatrix}
2 & 4 & 5 & 6 \\
1 & 2 & 1 & 2 \\
3 & 1 & 2 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
0 \\
2
\end{bmatrix}
=
\begin{bmatrix}
14 \\
5 \\
5
\end{bmatrix}
```

So:

```math
\text{Matrix} \times \text{Vector}
\rightarrow
\text{Vector}
```

A simple way to think about it is:

```math
\text{1 Row}
\rightarrow
\text{1 Dot Product}
\rightarrow
\text{1 Result}
```

---

## Matrix-Matrix Multiplication

Matrix-Matrix Multiplication extends the same idea.

Let:

```math
U =
\begin{bmatrix}
2 & 4 & 5 & 6 \\
1 & 2 & 1 & 2 \\
3 & 1 & 2 & 1
\end{bmatrix}
```

and:

```math
V =
\begin{bmatrix}
1 & 1 & 2 \\
0 & 0.5 & 1 \\
0 & 2 & 1 \\
2 & 1 & 0
\end{bmatrix}
```

The dimensions are:

```math
(3 \times 4)(4 \times 3)
=
(3 \times 3)
```

For Matrix Multiplication to be valid, the number of Columns in the first Matrix must be equal to the number of Rows in the second Matrix.

In general:

```math
(m \times n)(n \times p)
=
(m \times p)
```

Matrix-Matrix Multiplication can be understood as multiplying the first Matrix by each Column of the second Matrix.

Each entry in the resulting Matrix is obtained from:

```math
\text{Row of } U
\cdot
\text{Column of } V
```

The result is:

```math
UV =
\begin{bmatrix}
14 & 20 & 13 \\
5 & 6 & 5 \\
5 & 8.5 & 9
\end{bmatrix}
```

Therefore:

```math
\text{Matrix} \times \text{Matrix}
\rightarrow
\text{Matrix}
```

The three types of multiplication are closely related:

```math
\text{Vector} \times \text{Vector}
\rightarrow
\text{Scalar}
```

```math
\text{Matrix} \times \text{Vector}
\rightarrow
\text{Vector}
```

```math
\text{Matrix} \times \text{Matrix}
\rightarrow
\text{Matrix}
```

All three operations are based on the Dot Product.

---

## Identity Matrix

The **Identity Matrix** is usually represented by \(I\).

It is a Square Matrix where all elements on the **Main Diagonal** are 1 and all other elements are 0.

For example:

```math
I =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
```

The important property of the Identity Matrix is:

```math
AI = A
```

and:

```math
IA = A
```

as long as the Matrix dimensions are compatible.

The Identity Matrix plays a similar role in Matrix Multiplication to the number 1 in ordinary multiplication.

For example:

```math
5 \times 1 = 5
```

Similarly:

```math
A \times I = A
```

Therefore, the Identity Matrix can be thought of as the **multiplicative identity for matrices**.

---

## Inverse Matrix

The **Inverse Matrix** of a Matrix \(A\) is written as:

```math
A^{-1}
```

The defining property of an Inverse Matrix is:

```math
AA^{-1} = I
```

and:

```math
A^{-1}A = I
```

This means that multiplying a Matrix by its Inverse returns the Identity Matrix.

For example, let:

```math
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
```

Its Inverse is:

```math
A^{-1}
=
\begin{bmatrix}
-2 & 1 \\
1.5 & -0.5
\end{bmatrix}
```

Multiplying the two Matrices gives:

```math
AA^{-1}
=
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\begin{bmatrix}
-2 & 1 \\
1.5 & -0.5
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
=
I
```

The Inverse Matrix can be thought of as the Matrix equivalent of division.

For ordinary numbers:

```math
5 \times \frac{1}{5} = 1
```

For Matrices:

```math
A A^{-1} = I
```

However, not every Matrix has an Inverse.

A Matrix must be **Square** and **Invertible** in order for \(A^{-1}\) to exist.

Inverse Matrices are important in Linear Algebra and will appear again when we study **Linear Regression**.

In Machine Learning, a Matrix is often used to represent an entire dataset. It can be thought of as similar to a DataFrame, where each row represents one Sample or Observation and each column represents one Feature. A Matrix is therefore useful for storing many Samples and many Features together in a single structure.

A Vector, on the other hand, usually represents one set of values. Depending on the context, a Vector can represent the Features of a single Sample, the Target values, the Predictions produced by a Model, or the Parameters or Weights learned by the Model.

In simple terms, a Matrix usually represents multiple Samples and multiple Features, while a Vector usually represents one specific set of values.

For example, if \(X\) is the Feature Matrix, each row of \(X\) can be treated as a Feature Vector for one Sample. If the Model learns its Parameters as a Vector \(w\), the Matrix \(X\) can be multiplied by the Vector \(w\) to produce Predictions.

This is an important connection between Linear Algebra and Machine Learning, because data, Model Parameters, Targets, and Predictions can all be represented using Matrices and Vectors.
