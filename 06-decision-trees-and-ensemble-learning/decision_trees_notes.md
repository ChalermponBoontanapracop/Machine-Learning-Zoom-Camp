# Decision Trees and Ensemble Learning

Study notes on decision trees, learning algorithms, parameter tuning, random forests, gradient boosting, XGBoost, and final model selection.

## Contents

1. [Decision Trees](#1-decision-trees)
2. [Decision Tree Learning Algorithm](#2-decision-tree-learning-algorithm)
3. [Decision Tree Parameter Tuning](#3-decision-tree-parameter-tuning)
4. [Ensemble Learning and Random Forest](#4-ensemble-learning-and-random-forest)
5. [Gradient Boosting and XGBoost](#5-gradient-boosting-and-xgboost)
6. [XGBoost Parameter Tuning](#6-xgboost-parameter-tuning)
7. [Selecting the Best Model](#7-selecting-the-best-model)

## 1 Decision Trees

A decision tree automatically learns decision rules from data and uses them to make predictions. The rules take the form of `if/else` conditions. Each condition divides data into smaller groups, and splitting continues until a terminal node produces a prediction.

### Tree structure

```text
                         Root
                   Check a condition
                      /         \
                    True       False
                     /           \
              Decision node   Decision node
              Check condition Check condition
                /     \         /     \
              True   False    True   False
               /       \       /       \
             Leaf     Leaf    Leaf     Leaf
          Prediction at each terminal node
```

- **Root:** The starting node, where the first condition splits the data.
- **Decision node:** Checks an additional condition to split data into smaller groups.
- **Branch:** Connects nodes and represents the path determined by a condition's outcome.
- **Leaf:** A terminal node that produces a prediction without further splitting.

During training, the model learns splitting conditions. During prediction, an observation follows the appropriate branches from root to leaf. Tree depth is the number of edges on the longest root-to-leaf path, with the root at depth 0.

### Training a decision tree

Training determines which variables and conditions should split the data to create classification rules.

Before training, prepare data in a suitable format, including handling missing values and encoding categorical variables as numbers. Validation data must use the transformations learned from training data so that features remain consistent.

After training, evaluate the model on data not used for fitting to assess whether its rules generalize to new observations. This lesson uses ROC AUC.

### Overfitting

Decision trees can learn complex patterns, but are prone to overfitting, especially without a depth limit.

A very deep tree can divide data into small, highly specific groups and learn details or coincidences rather than patterns that transfer to new data. A key sign is very strong training performance combined with substantially weaker validation or test performance. The model memorizes training data but generalizes poorly.

Without restrictions, splitting may continue until leaves contain only one class. These are **pure leaves**. Perfect separation of training data does not guarantee good predictions on new data.

### Controlling tree complexity

Limiting depth with `max_depth` reduces the number of successive conditions and forces the tree to learn broader rules, reducing the chance of memorizing training details.

Reducing complexity may lower training performance while improving performance on new data. Choose tree size using validation results, not training results alone.

### Inspecting learned rules

Display learned rules as text to inspect variables, split conditions, and predictions along each path. This helps explain the decision process. For one-hot encoded variables, interpret conditions according to the encoded columns' meanings.

### Decision stump

A tree of depth 1 is a **decision stump**. It has one root split followed by prediction leaves and is the simplest decision tree.

```text
                Root
          Check a condition
             /         \
           True       False
            /           \
          Leaf         Leaf
       Prediction   Prediction
```

## 2 Decision Tree Learning Algorithm

This section explains how a tree chooses split conditions, what impurity means, and when splitting stops.

### Tree terminology

The tree begins at the root. Each splitting node contains a condition and two outgoing branches for true and false outcomes. A node being split is a **parent node**; the resulting nodes are **child nodes**. A terminal node is a leaf, where prediction occurs. Depth is measured in edges on the longest root-to-leaf path, starting at 0.

### Choosing features and thresholds

At each node, select:

- **Feature:** The variable used to split data.
- **Threshold:** The value defining the split condition.

For a numeric feature, split observations into values less than or equal to the threshold and values greater than it. Evaluate candidate thresholds for each feature by the purity of the resulting groups. Numeric thresholds can generally be placed midway between sorted, distinct observed values.

### Impurity

Impurity measures class mixing within a group. If all observations share one class, the group is pure and impurity is zero. Mixing classes increases impurity. Splitting aims to separate classes more clearly and reduce impurity.

### Misclassification rate

If a group predicts its majority class, observations from other classes are misclassified. The misclassification rate is therefore the proportion outside the majority class.

$$
\text{Misclassification rate}=1-\max_k p_k
$$

The maximum class proportion is the majority class's proportion. A single-class group has a misclassification rate of zero.

### Evaluating a split

Combine child impurities using a weighted average based on group sizes:

$$
I_{\text{split}}=\frac{n_L}{n}I_L+\frac{n_R}{n}I_R
$$

- $n$: Observations in the node before splitting.
- $n_L$, $n_R$: Observations in the left and right groups.
- $I_L$, $I_R$: Impurity of each group.

Larger groups contribute more. Choose the lowest weighted impurity, equivalently the largest reduction from the parent node's impurity.

### Considering multiple features

1. Consider features available for splitting.
2. Find candidate thresholds for each feature.
3. Split data using each candidate condition.
4. Calculate weighted impurity after splitting.
5. Choose the feature and threshold with the lowest impurity.

The choice uses data at the current node. It is locally best at that step and does not guarantee the globally best tree structure.

### Impurity criteria

| Criterion | Principle |
| --- | --- |
| Misclassification rate | Proportion outside the majority class |
| Gini impurity | Class mixing measured from class proportions |
| Entropy | Uncertainty about class membership within a group |
| MSE | Mean squared prediction error for regression |

Gini impurity and entropy are more sensitive to class-proportion changes than misclassification rate, making them useful classification split criteria.

Regression trees predict numeric values. Their splits reduce within-group prediction error rather than separate classes.

### Recursive splitting

After splitting the root, repeat the procedure for each child node.

```text
Data at the current node
          |
Check stopping criteria
      /         \
    Stop      Can split
     |            |
Create leaf   Find best split
and predict       |
            Create child nodes
               /       \
          Left group  Right group
               |       |
          Repeat for each group
```

Applying the same procedure to smaller groups is called **recursion**.

### Stopping criteria

Stop splitting when a specified condition is met:

- The group is pure, so further class separation is unnecessary.
- Maximum depth is reached, controlled by `max_depth`.
- Too few observations remain to satisfy minimum sample constraints.
- Maximum leaf count is reached, controlled by `max_leaf_nodes`.

| Parameter | What it controls |
| --- | --- |
| `min_samples_split` | Minimum observations a node must contain before splitting is considered |
| `min_samples_leaf` | Minimum observations required in each child after splitting |

These constraints control complexity and reduce overfitting risk, but do not guarantee complete prevention.

### Complete learning process

Starting at the root, check whether splitting is allowed, select the feature and threshold that reduce impurity most, and repeat for each child. When a stopping condition is met, create a leaf and determine its prediction from the observations it contains.

The next step uses these principles to select `max_depth` and `min_samples_leaf` for suitable complexity and generalization.

## 3 Decision Tree Parameter Tuning

Parameter tuning selects settings that control learning to improve predictions on new data. This lesson compares parameter combinations using validation ROC AUC.

### Main parameters

| Parameter | Role |
| --- | --- |
| `max_depth` | Limits maximum tree depth |
| `min_samples_leaf` | Minimum observations required in each leaf |

Both limit rule specificity and reduce the chance of memorizing training details.

### Selecting max_depth

`max_depth` limits successive splits along a root-to-leaf path. `None` removes the depth limit, although other stopping criteria still apply.

An overly shallow tree may miss important patterns and underfit. An overly deep tree may create detailed rules and overfit. Train at different depths and use validation results to find a suitable range instead of selecting by training results alone.

### Selecting min_samples_leaf

A split is not allowed if either child would contain fewer training observations than `min_samples_leaf` requires.

A low value permits small groups and detailed rules, increasing overfitting risk. A high value requires larger groups and broader rules; an excessively high value may prevent sufficient learning.

### Joint parameter tuning

Depth and leaf size interact. A suitable depth may change when minimum leaf size changes. The lesson uses this sequence:

1. Test depths to identify a promising range.
2. Combine those depths with several minimum leaf sizes.
3. Record validation AUC for each pair.
4. Choose the best pair within the search space.

Staged search reduces training runs, but narrowing too early may miss better combinations elsewhere. Begin with a **coarse search**, then refine promising regions within the available time and resources.

### Pivot tables and heatmaps

Arrange results with `min_samples_leaf` as rows, `max_depth` as columns, and validation AUC in the cells. This pivot table can be displayed as a heatmap using color to compare scores.

Heatmaps show high-scoring pairs, regions of similarly strong settings, and how performance changes with complexity. The best cell is only best among the settings tested on that validation set, not universally best.

### Bias and variance

- **Bias:** Error from limitations or excessive simplicity that prevent important patterns from being captured.
- **Variance:** Sensitivity to training-data changes, causing rules and predictions to change substantially across training samples.

| Parameter change | Complexity | Bias | Variance |
| --- | --- | --- | --- |
| Increase depth and decrease minimum leaf size | Higher | Lower | Higher |
| Decrease depth and increase minimum leaf size | Lower | Higher | Lower |

Find a balance that captures important patterns without excessive sensitivity to training-specific details.

### Training with selected settings

Train with the selected parameters and inspect rules to understand splitting order and complexity.

Check that values are valid and behave as intended, particularly after storing or reshaping results. Do not confuse `None`, meaning no depth limit, with `NaN`, representing a missing or nonnumeric value. A good score is insufficient if the configuration does not control the intended behavior.

## 4 Ensemble Learning and Random Forest

Combining multiple decision trees can make predictions more stable and improve generalization. This section also covers random forest parameter tuning.

### Ensemble learning

Ensemble learning combines several models into one final prediction. Models may learn different patterns and make different errors. Combining diverse models reduces dependence on one model and may outperform a single model.

Components are **base learners**. Some methods use **weak learners**, which have limited individual capability, but ensemble members do not always have to be weak learners.

### Combining predictions

- **Classes:** Vote and choose the most popular class.
- **Probabilities:** Average class probabilities.
- **Numeric values:** Average predicted values.

Probability averaging follows:

$$
P(y=c\mid x)=\frac{1}{M}\sum_{m=1}^{M}P_m(y=c\mid x)
$$

$M$ is the number of models. $P_m(y=c\mid x)$ is the probability of class $c$ predicted by model $m$.

### Random forest

A random forest combines independently trained decision trees. Averaging identical trees adds no benefit, so randomness creates diversity and reduces correlation between errors.

Randomness occurs through sampling training observations for each tree and sampling candidate features at each split. Combining diverse trees reduces variance, or sensitivity to training data, and stabilizes predictions.

### Bootstrapping

Bootstrapping samples training rows **with replacement**. A row can appear multiple times while others are absent. Trees receive separately sampled datasets and learn from different compositions of observations.

`bootstrap` controls this behavior. If disabled, each tree uses all training observations, but feature randomness can still create differences.

### Random features at each node

At each node, consider a random feature subset and choose its best split. **Sampling occurs at each node**, rather than assigning one fixed subset to an entire tree.

This reduces repeated selection of the same dominant variables across trees and increases structural and prediction diversity. `max_features` controls how many features are considered when searching for each split.

### Random forest workflow

```text
                     Training data
                           |
              Sample data for each tree
                 /         |         \
             Sample     Sample     Sample
                |          |          |
              Tree 1     Tree 2     Tree M
                |          |          |
       Random feature selection at each node
                |          |          |
           Prediction Prediction Prediction
                 \         |         /
                  Combine predictions
                           |
                    Final prediction
```

For classification in this lesson, average each class's probabilities across all trees. To predict a class, choose the highest average probability.

### Choosing n_estimators

`n_estimators` sets tree count. With few trees, results may be sensitive to randomness. Adding trees generally stabilizes predictions and may improve validation performance.

After enough trees, performance often levels off. Additional trees offer diminishing benefits while increasing training time, prediction time, and memory use. Validation scores can fluctuate, so more trees do not guarantee improvement at every step. Choose a stable result within resource limits.

### Tuning max_depth

`max_depth` limits each tree's depth. Shallow trees may miss patterns; very deep trees may learn training-specific details. Combining trees reduces variance but does not eliminate all overfitting. Compare validation results while considering tree count.

### Tuning min_samples_leaf

Small minimum leaf sizes allow detailed patterns but may base predictions on small groups. Larger values use more observations per leaf and broader rules, but excessive values may underfit.

The best setting can differ from a single tree because aggregation reduces variance at another level.

### Tuning procedure

1. Vary tree count to find where performance stabilizes.
2. Compare maximum depths.
3. Select a promising depth and vary minimum leaf size.
4. Choose settings using validation performance.
5. Train using the selected parameters.

Line plots compare score levels and stability as tree count increases. Consider overall trends as well as peak scores.

### Parameters to remember

| Parameter | Role |
| --- | --- |
| `n_estimators` | Number of trees |
| `max_depth` | Maximum depth per tree |
| `min_samples_leaf` | Minimum observations per leaf |
| `max_features` | Features considered per split |
| `bootstrap` | Whether to sample rows with replacement |
| `random_state` | Controls randomness for repeatable experiments under the same conditions |

Random forests depend on tree diversity and prediction aggregation. Tuning considers individual tree capability, combined stability, and computation cost.

## 5 Gradient Boosting and XGBoost

Gradient boosting builds models sequentially. Each new model corrects errors in the combined prediction built so far. With decision tree base learners, the method is called **gradient boosting trees**.

### Random forest versus gradient boosting

| Aspect | Random forest | Gradient boosting |
| --- | --- | --- |
| Training | Independent trees; can run in parallel | Sequential; each round depends on earlier results |
| Tree role | Predict the target | Improve current ensemble predictions |
| Aggregation | Average predictions | Add successive corrections |
| Main principle | Diversity and variance reduction | Progressive error reduction |

A new boosting tree corrects the entire current ensemble, not only the immediately preceding tree.

### Learning process

Start with an initial prediction and evaluate error using a loss function. Train a new tree to learn a direction that reduces loss, then add its correction to the ensemble.

```text
Initial prediction
        |
Evaluate current ensemble loss
        |
Find a correction direction using gradients
        |
Train a new correction tree
        |
Add its correction to the ensemble
        |
Evaluate and repeat
        |
Final prediction
```

A gradient describes how loss changes as predictions change. The algorithm uses this information to learn corrections that reduce loss.

For squared loss, the learning target for a new tree relates directly to the difference between actual and predicted values. For other losses, corrections follow the gradient of the corresponding loss.

### Aggregation and learning rate

At each round:

$$
F_m(x)=F_{m-1}(x)+\eta f_m(x)
$$

- $F_{m-1}(x)$: Combined prediction before the new tree.
- $f_m(x)$: Correction from the new tree.
- $\eta$: Learning rate.
- $F_m(x)$: Updated combined prediction.

Learning rate controls a new tree's influence. High values make larger corrections; low values learn gradually and usually require more rounds. Consider learning rate and round count together.

For classification, the combined value may be an internal score transformed into a probability. It is not an average of tree probabilities as in random forest.

### XGBoost

XGBoost implements gradient boosting with efficient training. This lesson uses it to build trees sequentially, monitor learning, and identify excessive fitting of training details.

Boosting rounds are sequential, but computations within a round can use multiple threads.

### Preparing data with DMatrix

The training approach here uses **DMatrix**, a data structure designed for XGBoost. It stores features and associated information such as targets and feature names to keep training and evaluation consistent.

Training and validation matrices must have matching feature meanings and column order.

### Main parameters

| Parameter | Role |
| --- | --- |
| `eta` | Learning rate controlling correction size |
| `max_depth` | Maximum tree depth |
| `min_child_weight` | Minimum child-node weight based on summed loss Hessians |
| `objective` | Learning task and loss |
| `num_boost_round` | Rounds adding trees or models |
| `eval_metric` | Monitoring and evaluation metric |
| `nthread` | Computation threads |
| `seed` | Random seed for reproducibility |
| `verbosity` | Diagnostic message level |

`min_child_weight` is not a direct minimum sample count. It resembles `min_samples_leaf` in controlling splits and complexity but uses a different criterion.

### Objective and evaluation metric

- **Objective:** Determines what the model learns to minimize during training.
- **Evaluation metric:** Measures prediction performance during training.

`binary:logistic` learns binary classification with logistic loss and outputs probabilities. Selecting AUC as the evaluation metric monitors AUC without making it the directly optimized loss.

### Monitoring training

Round count affects complexity, so monitor training and validation performance. The datasets evaluated after each round form the **watchlist**. Validation data are for evaluation, not fitting correction trees.

Plot recorded results to assess whether continued learning improves new-data predictions, whether training and validation scores diverge, and how many rounds to use.

### Overfitting in gradient boosting

Initially, additional trees may improve both training and validation results. Later, training-specific details may dominate. A warning sign is improving training performance while validation performance plateaus or declines.

More boosting rounds do not guarantee better generalization. This differs from adding forest trees to stabilize an average.

### Choosing boosting rounds

Choose rounds by the best validation performance, not training performance. The last round may not be best if validation results have already declined.

The next section balances correction size, tree complexity, and round count.

## 6 XGBoost Parameter Tuning

Tuning aims to capture important patterns without excessively fitting training details. Learning curves compare settings and select round count, using validation AUC as the main criterion.

### Main parameters

| Parameter | What it controls |
| --- | --- |
| `eta` or `learning_rate` | Correction size per round |
| `max_depth` | Maximum tree depth |
| `min_child_weight` | Minimum child-node weight for an accepted split |
| `num_boost_round` | Number of correction rounds |

The first three control learning behavior. Round count controls how long learning continues.

### Tuning eta

`eta` scales the new tree's correction before adding it to the ensemble, acting as the learning step size.

High values make large updates: scores can improve quickly, but the best point may be passed and overfitting can begin sooner. Low values make small updates and usually need more rounds. With too few rounds, the model may remain underfit.

A lower learning rate is not automatically better. Evaluate it together with round count and validation performance.

### Tuning max_depth

`max_depth` limits each added tree's complexity. Deep trees capture complex relationships but can learn training-specific details. Shallow trees offer simpler corrections and may need more rounds.

Individual boosting trees need not be highly complex because many corrections accumulate in the ensemble. More depth does not always improve generalization. Compare validation AUC and its trend across rounds.

### Tuning min_child_weight

This split constraint uses the sum of Hessians, or second-order loss derivatives, in each child. It is not directly a minimum observation count, although its complexity-control role resembles `min_samples_leaf`.

- **Low:** Permits finer splits and specific patterns.
- **High:** Restricts splitting and overly specific groups.
- **Excessively high:** May block useful splits and cause underfitting.

Very similar results across tested values may provide insufficient evidence that changing this parameter meaningfully improves performance.

### Comparing learning curves

Learning curves relate boosting rounds to evaluation scores. Compare learning speed, peak locations, stable high-performance regions, and declines with continued training.

Consider training and validation results together. Improving training scores with declining validation scores suggests overfitting.

Zooming the score axis exposes small differences, but read axis values: large visual gaps may represent only minor score differences.

### Tuning sequence

Change one parameter at a time while keeping others fixed:

```text
Try learning rates
        |
Select a suitable value
        |
Try maximum depths
        |
Select a suitable value
        |
Try minimum child weights
        |
Choose boosting round count
        |
Train with the selected settings
```

This simplifies comparisons and reduces experiments. However, interactions mean it does not guarantee the best combination overall. After changing depth or split constraints, revisit learning rate and round count if necessary.

### Selecting round count

Choose the region with the best validation AUC, not the highest training AUC. Too few rounds may underfit; too many may increase complexity and reduce new-data performance.

Round count is part of tuning and must be considered alongside learning rate and tree complexity.

### Other complexity controls

| Parameter | Principle |
| --- | --- |
| `subsample` | Fraction of training observations sampled per round |
| `colsample_bytree` | Fraction of features sampled for each tree |
| `lambda` or `reg_lambda` | L2 regularization on leaf scores |
| `alpha` or `reg_alpha` | L1 regularization on leaf scores |

Sampling reduces reliance on the same full dataset in every tree. Regularization limits correction magnitudes and controls complexity.

Train with the chosen settings. Suitable values depend on the data and problem; there is no universally best fixed configuration.

## 7 Selecting the Best Model

After tuning, compare performance, select the final model, and evaluate on held-out test data. **Use validation data for selection and test data for the final assessment.**

### Comparing models

Compare tuned models trained on the same training data and evaluated on the same validation set with the same metric. This makes the comparison consistent.

This lesson selects the highest validation ROC AUC among the tested models. The outcome applies to that dataset and experiment, not every problem.

### Model characteristics

| Model | Main characteristics |
| --- | --- |
| Decision tree | Easy to inspect and explain, but prone to overfitting |
| Random forest | Independent trees reduce variance and improve stability |
| XGBoost | Sequential correction trees require careful tuning and round selection |

XGBoost can learn complex tabular patterns but needs complexity control and monitoring. Random forests often work well with less detailed tuning, depending on the data and problem.

### Training the final model

After selecting model type, parameters, and round count, combine training and validation data into the **full training set**. Retrain with the chosen settings so the model can learn from more data.

The original validation set now contributes to training and cannot independently evaluate the retrained model.

### Preparing data for retraining

Separate the target from features to prevent the true answer from becoming a predictor. Fit preprocessing only on the full training set, then apply the learned transformations to test data, preserving feature meanings and order.

Do not use test data to fit preprocessing or tune parameters, as that would influence the model before final evaluation.

### Test evaluation

Keep the test set separate throughout training, tuning, and selection. It evaluates performance on unseen data at the final stage.

```text
Train and tune candidate models
              |
Compare on the same validation set
              |
Select model and parameters
              |
Combine training and validation data
              |
Retrain the final model
              |
Evaluate on the test set
```

Select the model before checking test results. If test scores guide selection or adjustment, that dataset becomes part of the decision process and no longer provides an independent final test.

### Assessing generalization

Test results close to validation results support transfer of learned patterns to unseen data and consistency between selection and final evaluation.

Similar scores alone do not guarantee a good model: both can be low. Consider score levels as well as their difference.

The relevant comparison here is **validation versus test**, not training versus test, because training scores may be high simply because the model has seen those observations.
