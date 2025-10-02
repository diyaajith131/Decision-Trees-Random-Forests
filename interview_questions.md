Frequently assking Questions: Decision Trees & Random Forests

1. **How does a decision tree work?**  
   A decision tree splits the dataset into smaller subsets based on feature values. Each internal node represents a feature, branches represent decision rules, and leaves represent outcomes.

2. **What is entropy and information gain?**  
   - **Entropy** measures the impurity/uncertainty in the dataset.  
   - **Information Gain** measures the reduction in entropy when a dataset is split on a feature.

3. **How is random forest better than a single tree?**  
   Random Forest is an ensemble of multiple decision trees. It reduces overfitting, improves accuracy, and provides more stable predictions compared to a single tree.

4. **What is overfitting and how do you prevent it?**  
   Overfitting occurs when a model learns noise instead of patterns.  
   Prevention techniques include: limiting tree depth, pruning, using cross-validation, or ensemble methods like Random Forest.

5. **What is bagging?**  
   Bagging (Bootstrap Aggregating) is a technique where multiple models are trained on different random subsets of the dataset, and their predictions are averaged to improve performance and reduce variance.

6. **How do you visualize a decision tree?**  
   Using libraries like `sklearn.tree.plot_tree()` or exporting the tree to Graphviz for detailed visualization.

7. **How do you interpret feature importance?**  
   Feature importance indicates how much each feature contributes to reducing impurity across the trees. Higher values mean the feature is more significant.

8. **What are the pros/cons of random forests?**  
   - Pros: High accuracy, robust to noise, reduces overfitting, handles missing values.  
   - Cons: Computationally expensive, less interpretable compared to a single decision tree.
