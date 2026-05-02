# Ship Happens: A Titanic Data Story

### Instructions for Execution
* Ensure that all project files are located within the same directory.
* First, execute the `Data_Preprocessing.ipynb` notebook to generate the `Titanic-Dataset-reduced.csv` file.
* Next, run the `Clustering_Analysis.ipynb` notebook to produce the final clustering results.

## Introduction
### Executive Summary.
This study analyzes passenger survival patterns from the 1912 Titanic disaster using the Kaggle Titanic dataset, with the goal of identifying how demographic and socioeconomic factors influenced survival outcomes. The analysis focuses on variables such as passenger class, fare, age, sex, and family composition to better understand how these characteristics shaped survival likelihood during a crisis. To preprocess the data, missing values were handled, the data was transformed to address skewed distributions and outliers, and features were engineered such as family size and passenger titles. Further, the study applies three unsupervised machine learning techniques: K-Means clustering, hierarchical clustering, and DBSCAN. These methods were used to uncover natural groupings of passengers based on shared characteristics, as opposed to directly predicting survival. The results reveal that survival outcomes were strongly associated with socioeconomic and demographic factors. Across all models, clusters defined by female passengers, higher ticket class, and moderate family sizes exhibited significantly higher survival rates. In contrast, large families, adult males, and lower-class passengers were consistently associated with lower survival. Among the models, K-Means clustering achieved the best overall alignment with survival outcomes, while hierarchical clustering provided the most interpretable group structures and DBSCAN identified more conservative, tightly defined clusters. Evaluation against random baselines demonstrates that all clustering approaches capture meaningful patterns in the data, with strong performance across metrics such as ROC AUC and cluster purity. These findings reinforce historical evidence of social inequality and prioritization during the disaster, including the influence of class and gender on access to lifeboats. Overall, this study highlights how unsupervised learning can reveal hidden structure in complex datasets and provides insight into how social hierarchy and demographic factors influence outcomes in emergency situations. The findings have broader implications for modern risk planning, emphasizing the importance of equitable access to safety resources and the value of data-driven approaches for identifying vulnerable populations in crisis scenarios.


### Problem Definition.
This study analyzes passenger survival outcomes from the Titanic sinking in 1912 using the Titanic dataset from Kaggle. Primarily, we determine whether there are patterns that distinguish survivors from non-survivors, and which socioeconomic and demographic factors may have influenced these outcomes. Specifically, the analysis examines whether features such as ticket class and passenger fare (which can be used to deduce socioeconomic status) as well as demographic features such as sex, age, and family composition (giving the number of siblings, spouses, and children on board) is associated with survival. Understanding these relationships is important because the Titanic serves as a historical case-study for decision making under crisis conditions. Analyzing survival patterns can provide insights as to social inequality, prioritization, and human behavior in crisis, which is essential to understand in risk planning.


## Related Work
The Titanic dataset has been widely used in both academic and educational contexts, particularly for exploring predictive modeling and introductory data science techniques. Much of the related work focuses on analyzing survival outcomes based on demographic and socioeconomic factors such as age, sex, passenger class, and fare (Youssefkhaled117, n.d.). Prior studies consistently found that female passengers had significantly higher survival rates than males, often exceeding 70% compared to less than 20% for men, reflecting the “women and children first” evacuation protocol (Youssefkhaled117, n.d.). Passenger class is also strongly associated with survival, with first-class passengers experiencing substantially higher survival rates than those in third class, indicating the role of socioeconomic status and access to lifeboats (Youssefkhaled117, n.d.). Age-based analyses show that younger passengers, particularly children, had higher survival probabilities, while older individuals were less likely to survive (Youssefkhaled117, n.d.). Correlation analyses further reveal that fare is positively correlated with passenger class, as higher fares are typically associated with first-class tickets, and that moderate family sizes (e.g., 2–4 members) are linked to higher survival rates, while very large families or solo travelers had lower chances of survival (Youssefkhaled117, n.d.). Exploratory data analysis using visualizations such as histograms, bar charts, and boxplots highlights skewed age distributions, higher fare variability among first-class passengers, and clear survival disparities across gender and class groups (Bingos, n.d.). In addition to exploratory data analysis, the dataset is frequently used to compare the performance of various machine learning classification models (Youssefkhaled117, n.d.). Common approaches include logistic regression, decision trees, random forests, support vector machines, and k-nearest neighbors. Because the primary task is to predict survival (a binary outcome), the problem is typically framed as a supervised classification task (Youssefkhaled117, n.d.). The Titanic dataset also gained widespread popularity through the Kaggle Titanic: Machine Learning from Disaster competition, where participants applied a wide range of machine learning techniques to predict passenger survival. Analyses from this competition demonstrated that feature engineering—such as extracting titles from names, creating family size variables, and handling missing age values—can significantly improve model performance. Additionally, ensemble methods like random forests and gradient boosting were frequently found to outperform simpler baseline models, highlighting the importance of combining multiple predictors and optimizing model parameters (Kaggle, n.d.).


## Data description
### Data Collection. 
Our dataset was obtained from Kaggle, which is a widely used and reputable platform for sharing datasets in the data science community. Specifically, we used the well-known Titanic dataset, which is commonly used for machine learning and exploratory data analysis projects.

According to the dataset description, the data was compiled using historical records, publicly available sources, and prior research on Titanic passengers. This likely includes archived passenger lists, ship records, and historical documentation that have been aggregated and cleaned by contributors before being uploaded to Kaggle. Because of this, we are working with a secondary dataset rather than raw, first-hand collected data.

While Kaggle datasets are generally reliable, it is important to note that we do not have full transparency into the exact data collection and cleaning methods used by the original contributors. This means there may already be preprocessing decisions or assumptions embedded in the dataset that we must be aware of when interpreting our results.


### Data Preprocessing. 
To transform the raw dataset into a form suitable for analysis and clustering, we performed several preprocessing steps focused on handling missing values, feature engineering, and reducing noise.

First, we addressed missing data. The Cabin feature was dropped entirely because it contained a very high number of missing values (687 out of 891 observations). Since the majority of this data is unavailable, attempting to impute it would likely introduce significant noise and unreliable assumptions. Keeping this feature would therefore hurt the overall quality of our analysis rather than improve it. For the Age variable, we filled missing values using the median instead of the mean. The age distribution is somewhat skewed and contains outliers, particularly at very young and very old ages. Because the median is more robust to extreme values, it provides a better representation of a “typical” passenger. For the Embarked feature, which only had two missing values, we used the mode (most frequent category) to fill in the gaps. Since the number of missing entries is very small, this approach minimizes distortion to the dataset.

Next, we applied a log transformation to the Fare variable. Fare is highly right-skewed, with a few very large values. The log transform compresses these extreme values and makes the distribution more normal, which helps reduce the influence of outliers without removing them entirely.

We also created new features to improve the dataset. One important feature is FamilySize, which combines SibSp and Parch (plus one to include the passenger themselves). These two original variables both describe family relationships, so combining them creates a more meaningful and simplified representation of whether a passenger was traveling alone or with family. This reduces redundancy and helps the model detect clearer patterns.

Another engineered feature is Title, which we extracted from the Name column. Titles such as “Mr,” “Mrs,” “Miss,” and “Master” provide useful demographic and social information, including gender, age group, and potential social status. Since there were also many rare titles (e.g., “Lady,” “Countess”), we grouped these into a single “Rare” category to avoid sparsity and reduce noise in the data.

We then removed unnecessary features, including PassengerId, Ticket, and Name. PassengerId is simply an identifier with no analytical value. Ticket consists of inconsistent alphanumeric codes that are difficult to encode meaningfully. Name was removed because we already extracted the most useful information from it (Title), and the remaining information would add unnecessary complexity.

To prepare the data for clustering, we applied one-hot encoding to categorical variables such as Sex, Embarked, and Title, since clustering algorithms require numerical input. We also scaled the features to ensure that variables with larger ranges did not dominate the analysis.

Finally, we calculated mutual information scores to evaluate feature importance and removed features with very low information gain, specifically the encoded Embarked variables and Title_Rare. This step helped reduce noise and improve model performance.

One potential data quality issue is class imbalance. The dataset contains significantly more male passengers than female passengers (approximately two-thirds male), and most passengers are in third class. This imbalance could influence patterns learned by the model and should be considered when interpreting results.


### Data Documentation. 
This dataset contains information on 891 passengers aboard the Titanic, where each row represents a single individual and each column represents a specific attribute that may be related to survival. The dataset includes a mix of numerical and categorical variables, along with some missing values that required attention during preprocessing.

Below is a detailed description of each feature, including its data type, range (min and max where applicable), and mode (most common value):

* **PassengerId** (int64): This is a numerical variable that serves as a unique identifier for each passenger. It ranges from a minimum value of 1 to a maximum value of 891. Although it is numeric, it does not carry meaningful information for analysis and is only used for identification purposes.
* **Survived** (int64, binary categorical): This variable indicates whether a passenger survived the disaster (0 = No, 1 = Yes). The minimum value is 0 and the maximum value is 1, with a mode of 0, meaning most passengers did not survive. While stored as an integer, it is conceptually a categorical variable.
* **Pclass** (int64, ordinal categorical): This represents the passenger’s ticket class and can be interpreted as a proxy for socioeconomic status (1 = First class, 2 = Second class, 3 = Third class). The values range from 1 to 3, with a mode of 3, indicating that most passengers were in third class.
* **Name** (object): This is a text-based categorical variable containing each passenger’s full name. The mode is “Abbing, Mr. Anthony,” although most names are unique. This feature also contains embedded information such as titles, which we later extracted into a separate variable.
* **Sex** (object, categorical): This variable indicates the passenger’s gender (male or female). The mode is “male,” showing that the dataset contains more male passengers than female passengers.
* **Age** (float64, numerical): This represents the passenger’s age in years. The values range from a minimum of 0.42 to a maximum of 80.0, with a mode of 28.0. There are 177 missing values in this column, which required imputation during preprocessing.
* **SibSp** (int64, numerical): This variable indicates the number of siblings or spouses the passenger had aboard the Titanic. It ranges from 0 to 8, with a mode of 0, meaning most passengers were not traveling with siblings or spouses.
* **Parch** (int64, numerical): This represents the number of parents or children the passenger had aboard. The values range from 0 to 6, with a mode of 0, indicating that most passengers were not traveling with parents or children.
* **Ticket** (object): This is a categorical/text variable representing the passenger’s ticket number. The values are highly inconsistent and mostly unique, with a mode of “1601.” Because of its irregular structure, it is difficult to interpret or encode meaningfully.
* **Fare** (float64, numerical): This variable represents the ticket fare paid by the passenger. It ranges from 0.0 to 512.3292, with a mode of 8.05. The distribution is highly right-skewed, with a few very large values.
* **Cabin** (object): This is a categorical variable indicating the passenger’s cabin number. It has a very large number of missing values (only 204 non-null entries out of 891), making it unreliable for analysis. Most values are unique or sparsely distributed.
* **Embarked** (object, categorical): This variable represents the port where the passenger boarded the ship. The possible values are C (Cherbourg), Q (Queenstown), and S (Southampton). The mode is “S,” indicating Southampton is the most common embarkation point. There are only 2 missing values in this column.

In addition to the original features, we created two engineered features during preprocessing:

* **FamilySize** (int64, numerical): This feature was created by combining SibSp and Parch and adding 1 to include the passenger themselves. It represents the total number of family members traveling together. The range is from 1 (traveling alone) up to 11 (largest observed family group), with a mode of 1, meaning most passengers were traveling alone.
* **Title** (object, categorical): This feature was extracted from the Name column and includes titles such as “Mr,” “Mrs,” “Miss,” and “Master,” along with a grouped “Rare” category for less common titles. The mode is “Mr,” which is consistent with the higher proportion of male passengers.

Overall, the dataset includes a combination of demographic (Age, Sex, Title), socioeconomic (Pclass, Fare), and family-related (SibSp, Parch, FamilySize) variables. While the dataset is rich in information, the presence of missing values (especially in Cabin and Age) and skewed distributions (such as Fare) required careful preprocessing to ensure reliable analysis.


### Variables.  
In our analysis, the target (dependent) variable is Survived, which indicates whether a passenger lived (1) or died (0) during the Titanic disaster.

The independent variables (features) used in our model include:
* *Pclas*s: Passenger ticket class (1st, 2nd, or 3rd), which serves as a proxy for socioeconomic status
* *Title*: Extracted from the passenger’s name (Mr, Mrs, Miss, Master)
  * This feature was split into four separate features by one-hot encoding: *Title_Mr*, *Title_Mrs*, *Title_Miss*, and *Title_Master*
* *Sex*: Gender of the passenger
  * This feature was split into two separate features by one-hot encoding: *Sex_male* and *Sex_female*
* *Age*: Age in years
* *SibSp*: Number of siblings or spouses aboard
* *Parch*: Number of parents or children aboard
* *Fare*: Ticket price paid by the passenger
* *FamilySize*: Engineered feature representing total family members traveling together

These features were selected because they capture meaningful demographic, social, and economic characteristics that are likely to influence survival outcomes. By focusing on these variables, we aim to better understand patterns in survival and improve the effectiveness of our clustering and analysis.



## Modeling
### Description of Algorithm 1 and its model.
We applied the K-Means clustering algorithm to identify natural groupings of passengers. K-Means is a centroid-based algorithm that partitions the data into K distinct clusters by minimizing within-cluster variance. We used a feature set that included Age, Pclass, Fare, SibSp, Parch, FamilySize, Sex, and Title. The key parameter was the number of clusters, K. Since K-Means relies on Euclidean distance, all features were standardized using z-score scaling to ensure equal contribution across variables. We fine-tuned the parameters. A random state of 42 was used to ensure reproducibility across runs. To determine the optimal number of clusters, we created an elbow plot and silhouette scores plot.

The elbow plot suggested K=5 or K=6 as optimal candidates, while silhouette scores indicated that K=7 and K=6 performed best. Ultimately, K=6 was selected as the final model because it offered the best balance of clear cluster interpretability, stable separation between groups based on visual inspection, and preservation of meaningful subpopulation structure. The model was allowed to run until convergence, with a maximum iteration limit set by the default implementation.


### Description of Algorithm 2 and its model.
The second algorithm we used was Hierarchical clustering to group similar data points based on their feature similarity. Specifically, it builds a hierarchy of clusters by starting with each observation as its own cluster and then iteratively merging the closest pairs of clusters until all points are merged into a single cluster. The distance between two clusters is measured using Euclidean distance and we use Ward’s method to minimize the total within-cluster variance at each step of merging. This approach produces compact, spherical clusters by choosing merges that result in the smallest increase in overall variance. To visualize hierarchical clustering, a dendrogram is generated using scipy. In this implementation, the truncate_mide=’lastp’ parameter is used to simplify the visualization by showing only the last 30 merged clusters (p=30), rather than the full tree, which helps to improve readability, especially for larger datasets. The show_leaf_counts=True option displays how many original observations are contained in each cluster, while distance_sort=’descending’ organizes the merges by height to emphasize the most significant cluster separations. The show_contracted=True parameter visually compresses intermediate merges for clarity. Based on the dendrogram, there was clear separation between two clusters; however, we can assume that this was simply separating men and women. In order to extract meaningful and interpretable patterns beyond just men and women, we decided to use four clusters for our hierarchical clustering analysis. 


### Description of Algorithm 3 and its model.
For our third algorithm, we used DBSCAN clustering. DBSCAN is a density-based clustering algorithm that groups data points based on regions of high point density, allowing it to detect clusters of arbitrary shape while also identifying outliers as noise. This makes it particularly useful for uncovering patterns that may not be captured by centroid-based methods such as k-means. We used a feature set that included Age, Pclass, Fare, SibSp, Parch, FamilySize, Sex, and Title. Since DBSCAN relies on distance, the data was first scaled using the Standard Scaler from SkLearn. Further, In order to determine the epsilon neighborhood value, we created an elbow plot using k-distances with k=24. We chose k=24 because as a rule of thumb, k is the number of features times 2 (Shreiber, 2020). With this value in mind for k, our elbow plot appeared to have an elbow at 2.5, and so an epsilon of 2.5 was used for our model. Choosing ε = 2.5 is reasonable because it captures the transition between dense clusters and sparse regions, allowing DBSCAN to group the main structure of the data while labeling the more isolated points as noise. Further, for our minimum samples, we used the same value of k as we did for determining the k-distance plot. 



## Evaluation
### Baseline.
Our baseline, based on the passenger data from History on the Net, was that approximately 38% of passengers survived, so a simple non-machine learning approach is to assign a 38% probability of survival to each individual randomly. This provides a benchmark to compare more sophisticated models. However, the dataset is imbalanced, as 62% of passengers did not survive. We assume that each individual has a fixed probability of survival (set to 0.38) and then generate predictions by randomly assigning each observation a 0 (did not survive) or 1 (survived) based on that probability. The use of a fixed random seed ensures that the results are reproducible. In addition to these binary predictions, the model assigns the same constant probability (0.38) to every observation, which is used for computing ROC AUC. This means the model has no ability to distinguish between individuals and treats everyone identically in terms of predicted risk. The resulting performance metrics such as accuracy, precision, recall, and F1 score reflect the behavior of random guessing, while the ROC AUC will typically be around 0.5, indicating no discriminatory power.

To further evaluate our clustering analysis we also used a random cluster assignment as our baseline. We assign each data point to a random cluster. The random clustering baseline collapses to predicting only the majority class (non-survival), resulting in zero precision and recall for the positive class. This occurs because randomly formed clusters reflect the overall class imbalance, with most clusters containing fewer than 50% survivors. As a result, no cluster is labeled as “survived” under majority voting. This highlights that without meaningful structure in the feature space, clustering cannot recover minority-class patterns and that clustering reveals structure that actually aligns with survival.

The inclusion of random probability and random-clustering baselines ensures that improvements are due to meaningful structure in the data rather than chance or class imbalance, and demonstrates that clustering methods successfully recover patterns strongly associated with survival.


### Metrics.
To ensure that we account for the imbalance in the dataset since majority of the passengers did not survive, we went beyond using just simple accuracy as a metric and used metrics such as precision, recall, ROC AUC curve and F1-score to ensure our model effectively identifies survivors rather than being biased toward the majority class. The baseline achieves moderate accuracy (basically has a 50/50 shot of making the correct prediction) but performs poorly on precision, recall, and F1 score. The ROC AUC is 0.5 because we are making predictions off of a random generator. On the other hand, all clustering methods significantly outperform the random guessing across every metric, indicating that survival is strongly associated with underlying feature groupings (e.g., gender, class, family structure). K-Means is the best performing model as it has the best overall balance and is best at correctly identifying survivors without too many false positives. It also had a high ROC AUC (0.83) which reinforces the classic Titanic pattern of "women and children first" as the high-survival clusters were women in the higher classes. Hierarchical clustering had the most balanced recall, meaning that it performed the best at finding survivors. It has slightly more false positives than K-Means. DBSCAN produces tighter, more conservative clusters, leading to fewer false positives but more false negatives (so it misses more actual survivors). The model with the best cluster separation (Hierarchical) is not the best at predicting survival meaning that better geometric clustering does not necessarily translate to better alignment with real-world outcomes like survival. The clustering methods performing stronger than our random baseline suggests that survival on the Titanic was not random but strongly driven by structured demographic and socioeconomic factors. Among the models, K-Means achieved the best overall predictive alignment with survival outcomes, while hierarchical clustering revealed more clear and precise clusters but with slightly worse predictive performance. DBSCAN was more conservative and less sensitive to actually identifying survivors.

Although clustering algorithms are unsupervised and do not use labels during training, we evaluated their performance by comparing cluster assignments to ground truth survival labels. This is done by assigning each cluster its majority class and assessing agreement with both the baseline and observed survival outcomes.  

**Additional K-Means metrics**

For KMeans, we reported SSE (inertia), silhouette score, and purity. SSE decreases rapidly as k increases up to k equals 6, after which the rate of improvement slows, suggesting diminishing returns in cluster compactness. This supports k equals 6 as a reasonable balance between model simplicity and within-cluster variance. The silhouette scores range from 0.35 to 0.45, with higher values around k equals 6, indicating moderate separation between clusters. The purity score of 0.81 shows strong alignment between clusters and survival outcomes, suggesting that meaningful structure exists in the feature space. 

**Additional Hierarchical Clustering Metrics**

For hierarchical clustering, we reported silhouette score, cophenetic correlation, and purity. The silhouette score of 0.496 reflects moderate cluster separation. The cophenetic correlation of 0.766 indicates that the dendrogram preserves the original pairwise distances reasonably well, meaning the hierarchical structure is a good representation of the data. The purity score of 0.79 suggests slightly weaker but still meaningful alignment with survival compared to the other methods. 

**Additional DBSCAN Metrics**

For DBSCAN, we used silhouette score, Davies-Bouldin index, and purity. The silhouette score of 0.518 indicates moderate separation between clusters, while the Davies-Bouldin index of 0.99 suggests relatively compact clusters with limited overlap (lower values indicate better separation). The purity score of 0.81 further indicates strong but not perfect alignment with survival structure, showing that density-based clustering captures meaningful groupings in the data. 

**Holistic Summary**

Overall, the three clustering methods achieve similar performance in terms of purity (~0.79–0.81) and moderate cluster separation (silhouette scores ~0.35–0.52), indicating that each method captures meaningful structure in the data but emphasizes different aspects. K-Means shows the strongest alignment with survival outcomes, making it the most effective method for identifying survival-related structure. DBSCAN produces the most clearly separated density-based clusters, but these are less aligned with survival labels. Hierarchical clustering provides the most faithful representation of overall data structure but offers slightly weaker predictive alignment. Overall, clustering performance varies by objective: K-Means best captures survival structure, DBSCAN best identifies dense regions, and hierarchical clustering best preserves global relationships.


## Discussion
### Discussion of the Results. 
Our clustering analysis of Titanic passengers reveals underlying structure in survival outcomes that is not immediately visible from raw variables alone. While survival on the Titanic is known to have been influenced by factors such as socioeconomic status, age, and gender (Youssefkhaled117, n.d.), the clustering approach allows these variables to interact in multivariate space, uncovering latent groupings of passengers with distinct survival profiles. Rather than treating each feature independently, clustering captures combinations of demographic and socioeconomic characteristics that jointly shape likelihood survival. 

#### **K Means Clustering Results**

The results of the K Means clusters exhibit six interpretable groupings, each corresponding to distinct passenger profiles with different survival rates.

**Cluster 0**

Cluster 0 represents large family groups primarily drawn from second and third class, with a mean survival rate of only 20%, making it one of the least likely groups to survive. This cluster is characterized by the largest family sizes in the dataset (with a mean of 6.86), along with high levels of siblings/spouses (mean of 3.37) and parents/children (mean 2.49), suggesting extended family travel groups. Passengers are relatively young (mean age of 23), and show a mixed composition of titles, including Mr. Mrs, and Miss, indicating a range of family roles within this group. Despite moderately high fares, this cluster’s strong defining feature is its family size and third class representation (mean class of 2.68). The low survival rate suggests that being part of a large traveling family may have reduced mobility and evacuation efficiency during the disaster.

**Cluster 1**

Cluster 1 represents mostly young, unmarried females (98.8% Title_Miss) with very small family sizes and moderate second class representation (mean Pclass = 2.23) with moderate fare values. The group has a high survival rate (77%), which is consistent with historical evacuation priorities favoring women and younger passengers and results of related work which have consistently shown that female passengers had survival rates exceeding 70% (Youssefkhaled117, n.d.). The small family sizes (mean 1.62) in this cluster may have further improved outcomes by increasing mobility during evacuation.

**Cluster 2**

Cluster 2 had the highest survival rate at 82%, and is composed of almost entirely married women (99.2% Title_Mrs) with significant first class representation (mean Pclass = 1.93). The mean age in this cluster is 34.6, and family sizes remain moderate (mean = 2.28), indicating a mix of spouses or small family units. This shows that within the first class, married women were the most protected subgroup.

**Cluster 3**

Cluster 3 consists primarily of adult men (89.1% Title_Mr) with a mean survival rate of 28%, despite relatively high socioeconomic status (mean Pclass = 1.28). This group is the oldest among all clusters, with a mean age of 40.1, and small family sizes (mean = 1.59). This cluster is particularly important because it demonstrates that high socioeconomic status alone did not guarantee survival. While prior work emphasizes first-class advantages, it also consistently shows that male passengers had significantly lower survival rates overall, often below 20% (Youssefkhaled117, n.d.). The first class men have a survival rate above 20%, but did not overcome strong gender-based evacuation priorities. This reinforces the interpretation that gender was a more dominant survival factor than class in certain contexts.

**Cluster 4**

Cluster 4 represents the most disadvantaged group in the dataset, with an extremely low survival rate of 11%. This cluster is overwhelmingly adult male (99.2% Title_Mr), with a mean age of 28.5, and the smallest family sizes on average (1.20), indicating mostly solo travelers. This group is strongly concentrated in the third class (mean Pclass = 2.82), reinforcing its socioeconomic disadvantage. The extremely low survival rate aligns closely with historical accounts in which third-class passengers faced physical barriers, restricted access to lifeboats, and delayed evacuation. This finding directly supports prior research showing that third-class passengers experienced the lowest survival rates due to structural inequality on the ship (Youssefkhaled117, n.d.).

**Cluster 5**

Cluster 5 consists entirely of children (Title_Master = 100%) with a mean survival rate of 57% and a very low mean age of 6.9. This cluster also has relatively large family sizes (mean = 4.68), suggesting children were typically traveling with parents or siblings. This cluster clearly reflects the “women and children first” evacuation policy, which has been widely documented as a key factor influencing survival outcomes. Prior work shows that children generally had higher survival probabilities than adult males, though outcomes varied depending on family structure and access to lifeboats (Youssefkhaled117, n.d.).



#### **Hierarchical Clustering Results**
Overall, the hierarchical clustering model identified four distinct passenger profiles that align closely with known survival patterns of the Titanic sinking, as can be seen in the corresponding dendrogram shown below as well as the t-SNE and PCA visualizations of the four clusters. 

Low-survival clusters consisted primarily of lower-class individuals traveling alone, while high-survival clusters were made up of wealthier passengers and smaller family groups. A particularly notable cluster (cluster 2) captured large families with children, revealing a nuanced trade-off where the survival advantage of youth was offset by the challenges of evacuating larger groups and men. Overall, the clustering results highlight the combined influence of socioeconomic status, age, and family structure on survival outcomes.

**Cluster 0**

We can characterize Cluster 0 as low-class, low survival males. This cluster contains mostly males with the Title_Master feature. This was the largest cluster with 534 passengers and a low survival rate of only 16%. Mostly contains 3rd class individuals (median = 3) and an average age of 32, meaning it consists mostly of adults. The family size is ~1.4 so we can assume that most of these individuals traveled alone. And they had the lowest fare overall. This cluster represents working-class adults mostly traveling alone, likely men in third class.

**Cluster 1**

We can characterize Cluster 1 as young women and unmarried women. This cluster contains majority females as indicated by being almost entirely made up of Title_Miss. This group has a fairly high survival rate of 72%. Their average age is 23 (young) and an average family size of ~2 with an average Sibsp of 0.58 and an average Parch of 0.51. Indicating that these are single women traveling with either a parent or sibling. The class level is mixed, but leans towards 2nd and 3rd class. The fare level is moderate so overall, this cluster represents the young women (many unmarried) traveling with one other person.

**Cluster 2**

We characterize Cluster 2 as large families with children. This is the smallest cluster with only 46 passengers. This contains a mixture of titles including Title_Master at ~87%, Title _Miss at 6.5% and Title_Mrs at 6.5%. The survival rate is only 50% indicating a likely mixture of genders. The average age is low at ~9.7 which indicates that there is a substantial amount of children in the cluster. The feature SibSp has an average value of 3.04 meaning that there are many siblings. The Parch feature has an average value of 1.46 indicating that there are presents in the cluster. The survival rate being mixed at 50% grants that insight that the survival advantage of children is partially offset by the logistical challenges of large families and the low survival rate of men.

**Cluster 3**

Lastly, Cluster 3 can be characterized as wealthier adults and families. This cluster has the highest survival rate (~79%) with a mean class level of ~2 (many 1st class). They also have the highest fare indicating that they are the wealthiest group. Their average age is ~35 and the family size is ~2.5. This cluster is composed primarily of people with the title Title_Mrs which reveals that this was composed of married women and their children. Overall, this cluster represents wealthier passengers (mostly women & families) in better cabin locations.

A boxplot of Fare distribution across hierarchical clusters indicates that Clusters 2 and 3 (primarily female clusters with families) tend to have higher median fares and a wider spread, suggesting a higher class composition compared to Clusters 0 and 1. From the boxplot we can see a positive correlation between survival rate and fare, indicating that there may be an association between wealth and class that played a role in survival outcomes.


#### **DB-Scan Results**
The DB-Scan results yield 4 interpretable clusters, with one cluster accounting for noise, or rare combinations of features, in the dataset. Each cluster corresponds to groups distinctly separated by factors such as gender, age, marital status, and socioeconomic status.


**Cluster -1**

Cluster -1 represents the noise points identified by DBSCAN - passengers who do not belong to any dense grouping and therefore exhibit more unique or less common combinations of features. This cluster is primarily composed of younger passengers, with an average age of around 16, and ~62% of the passengers in this grouping hold the title “Master,” indicating a strong presence of male children. These passengers tend to travel in larger family groups, as reflected by higher values of siblings/spouses (SibSp is ~2.75) and parents/children (Parch is ~1.97). Although most belong to third class, their fares are relatively higher compared to typical third-class passengers, suggesting some variability in socioeconomic status. The survival rate in this cluster is moderate (approximately 43%), which may reflect the combined influence of youth and family protection during evacuation. Overall, this cluster captures atypical profiles where age and family structure create different survival dynamics than those seen in more clearly defined groups.

**Cluster 0**

Cluster 0 is the largest and most clearly defined group, consisting entirely of male passengers, with ~96% of those holding the title “Mr.” These individuals are typically traveling alone, as indicated by very low values for both SibSp (~0.21) and Parch (~0.11), and are predominantly in the third class with the lowest observed fares (~2.7). This cluster has the lowest survival rate (approximately 16%), making it the most vulnerable group in the dataset. The results strongly reflect historical patterns of evacuation, where adult men - especially those of lower socioeconomic status—were given the lowest priority for lifeboat access. The combination of being male, lower class, and traveling alone appears to significantly reduce survival likelihood, making this cluster a clear representation of the most disadvantaged passenger group.

**Cluster 1**

Cluster 1 consists entirely of female passengers with the title “Mrs,” indicating married women. These passengers are older on average (around 34 years old) and typically travel in small family groups (FamilySize ~2.3, SibSp ~0.7, Parch ~0.59). They are more likely to belong to first or second class and tend to have higher fares (Fare ~3.48), reflecting higher socioeconomic status. This cluster has the highest survival rate (approximately 83%), demonstrating the strong combined effect of gender and class advantage. The findings align with the “women and children first” evacuation policy, as well as the increased access to lifeboats often afforded to higher-class passengers. This group represents one of the most protected populations on the Titanic, where both social norms and economic status contributed to significantly better survival outcomes.

**Cluster 2**

Cluster 2 is composed entirely of female passengers, mostly young (average Age ~23) with ~99% holding the title “Miss,” indicating unmarried women. These passengers typically travel in smaller family groups, with moderate values for SibSp (~0.51) and Parch (~0.48). Their fares fall within a mid-range (~3.15), and they are distributed across passenger classes rather than concentrated in the highest class. Despite having slightly fewer advantages than married women in Cluster 1, this group still achieves a high survival rate (approximately 72%). This suggests that gender played a dominant role in survival, even when socioeconomic status was more variable. The slightly lower survival rate compared to married women may reflect differences in age, social status, or access to resources during evacuation.

Further, a boxplot on the fare distribution across each group may help reveal the length of travel or class of each cluster. Fare varies noticeably by cluster, with Cluster 0 showing the lowest and most concentrated fares, while Clusters -1 and 1 have higher medians and greater spread; however, high-fare outliers appear in all clusters, indicating that expensive tickets are not confined to a single group.


#### **Integrated Holistic Summary of Results**
Across K-Means, hierarchical clustering, and DBSCAN, the results consistently reveal three stable passenger archetypes that structure survival outcomes: (1) low-survival adult men in lower classes, (2) high-survival female passengers, and (3) moderate-survival children embedded in family groups. While each algorithm partitions the data differently, these groups remain highly consistent across all models, indicating strong underlying structure in the dataset.

The most robust and clearly defined pattern is the low-survival male, third-class cluster, which appears in all three methods with nearly identical characteristics. Survival in this group ranges from 11% (K-Means) to 16% (hierarchical and DBSCAN). These clusters are consistently defined by male dominance (>95% Title_Mr), and third class representation (mean class ~2.3-2.8), and small family sizes (~1.1-1.4). This stability across methods confirms adult men in lower socioeconomic groups formed the most disadvantaged survival category, aligning with prior historical findings (Youssefkhaled117, n.d.). 

In contrast, all models consistently identify high-survival female clusters, with survival rates ranging from 72%-83%. These clusters vary slightly by marital status and class but remain structurally similar across methods. Married first-class women consistently show the highest survival (~82-83%, mean age ~34-35, Pclass~ 1.9-2.0), while younger unmarried women show slightly lower but high survival (~72-77%, mean age 23-24). This reinforces the strong gender-based survival advantage.

The third consistent structure is the child and family-based cluster, which shows more variability across methods but remains clearly identifiable in all models. Survival rates range from 43% (DBSCAN noise cluster) to 57% (K-Means) to approximately 50% (hierarchical). These clusters are characterized by very young mean ages (~6-10), larger family sizes (~4-6 members), and mixed class composition. These results suggest that while children benefitted from evacuation priority, this advantage was partially offset by the constraints of traveling in larger family groups, which likely reduced mobility during evacuation.

Overall, the consistency of these three core structures across all clustering methods indicates that the dataset has strong and interpretable latent organization. While these core patterns remain stable, the clustering methods differ in how finely they segment the population. K-Means produces the most detailed partitioning, separating female passengers into multiple subgroups based on age and marital status. Hierarchical clustering instead produces broader, more aggregated social groups, while DBSCAN uniquely identifies a smaller “noise” cluster that captures atypical passengers with mixed characteristics (mean survival ~43%, mean age ~16, higher family size variability), isolating irregular or low-density cases. Importantly, these differences affect granularity rather than interpretation. All methods converge on the same conclusion: survival was primarily shaped by gender, secondarily reinforced by class, and further modified by family structure, forming stable social survival profiles rather than independent effects. 


### Recommendations and implications of your report. 
This analysis suggests that survival outcomes in the Titanic dataset were strongly influenced by underlying social structure and physical positioning rather than occurring randomly. While the findings are specific to this historical dataset, they offer useful insights into how demographic and socioeconomic factors can interact to shape outcomes in constrained emergency situations.

From a historical perspective, the results reinforce that evacuation outcomes were closely tied to social hierarchy; access to safety resources was not uniform. Clusters characterized by higher class status, female passengers, and certain titles show higher survival rates, reflecting the influence of early 20th century norms, social class, and prioritization practices during evacuation. 

In terms of modern implications, the analysis suggests several directions for improving emergency preparedness and system design. First, it emphasizes the importance of ensuring equitable access to exits and safety resources in buildings, transportation systems, and public venues, particularly under high-stress evacuation conditions. Second, it supports the use of data-driven approaches, such as clustering, to identify vulnerable groups that may be disproportionately impacted during emergencies. Finally, it highlights the value of incorporating demographic considerations into safety planning to reduce structural disadvantages during evacuation scenarios.

A key actionable direction from this work is the application of this analytical framework to other disaster-related datasets, such as building evacuation records, transportation accidents, wildfires, or public health emergencies. For example, similar patterns could be examined in historical maritime shipwrecks such as the Lusitania, which provides a potential comparative case for investigating how differences in time pressure and evacuation conditions may shape survival outcomes and the relative influence of social structure (Frey et al., 2010), or in modern healthcare triage systems, where clustering is used to prioritize patients based on risk profiles (Kim et al., 2025). Applying clustering in these contexts could help identify vulnerability patterns and improve risk stratification beyond what is visible through aggregate statistics alone. This approach could support more targeted safety planning, improve emergency preparedness strategies, and guide resource allocation towards groups and conditions associated with higher risk.

Overall, while the Titanic dataset reflects a specific historical event, the patterns identified provide a useful framework for thinking about how social and physical structures can influence survival outcomes. These insights can inform both retrospective understanding of historical disasters and future improvements in modern safety and emergency response systems.
