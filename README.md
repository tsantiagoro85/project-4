## Predicting Obesity using Tensorflow Deep Learning
This analysis aims to predict obesity or overweight status based on various lifestyle and demographic features using deep learning models. The dataset was downloaded from https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition. The dataset was then processed using PySpark for initial data handling, followed by the use of a neural network built with TensorFlow. The analysis is conducted in multiple stages, focusing on a mix of different features.

## Compiling, Training, and Evaluating the Model
*Data Loading and Preparation:*

The dataset is loaded using PySpark, and relevant columns are selected for analysis.
Categorical data, specifically the 'Obese/Overweight' label, is encoded using LabelEncoder to convert it into a numerical format suitable for model training.
The features (X) and target (y) variables are defined for each feature set.

*Data Scaling:*

The data is split into training and testing sets using train_test_split.
Standardization is applied to the features using StandardScaler to ensure that the neural network can learn effectively.

*Model Training:*

A sequential neural network model is defined with two hidden layers and one output layer for each feature set.
The model is compiled with the Adam optimizer and binary cross-entropy loss function.
The model is trained on the scaled training data for 100 epochs, with validation split to monitor performance.

*Model Evaluation:*

After training, the model is evaluated on the test set to determine its accuracy. Many features have accomplished the target accuracy percentage of 75%, whilst some do not. However, those features that do not, are within 2% of the desired accuracy. 
It can be concluded that these features contribute less to the obesity factors and are in agreement that obesity is multifactorial in nature.

*Improving Accuracy:*

Several actions were taken to improve the accuracy of the model:
* Experimenting with different activation functions (e.g., ReLU, sigmoid).
* Adjusting the number of neurons and layers to find the optimal architecture.
* Implementing dropout layers to reduce overfitting.
* Increased the number of epochs during training to improve learning.

There are additional actions that could be implemented in the future to further improve accuracy, such as:
* Explore additional features that may more directly influence obesity.
* Increase the data sample size.
* Use a more advanced learning model.

## Compiled Results: Neural Network Accuracies
Numerical Variables
* Height: 73.52% (Loss: 0.5550)
* Age: 75.89% (Loss: 0.4354)
* Meals Per Day: 77.78% (Loss: 0.4906)
* Frequency of Physical Activity: 72.10% (Loss: 0.5392)
* Water Intake: 73.52% (Loss: 0.4813)
* Vegetable Intake: 73.52% (Loss: 0.5614)
* Technology Use: 73.52% (Loss: 0.5595)

Categorical Variables
* Food Between Meals: 83.69% (Loss: 0.4374)
* Mode of Transportation: 75.67% (Loss: 0.5533)
* Alcohol Intake: 75.15% (Loss: 0.5554)
* Smoking Status: 75.15% (Loss: 0.5619)
* Family History with Overweight: 83.69% (Loss: 0.5327)
* High Caloric Food: 75.15% (Loss: 0.5647)
* Monitor Calories: 75.15% (Loss: 0.5647)
* Gender: 75.15% (Loss: 0.5647)

## Results Summary
* Best Performers: The categorical variables "Food Between Meals" and "Family History with Overweight" demonstrated the highest accuracy (83.69%), indicating they are strong predictors of the outcome.
* Numerical Variables: "Meals Per Day" had the best performance among numerical variables (77.78%), suggesting it is a significant factor.
* Moderate Predictive Power: Other variables (both numerical and categorical) generally showed moderate accuracy (around 72-77%), indicating that while they contribute to the model, there may be opportunities for feature engineering or data collection improvements to enhance predictive performance.
* Loss Values: The loss values provide insight into the model's error; lower loss values (especially for age and food between meals) suggest better model performance.

## Random Forest Classifier
To rank the variables in order of importance, a Random Forest Classifier was trained using the obesity dataset, where the target variable is whether an individual is "Obese/Overweight" or not.
The dataset was split into training (80%) and testing (20%) sets using the train_test_split method.

*Feature Importance:*

The model was used to extract feature importance scores, which indicate how much each feature contributes to the model's predictions.
The features were ranked by their importance, with the following being the top contributors:
* Food Between Meals: 0.165562
* Age: 0.156754
* Family History with Overweight: 0.132402
* Height: 0.087969
* Physical Activity Frequency: 0.084680

Notably, "Food Between Meals", "Family History with Overweight" and "Age" were also among the categories with the highest accuracy values when performing the neural network analysis.

*Visualization:*

A scatter plot was created to visualize the importance of each feature. This plot helps in understanding which features have the most significant impact on predicting obesity levels.

![random_forest_feature_importance](https://github.com/user-attachments/assets/5edb970b-744d-4bb6-bb47-2f6723caafed)

*Model Evaluation:*

While the summary does not include specific evaluation metrics (like accuracy, precision, recall, etc.), these can be calculated using the testing set predictions to assess the model's performance.  

## Conclusion
The results found using this model are crucial in understanding obesity risk factors. These results can be utilized to provide accurate public health strategies, and encourage individual lifestyle changes. While the model shows promise in predicting obesity based on lifestyle and demographic features, further exploration of additional relevant features and larger datasets could enhance its accuracy and utility.

