#**Logistic Regression – Airline Customer Satisfaction Prediction**

**Project Overview**
In this project, you will analyze passenger survey data using Python and Scikit-learn to build a Logistic Regression classification model. You will learn how to engineer features, encode categorical variables, split data for training/testing, implement binomial logistic regression, and evaluate model performance using confusion matrices, precision, and recall. This project helps you translate classification results into actionable business strategies for improving customer retention.

### Import Data Files from Google Drive
import requests
import pandas as pd
from io import StringIO
def read_gd(sharingurl):
    file_id = sharingurl.split('/')[-2]
    download_url='https://drive.google.com/uc?export=download&id=' + file_id
    url = requests.get(download_url).text
    csv_raw = StringIO(url)
    return csv_raw

url = "https://drive.google.com/file/d/1JuNVOWDOeyKdx-3eWd5v1yk8i6llrTKA/view?usp=sharing"
gdd = read_gd(url)

df = pd.read_csv(gdd)

## Encode categorical predictors into a format suitable for Scikit-**learn**
### Converting all the categorical variables to dummy variables
dfCat = pd.get_dummies(df[['Customer Type','Type of Travel','Class']])
dfCat

### Seperating the numerical variables
dfNum = df[['Age','Flight Distance','Seat comfort','Departure/Arrival time convenient','Food and drink','Gate location','Inflight wifi service','Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]
dfNum.shape

### Preparing the X variables
X = pd.concat([dfCat, dfNum], axis=1)
print(X.shape)
### Preparing the Y variable
Y = df['satisfaction']
print(Y.shape)
X.head()

## Split data into training and testing sets for unbiased model evaluation
### Splitting the data into train and test sets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

## Build a binomial Logistic Regression model to predict probability of "Satisfied"
### Defining the LogisticRegression function
model = LogisticRegression()
model.fit(X_train, y_train)

## Error Metrics
pred = model.predict(X_test)
print('Accuracy of Logisticr regression model prediction on test set: {:.2f}'.format(model.score(X_test, y_test)))

Accuracy of Logisticr regression model prediction on test set: 0.79

## Generate a Confusion Matrix and calculate Precision and Recall (not just Accuracy)
# Confusion Matrix for the model
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test, pred)
print(confusionMatrix)

[[12337  5261]
 [ 3101 18265]]

 from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
precision    recall  f1-score   support

dissatisfied       0.80      0.70      0.75     17598
   satisfied       0.78      0.85      0.81     21366

    accuracy                           0.79     38964
   macro avg       0.79      0.78      0.78     38964
weighted avg       0.79      0.79      0.78     38964

<img width="688" height="554" alt="log_confusion_matrix" src="https://github.com/user-attachments/assets/6ad6c77a-3d2d-4d93-8de1-416c067dcc34" />

The confusion matrix provides a detailed breakdown of the model's performance beyond just overall accuracy. Let's interpret it:

True Negatives (TN): 12,337
These are instances where the model correctly predicted that a customer would be dissatisfied.
False Positives (FP): 5,261
These are instances where the model incorrectly predicted that a customer would be satisfied, but they were actually dissatisfied. This is a Type I error. For the airline, this means they might miss opportunities to intervene and improve the experience for these 5,261 customers, as they erroneously believe these customers are satisfied.
False Negatives (FN): 3,101
These are instances where the model incorrectly predicted that a customer would be dissatisfied, but they were actually satisfied. This is a Type II error. For the airline, this means they might unnecessarily focus resources on trying to improve the experience for these 3,101 already satisfied customers.
True Positives (TP): 18,265
These are instances where the model correctly predicted that a customer would be satisfied.
From the classification report, we also see:

Precision for 'dissatisfied' (0.80): When the model predicts 'dissatisfied', it is correct 80% of the time.
Recall for 'dissatisfied' (0.70): The model correctly identifies 70% of all truly 'dissatisfied' customers.
Precision for 'satisfied' (0.78): When the model predicts 'satisfied', it is correct 78% of the time.
Recall for 'satisfied' (0.85): The model correctly identifies 85% of all truly 'satisfied' customers.
The model is better at identifying truly 'satisfied' customers (higher recall for 'satisfied') but has a higher rate of incorrectly predicting 'satisfied' when the customer is actually 'dissatisfied' (5,261 False Positives). Depending on the business goals, the airline might want to reduce false positives (avoiding misidentifying unhappy customers as happy) or false negatives (avoiding misidentifying happy customers as unhappy).

## Interpret model coefficients to identify key drivers of satisfaction
	Feature	Coefficient	Absolute_Coefficient
14	Inflight entertainment	0.724455	0.724455
5	Class_Eco	-0.438176	0.438176
1	Customer Type_disloyal Customer	-0.365942	0.365942
4	Class_Business	0.308486	0.308486
13	Inflight wifi service	-0.264628	0.264628
16	Ease of Online booking	0.247986	0.247986
12	Gate location	-0.243098	0.243098
10	Departure/Arrival time convenient	-0.239183	0.239183
9	Seat comfort	0.223689	0.223689
17	On-board service	0.206083	0.206083
3	Type of Travel_Personal Travel	-0.193145	0.193145
0	Customer Type_Loyal Customer	0.175438	0.175438
21	Cleanliness	-0.128615	0.128615
18	Leg room service	0.127961	0.127961
15	Online support	0.116163	0.116163
19	Baggage handling	-0.095218	0.095218
20	Checkin service	0.064408	0.064408
6	Class_Eco Plus	-0.060814	0.060814
7	Age	-0.026202	0.026202
11	Food and drink	-0.012608	0.012608
24	Arrival Delay in Minutes	-0.008070	0.008070
23	Departure Delay in Minutes	0.003040	0.003040
22	Online boarding	-0.002954	0.002954
2	Type of Travel_Business travel	0.002641	0.002641
8	Flight Distance	-0.000430	0.000430

Features like Inflight entertainment, Class (Business vs. Eco), and Customer Type have the most significant impact on satisfaction, with higher positive coefficients indicating a strong positive correlation with satisfaction and higher negative coefficients indicating a strong negative correlation.

## Business Recommendations for Improving Customer Satisfaction

Based on the logistic regression model's coefficients, here are key recommendations for the airline:

1.  **Prioritize Inflight Entertainment:** 'Inflight entertainment' is the strongest positive driver of satisfaction. Investing in high-quality and diverse entertainment options (e.g., more movie choices, interactive games, up-to-date content) should be a top priority. Promoting these offerings prominently can also boost perceived value.

2.  **Enhance Economy Class Experience:** The 'Eco' class is a significant negative driver of satisfaction. While cost-effectiveness is a factor, the airline should look for ways to improve the basic economy experience without significantly increasing costs. This could include:
    *   **Seat Comfort:** Even small improvements in seat ergonomics or providing basic comfort amenities (e.g., better headrests, lumbar support) could make a difference, as 'Seat comfort' is also a positive driver.
    *   **Food and Drink:** Although a small coefficient, ensuring decent quality and variety in complimentary food and drink in Eco class could help mitigate dissatisfaction.

3.  **Cultivate Loyal Customers:** 'Loyal Customer' status is a strong positive driver, while 'disloyal Customer' is a strong negative driver. The airline should focus on:
    *   **Loyalty Programs:** Enhance and actively promote loyalty programs with tangible benefits (e.g., upgrades, priority boarding, exclusive offers).
    *   **Customer Relationship Management (CRM):** Use data to identify and address the pain points of 'disloyal Customers' to convert them into loyal ones. Personalized communication and problem resolution are crucial.

4.  **Leverage Business Class:** 'Business Class' is a strong positive driver. The airline should continue to differentiate and market its Business Class offerings effectively, ensuring the premium experience justifies the cost. Consider enhancing aspects that contribute to Business Class satisfaction, such as 'On-board service' and 'Leg room service'.

5.  **Re-evaluate Inflight Wi-Fi Service:** 'Inflight wifi service' is a negative driver, suggesting that the current service might be unreliable, slow, or expensive. The airline should:
    *   **Improve Connectivity:** Invest in better technology to ensure consistent and faster internet speeds.
    *   **Pricing Strategy:** Reconsider the pricing model for Wi-Fi. Offering free basic access or more affordable premium options could turn this into a positive factor.

6.  **Optimize Online Booking and Support:** 'Ease of Online booking' and 'Online support' are positive drivers. Continue to streamline the online booking process and ensure that online support channels are efficient and responsive.

7.  **Address Departure/Arrival Time Convenience and Gate Location:** These are negative drivers. While not always fully controllable, the airline should:
    *   **Schedule Optimization:** Work to minimize inconvenient departure/arrival times where possible.
    *   **Communication:** Provide clear and timely information regarding gate changes or potential delays to minimize passenger frustration.

By focusing on these key areas, the airline can strategically allocate resources to maximize customer satisfaction and potentially improve retention and revenue.

