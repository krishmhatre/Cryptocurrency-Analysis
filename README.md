# Analyzing the Relationship of Cryptocurrencies with Foriegn Exchange Rates and Global Stock Market Indices

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-332/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-332/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-332/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-332/actions)
Status: final, Type: Project

* Krish Hemant Mhatre

* [fa20-523-332](https://github.com/cybertraining-dsc/fa20-523-332/)
* [Edit](https://github.com/cybertraining-dsc/fa20-523-332/blob/main/project/project.md)
* <krishmhatre@icloud.com>

{{% pageinfo %}}

## Abstract

The project involves analyzing the relationships of various cryptocurrencies with Foreign Exchange Rates and Stock Market Indices. Apart from analyzing the relationships, the objective of the project is also to estimate the trend of the cryptocurrencies based on Foreign Exchange Rates and Stock Market Indices. We will be using historical data of 6 different cryptocurrencies, 25 Stock Market Indices and 22 Foreign Exchange Rates for this project. The project will use various machine learning tools for analysis. The project also uses a fully connected deep neural network for prediction and estimation. Apart from analysis and prediction of prices of cryptocurrencies, the project also involves building its own database and giving access to the database using a prototype API. The historical data and recent predictions can be accessed through the public API. 

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** cryptocurrency, stocks, foreign exchange rates.


## 1. Introduction

The latest type of investment in the finance world and one of the latest global medium of exchange is Cryptocurrency. The total market capitalizations of all cryptocurrencies added up to $237.1 Billion as of 2019[^1], making it one of the fastest growing industries in the world. Cryptocurrency systems do not require a central authority as its state is maintained through distributed consensus[^2]. Therefore, determining the factors affecting the prices of cryptocurrencies becomes extremely difficult. There are several factors affecting the prices of cryptocurrency like transaction cost, reward system, hash rate, coins circulation, forks, popularity of cryptomarket, speculations, stock markets, exchange rates, gold price, interest rate, legalization and restriction[^3]. This project involves studying and analysing the relationships of various cryptocurrencies with Foreign Exchange Rates and Stock Market Indices. Furthermore, the project also involves predicting the cryptocurrency price based on stock market indices and foreign exchange rates of the previous day. The project also involves development of a public API to access the database of the historical data and the predictions. 


## 2. Resources

**Table 2.1:** Resources

| **No.** | **Name** | **Version** | **Type** |     **Notes**     |
| :---  |    :----:    |    :----:    |    :----:    |  ---:  |
| 1. |  Python  | 3.6.9 | Programming language  |Python is a high-level interpreted programming language. |
| 2. |  MongoDB |  4.4.2 |  Database  | MongoDB is a NoSQL Database program that uses JSON-like documents.  | 
| 3. |  Heroku  | 0.1.4 |  Cloud Platform| Heroku is a cloud platform used for deploying applications. It uses a Git server to handle application repositories. |
| 4. |  Gunicorn  | 20.0.4  | Server Gateway Interface  | Gunicorn is a python web server gateway interface . It is mainly used in the project for running python applications on Heroku. |
| 5. |  Tensorflow  | 2.3.1 |  Python Library|  Tensorflow is an open-source machine learning library. It is mainly used in the project for training models and predicting results.|
| 6. |  Keras | 2.4.3|Python Library|Keras is an open-source python library used for interfacing with artificial neural networks. It is an interface for the Tensorflow library. |
| 7. | Scikit-Learn|0.22.2|Python Library|Scikit-learn is an open-source machine learning library featuring various algorithms for classification, regression and clustering problems.|
| 8. |Numpy|1.16.0|Python Library|Numpy is a python library used for handling and performing various operations on large multi-dimensional arrays.|
| 9. |Scipy|1.5.4|Python Library|Scipy is a python library used for scientific and technical computing. It is not directly used in the project but serves as a dependency for tensorflow.|
| 10.|Pandas|1.1.4|Python Library|Pandas is a python library used mainly for large scale data manipulation and analysis. |
|11.|Matplotlib|3.2.2|Python Library|Matplotlib is a python library used for graphing and plotting. |
|12.|Pickle|1.0.2|Python Library|Pickle-mixin is a python library used for saving and loading python variables.|
|13.|Pymongo|3.11.2|Python Library|Pymongo is a python library containing tools for working with MongoDB.|
|14.|Flask|1.1.2|Python Library|Flask is a micro web framework for python. It is used in the project for creating the API.|
|15.|Datetime|4.3|Python Library|Datetime is a python library used for handling dates as date objects.|
|16.|Pytz|2020.4|Python Library|Pytz is a python library used for accurate timezone calculations.|
|17|Yahoo Financials|1.6|Python Library|Yahoo Financials is an unofficial python library used for extracting data from Yahoo Finance website by web scraping.|
|18|Dns Python|2.0.0|Python Library|DNS python is a necessary dependency of Pymongo.|



## 3. Dataset

The project builds its own dataset by extracting the data from Yahoo Finance website using Yahoo Financial python library [^4]. The data includes cryptocurrency prices, stock market indices and foreign exchange rates from September 30 2015 to December 5 2020. The project uses historical data of 6 cryptocurrencies - Bitcoin (BTC), Ethereum (ETH), Dash (DASH), Litecoin (LTC), Monero (XMR) and Ripple (XRP), 25 stock market indices - S&P 500 (USA), Dow 30 (USA), NASDAQ (USA), Russell 2000 (USA), S&P/TSX (Canada), IBOVESPA (Brazil), IPC MEXICO (Mexico), Nikkei 225 (Japan), HANG SENG INDEX (Hong Kong), SSE (China), Shenzhen Component (China), TSEC (Taiwan), KOSPI (South Korea), STI (Singapore), Jakarta Composite Index (Indonesia), FTSE Bursa Malaysia KLCI (Malaysia), S&P/ASX 200 (Australia), S&P/NZX 50 (New Zealand), S&P BSE (India), FTSE 100 (UK), DAX (Germany), CAC 40 (France), ESTX 50 (Europe), EURONEXT 100 (Europe), BEL 20 (Belgium), and 22 foreign exchange rates - Australian Dollar, Euro, New Zealand Dollar, British Pound, Brazilian Real, Canadian Dollar, Chinese Yuan, Hong Kong Dollar, Indian Rupee, Korean Won, Mexican Peso, South African Rand, Singapore Dollar, Danish Krone, Japanese Yen, Malaysian Ringgit, Norwegian Krone, Swedish Krona, Sri Lankan Rupee, Swiss Franc, New Taiwan Dollar, Thai Baht. This data is, then, posted to MongoDB Database. The three databases are created for each of the data types - Cryptocurrency prices, Stock Market Indices and Foreign Exchange Rates. The three databases each contain one collection for every currency, index and rate respectively. These collections have a uniform structure containing 6 columns - "id", "formatted_date", "low", "high", "open" and "close". The tickers used to extract data from Yahoo Finance [^4] are stated in Figure 3.1.

![Ticker Information](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/tickers.png)

**Figure 3.1:** Ticker Information 

The data is, then, preprocessed to get only one column per date ("close" price) and to add missing information by replicating previous day's values, which is used to make a large dataset including the prices of all indices and rates for all the dates within the given range. This data is saved in a different MongoDB Database and collection, both, called nn_data. This collection has 54 columns containing closing prices for each cryptocurrency price, stock market index and foreign exchange rate and the date. The rows represent different dates. 

One additional database is also created - Predictions - which contain the predictions of cryptocurrency prices for each day and it's true value. The collection has 13 columns containing a date column and 2 columns for each cryptocurrency (prediction value and true value). New rows are inserted everyday for all collections except the "nn_data" collection. Figure 3.2 represents the overview of the MongoDB Cluster. Figure 3.3 shows the structure of the nn_data collection.

![MongoDB Cluster Overview](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/mongodb.png)

**Figure 3.2:** MongoDB Cluster Overview

![Short Structure of NN_data Collection](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/sample_data.png)

**Figure 3.3:** Short Structure of NN_data Collection


## 4. Analysis

### 4.1 Principal Component Analysis

Principal Component Analysis uses Singular Value Decomposition (SVD) for dimensionality reduction, exploratory data analysis and making predictive models. PCA helps understand a linear relationship in the data[^5]. In this project, PCA is used for the preliminary analysis to find a pattern between the target and the features. Here we have tried to make some observations by performing PCA on various cryptocurrencies with stocks and forex data. In this analysis, we reduced the dimension of the dataset to 3D, represented in Figure 4.1. The first and second dimension is on x-axis and y-axis respectively whereas the third dimension is used in the color. On observing the scatter plots in Figure 4.1, we can clearly see the patterns formed by various relationships. Therefore, it can be stated that the target and features are related in some way based on the principal component analysis. 

![Principal Component Analysis](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/pca.png)

**Figure 4.1:** Principal Component Analysis

### 4.2 TSNE Analysis

T-Distributed Stochastic Neighbour Embedding is mainly used for non-linear dimensionality reduction. TSNE uses local relationships between points to create a low-dimensional mapping. TSNE uses Gaussian distribution to  create a probability distribution. In this project, TSNE is used to analyze non-linear relationships between cryptocurrencies and the features (stock indices and forex rates), which were not visible in the principal component analysis. It can be observed in Figure 4.2, that there are visible patterns in the data i.e. same colored data points are in some pattern, proving a non linear relationship. The t-SNE plots in Figure 4.2 are not like the typical t-SNE plots i.e. they do not have any clusters. This might be because of the size of the dataset. 

![t-SNE Analysis](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/tsne.png)

**Figure 4.2:** t-SNE Analysis

### 4.3 Weighted Features Analysis

Layers of neural networks have weights assigned to each feature column. These weights are updated continuously while training. Analyzing the weights of the model which is trained for this project, can give us a picture of the important features. To perform such an analysis, the top five feature weights are noted for each layer. The number of times a feature is present in the top five of a layer, is also noted. This is represented in Figure 4.3, where we can observe that the New Zealand Dollar and the Canadian Dollar are repeated most number of times in the top five weights of layers. 

![No. of repetitions in top five weights](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/TOP_WEIGTHS.png)

**Figure 4.3:** No. of repetitions in top five weights

The relationships of these two features - New Zealand Dollar and Canadian Dollar with various cryptocurrencies are, then, analyzed in Figure 4.4 and Figure 4.5. It can be observed that Bitcoin has a direct relationship with these rates. Bitcoin can be observed to increase with an increase in NZD to USD rate and an increase in CAD to USD rate. For the rest of the cryptocurrencies, we can observe that they tend to rise when the NZD to USD rate and the CAD to USD rate are stable and tend to fall when the rates move towards either of the extremes. 

![Relationship of NZD with Cryptocurrencies](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/nz_vs_crypto.png)

**Figure 4.4:** Relationship of NZD with Cryptocurrencies

![Relationship of CAD with Cryptocurrencies](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/cad_vs_crypto.png)

**Figure 4.5:** Relationship of CAD with Cryptocurrencies

## 5. Neural Network

### 5.1 Data Preprocessing

The first step to build a neural network for predicting cryptocurrency prices, is to clean the data. In this step, data from the "NN_data" collection is imported. Two scalers are used to normalize the data, one for feature columns and other for the target columns. For this purpose, "StandardScaler" from Scikit-learn library is used. These scalers are made to fit with the data and then saved to a file using pickle-mixin, in order to use it later for predictions. These scalers are then used to normalize the data using mean and standard deviation. This normalized data is shuffled and split into a training set and a test set. This procedure is done by using the "train_test_split()" function from the Scikit-learn library. The data is split into 94:6 ratio for training and testing respectively. The final data is split into four - X_train, X_test, y_train and y_test and is ready for training the neural network model.



### 5.2 Model

For the purpose of predicting the prices of cryptocurrency based on previous day’s stock indices and forex rates, the project uses a fully connected neural network. The solution to this problem could have been perceived in different ways like making it a classification problem by predicting rise or fall in price or by making it a regression problem by either predicting the actual price or by predicting the growth. After trying all these ways of solution, it was concluded that predicting the price regression problem was the best option. 

The final model comprises three layers - one input layer, one hidden layer and one output layer. The first layer uses 8 units with an input dimension of (None, 47) and uses Rectified Linear Unit (ReLU) as its activation function, and He Normal as its kernel initializer. The second layer which is a hidden layer uses 2670 hidden units with Rectified Linear Unit (ReLU) Activation function. ReLU is used because of its faster and effective training in regression models. The third layer which is the output layer has 6 units, one each for predicting 6 cryptocurrencies. The output layer uses linear activation function. 

The overview of the final model can be seen in Figure 5.2. The predictions using the un-trained model can be seen in Figure 5.3, where we can observe the initialization of weights. 

![Model Overview](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/model.png)

**Figure 5.2:** Model Overview

![Visualization of Initial Weights](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/intialize.png)

**Figure 5.3:** Visualization of Initial Weights

### 5.3 Training

The neural network model is compiled before training. The model is compiled using Adam optimizer with a default learning rate of 0.001. The model uses Mean Squared Error as its loss function in order to reduce the error and give a close approximation of the cryptocurrency prices. Mean squared error is also used as a metric to visualize the performance of the model.

The model is, then, trained by using X_train and y_train, as mentioned above, for 5000 epochs and by splitting the dataset for validation (20% for validation). The performance of the training of the final model for first 2500 epochs can be observed in Figure 5.4.

![Final Model Training](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/dnn_with_normal_init.png)

**Figure 5.4:** Final Model Training

This particular model was chosen because of its low validation mean squared error as compared to the performance of other models. Figure 5.5 represents the performance of a similar fully connected model with Random Normal as its initializer instead of He Normal. Figure 5.6 represents the performance of a Convolutional Neural Network. This model was trained with a much lower mean squared error but had a higher validation mean squared error and was therefore dropped. 

![Performance of Fully Connected with Random Normal](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/dnn_with_random_normal_init.png)

**Figure 5.5:** Performance of Fully Connected with Random Normal

![Performance of Convolutional Neural Network](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/cnn.png)

**Figure 5.6:** Performance of Convolutional Neural Network


### 5.4 Prediction

After training, the model is stored in a .h5 file, which can be used to make predictions. For making predictions, the project preprocesses the data provided which needs to be of the input dimension of the model i.e. of shape (1, 47). Both the scalers which were saved earlier in the preprocessing stage are loaded again using pickle-mixin. The feature scaler is used to transform the new data to normalized data. This normalized data of the given dimension is then used to predict the prices for six cryptocurrencies. Since regression models do not show accuracy directly, it can be measured manually by rounding off the predicted values and the corresponding true values to the decimal place of one or two and then getting the difference between the two and comparing it to a preset threshold. If the values are rounded off to one decimal place and the threshold is set to 0.05 on the normalized predictions, the accuracy of the prediction is approximately 88% and if the values are rounded off to two decimal places, the accuracy is approximately 62%. The predictions of the test data and the corresponding true values for Bitcoin can be observed in Figure 5.7, where similarities can be observed. Prediction for a new date for the prices of all six cryptocurrencies and its true values can be observed in Figure 5.8. Figure 5.9 also displays the actual result of this project as it can be observed that the predictions and the true values have similar trend with a low margin of error.

![Prediction vs. True](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/prediction.png)

**Figure 5.7:** Prediction vs. True

![Prediction vs. True for one day’s test data](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/pred.png)

**Figure 5.8:** Prediction vs. True for one day’s test data

![Prediction vs. True for all cryptocurrencies](https://github.com/cybertraining-dsc/fa20-523-332/raw/main/project/images/pred_vs_true.png)

**Figure 5.9:** Prediction vs. True for all cryptocurrencies



## 6. Deployment

### 6.1 Daily Update

The database is supposed to be updated daily using a web-app deployed on Heroku. Heroku is a cloud platform used for deploying web-apps of various languages and also uses a Git-server for repositories [^7]. This daily update web-app is triggered daily at 07.30 AM UTC i.e 2.00 AM EST. The web-app extracts the data for the previous day and updates all the collections. The new data is then preprocessed by using the saved feature normalizer. This normalized data is used to get predictions for the prices of cryptocurrencies for the day that just started. The web-app then gets the true values of the cryptocurrency prices for the previous day and updates the predictions collection using this data for future comparison. The web-app is currently deployed on Heroku and is triggered daily using Heroku Scheduler. The web-app is entirely coded in Python.

### 6.2 REST Service

The data from the MongoDB databases can be accessed using a public RESTful API. The API is developed using Flask-Python. The API usage is given below.

URL - ```https://crypto-project-api.herokuapp.com/```

----
***/get_data/single/market/index/date***

*Type - GET*

Sample Request - 

```https://crypto-project-api.herokuapp.com/get_data/single/crypto/bitcoin/2020-12-05```

Sample Respose - 

```
{
  "data":
  [
    {
      "close":19154.23046875,
      "date":"2020-12-05",
      "high":19160.44921875,
      "low":18590.193359375,
      "open":18698.384765625
    }
  ],
  "status":"Success"
}
```

----

***/get_data/multiple/market/index/start_date/end_date***

*Type - GET*

Sample Request - 

```https://crypto-project-api.herokuapp.com/get_data/multiple/crypto/bitcoin/2020-12-02/2020-12-05```

Sample Respose - 

```
{
  "data":
  [
    {
      "close":"19201.091796875",
      "date":"2020-12-02",
      "high":"19308.330078125",
      "low":"18347.71875",
      "open":"18801.744140625"
    },
    {
      "close":"19371.041015625",
      "date":"2020-12-03",
      "high":"19430.89453125",
      "low":"18937.4296875",
      "open":"18949.251953125"
    },
    {
      "close":19154.23046875,
      "date":"2020-12-05",
      "high":19160.44921875,
      "low":18590.193359375,
      "open":18698.384765625
    }
  ],
  "status":"Success"
}
```
----

***/get_predictions/date***

*Type - GET*

Sample Request - 

```https://crypto-project-api.herokuapp.com/get_predictions/2020-12-05```

Sample Respose - 

```
{
  "data":
    [
      {
        "bitcoin":"16204.04",
        "dash":"24.148237",
        "date":"2020-12-05",
        "ethereum":"503.43005",
        "litecoin":"66.6938",
        "monero":"120.718414",
        "ripple":"0.55850273"
      }
    ],
  "status":"Success"
}
```

----
## 7. Conclusion

After analyzing the historical data of Stock Market Indices, Foreign Exchange Rates and Cryptocurrency Prices, it can be concluded that there does exist a non-linear relationship between the three. It can also be concluded that cryptocurrency prices can be predicted and its trend can be estimated using Stock Indices and Forex Rates. There is still a large scope of improvement in reducing the mean squared error. The project can further improve the neural network model for better predictions. In the end, it is safe to conclude that the indicators of international politics like Stock Market Indices and Forex Exchange Rates are factors affecting the prices of cryptocurrency. 

## 8. Acknowledgement

Krish Hemant Mhatre would like to thank Indiana University and Luddy School of Informatics, Computing and Engineering for providing me with the opportunity to work on this project. He would also like to thank Dr. Geoffrey C. Fox, Dr. Gregor von Laszewski and the Assistant Instructors of ENGR-E-534 Big Data Analytics and Applications for their constant guidance and support. 


## References

[^1]: Szmigiera, M. "Cryptocurrency Market Value 2013-2019." Statista, 20 Jan. 2020, <https://www.statista.com/statistics/730876/cryptocurrency-maket-value>.

[^2]: Lansky, Jan. "Possible State Approaches to Cryptocurrencies." Journal of Systems Integration, University of Finance and Administration in Prague Czech Republic, <http://www.si-journal.org/index.php/JSI/article/view/335>.

[^3]: Sovbetov, Yhlas. "Factors Influencing Cryptocurrency Prices: Evidence from Bitcoin, Ethereum, Dash, Litcoin, and Monero." Journal of Economics and Financial Analysis, London School of Commerce, 26 Feb. 2018, <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3125347>.

[^4]: Sanders, Connor. "YahooFinancials." PyPI, JECSand, 22 Oct. 2017, <https://pypi.org/project/yahoofinancials/>.

[^5]: Jaadi, Zakaria. "A Step-by-Step Explanation of Principal Component Analysis." Built In, <https://builtin.com/data-science/step-step-explanation-principal-component-analysis>.

[^6]: Violante, Andre. "An Introduction to t-SNE with Python Example." Medium, Towards Data Science, 30 Aug. 2018, <https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1>.

[^7]: "What Is Heroku." Heroku, <https://www.heroku.com/what>.


