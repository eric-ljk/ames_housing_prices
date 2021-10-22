<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Project 2 - Ames Housing Data and Kaggle Challenge

--- 

# Executive Summary

Real estate agents rely on their access to information to help both buyers and sellers make informed decisions. Machine learning models can help integrate such information and provide predictions to further boost informational advantage. Available information on real-estate transactions in Ames, Iowa, was used to predict the sale price of a house based on the features of the property and the surrounding area such as its neighbourhood. Linear, Lasso and Ridge regression models were tested on the data after an extensive process of feature engineering. All 3 variants had similar performance, with the RidgeCV model selected as it scored higest on the evaluation metrics, as follows: The cross-validated R-squared score was 0.907, and the Root Mean Squared Error (RMSE) was 24,660. Based on comparisons between the training and testing portions of the data, the model was able to generalise well to new data and was not overfit to the training data. The results show a high ability for the model to explain the variability in the sale price with the given input features. The size of the living area, the overall quality of the house and the total number of baths in the property were found to be strongly positively correlated to the sale price. The number of full baths and the property being a Townhouse End Unit or a Townhouse Inside Unit were found to be negatively correlated with the sale price. This model provides a good basis in giving real estate agents additional information to do their job and maintain the balancing act between buyers and sellers. Buyers would be able to know if they are overpaying for their house, and sellers would be able to know if they are underpricing their house.

# Problem Statement

Real estate agents rely on their access to information to help both buyers and sellers make informed decisions.

This project will use available information on real-estate transactions in Ames, Iowa, to predict the sale price of a house based on the features of the property and the surrounding area such as its neighbourhood.

To do this, a linear regression model will be trained from the data, with regularisation and feature engineering where necessary. Regularisation will involve Lasso and Ridge regularisation. The best iteration will be selected, where the predictive power will be evaluated by its R-squared score and Root Mean Squared Error (RMSE). The R-squared score will be used as a threshold - the final model should have an R-squared score of at least 0.8.

By predicting sale price and revealing strong predictors, this project and the model will aid estate agents in helping buyers and sellers find the best price. Buyers can avoid over-paying for their house, and sellers can avoid under-pricing it. As a result, real estate agents have even more insights and information to perform their job.

# Background and Research

Ames is a city in the state of Iowa in the United States. It was ranked 15/100 in livability.com's "The 2020 Top 100 Best Places to Live in America", which is a "data-driven list" ([*source*](https://livability.com/best-places/the-2020-top-100-best-places-to-live-in-america/)). Iowa is also No. 1 in the "10 Best States to Retire in 2021 (MoneyRates, 2021)", among other accolades listed on the website of the City of Ames ([*source*](https://www.cityofames.org/about-ames/awards-accolades-achievements)).

Ames is known as a college town. Iowa State University has an enrollment of 30,708 alone ([*source*](https://www.registrar.iastate.edu/resources/enrollment-statistics)). This figure is already 46% of the city's population of 66,772 ([*source*](https://worldpopulationreview.com/us-cities/ames-ia-population)). The university is is the largest employer in Ames ([*source*](https://khak.com/ames-and-iowa-city-among-top-20-places-to-live-in-america/)). From the City of Ames website, Ames was ranked No. 1 in the "Best U.S. Job Market (CNBC, 2018)", and the No. 1 in "Top Cities for Career Opportunities in 2018 (SmartAsset, 2018)" ([*source*](https://www.cityofames.org/about-ames/awards-accolades-achievements)).

With these considered, it is not surprising that there would be a demand for real estate services in the city. 

Real estate agents need as much real estate data they can get to perform their job ([*source*](https://realtyna.com/blog/do-realtors-have-access-more-real-estate-data-than-public/)). Information advantages or asymmetries that real estate agents have can indeed translate to better deals when transacting properties. In this study of an urbanised city, Singapore, real estate agents were found to have purchased their own properties at 2.54% lower than when similar houses were bought by normal buyers ([*source*](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7378&context=lkcsb_research)). From the source, the factors giving the agents an advantage were information on the houses available, and the previous sale prices. These factors are similar to what we have in the Ames data.

The question, then is whether the current data can be used to build regression models. 

Hedonic pricing is a model that assumes the price of a good is affected by its inherent characteristics and related outside factors ([*source*](https://www.investopedia.com/terms/h/hedonicpricing.asp)). Hedonic regression, specifically, is the use of a regression model to study this ([*source*](https://www.investopedia.com/terms/h/hedonic-regression.asp)). The features found in the Ames dataset can be used as features for hedonic regression, including house characteristics ([*source*](https://www.researchgate.net/publication/5151851_The_Value_of_Housing_Characteristics_A_Meta_Analysis)) and neighbourhood details ([*source*](http://www.aessweb.com/pdf-files/31-44.pdf)). 

Hence, the dataset can potentially allow us to build a linear regression model. 

One of the columns is for lot frontage. There are minimum values for lot frontage specified in the City of Ames Municipal Code ([*source*](https://www.cityofames.org/government/departments-divisions-i-z/legal/city-of-ames-municipal-code)). 

# Data sources

The following files are the initial data for this project:

* train.csv: 2006-2010 Ames Housing Dataset (Training set split by Kaggle)
* test.csv: 2006-2010 Ames Housing Dataset (Testing set split by Kaggle)

The data contains information on house sales, where each row is an entry for the sale of a house and includes its associated features and information within the columns.

# Data Dictionary

The Data Dictionary is documented here ([*source*](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)) and is as follows:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**street**|object|train_reference|Type of road access to property.|
|**land_contour**|object|train_reference|Flatness of the property.|
|**lot_config**|object|train_reference|Lot configuration.|
|**neighborhood**|object|train_reference|Physical locations within Ames city limits.|
|**condition_1**|object|train_reference|Proximity to various conditions.|
|**bldg_type**|object|train_reference|Type of dwelling.|
|**house_style**|object|train_reference|Style of dwelling.|
|**overall_qual**|int64|train_reference|Rates the overall material and finish of the house.|
|**roof_style**|object|train_reference|Type of roof.|
|**roof_matl**|object|train_reference|Roof material.|
|**exterior_1st**|object|train_reference|Exterior covering on house.|
|**mas_vnr_type**|object|train_reference|Masonry veneer type.|
|**mas_vnr_area**|float64|train_reference|Masonry veneer area in square feet.|
|**exter_qual**|float64|train_reference|Evaluates the quality of the material on the exterior.|
|**foundation**|object|train_reference|Type of foundation.|
|**bsmt_qual**|float64|train_reference|Evaluates the height of the basement.|
|**total_bsmt_sf**|float64|train_reference|Total square feet of basement area.|
|**heating**|object|train_reference|Type of heating.|
|**heating_qc**|float64|train_reference|Heating quality and condition.|
|**central_air**|object|train_reference|Central air conditioning.|
|**gr_liv_area**|int64|train_reference|Above grade (ground) living area in square feet.|
|**full_bath**|int64|train_reference|Full bathrooms above grade.|
|**kitchen_qual**|float64|train_reference|Kitchen quality.|
|**fireplace_qu**|float64|train_reference|Fireplace quality.|
|**garage_type**|object|train_reference|Garage location.|
|**garage_finish**|float64|train_reference|Interior finish of the garage.|
|**sale_type**|object|train_reference|Type of sale.|
|**saleprice**|int64|train_reference|Sale price in dollars.|
|**house_age**|int64|train_reference|Age of the house in years.|
|**age_at_remod**|int64|train_reference|Age at which the house was remodelled, in years.|
|**total_baths**|float64|train_reference|Total number of baths in the property.|
|**garage_overall**|float64|train_reference|Interaction feature from the garage quality and the garage car capacity.|

# Results and Analysis

The Dummy Regressor, Linear Regressor, Lasso Regressor and Ridge Regressor were tested on the dataset after extensive feature engineering. There was a great improvement from the baseline model (Dummy Regressor). The baseline R-squared score was zero, as it simply predicts one value no matter the value of the features. The baseline RMSE showed that the dummy regressor was generally off-target by $74,506 in predicting the sale price.

Between Linear, Lasso and Ridge regression, the results of the calculated metrics were very close. This may be because of the thorough feature engineering which resulted in the removal of a great deal of columns. Lasso and Ridge would have effectively removed features due to the regularisation penalty. This might mean that the remaining features already contributed to the model in a way that does not require extensive optimisation through regularisation.

The models performed similarly between the train and test data sets with a relatively small difference in the scores. Hence, it can be concluded that the models were not overfitting to the training data, and they can generalise well to test data.

The cross-validated results were close to the test results which means that the test split was representative. Based on both the R-squared and RMSE values, the Ridge model had the highest R-squared values and lowest RMSE values in general. Hence, the Ridge CV model was selected as the final candidate. The model's R-Squared score of 0.907 showed a high ability to explain the variability in the sale price with the given input features. The model's RMSE showed that the model would be estimated to be off-target by $24,660, a difference of $49,846.

# Conclusion and Recommendations

The data cleaning and feature engineering process led to a functional model with an R-squared score higher than 0.8, thus surpassing the initial objective set in the problem statement. This model was able to generalise well to new data and was not overfit to the training data. The Ridge Regression Cross-Validated R-Squared value was 0.907, which meant that the model was able to explain 90% of the variability in sale price with the given set of input features. As compared to just guessing the mean sale price, the use of the model reduced that error by $49,846 and led to a final RMSE of $24,660. This would provide a good foundation for buyers to know if they are overpaying for their house, or if sellers are underpricing their house. Such a predictive service would be especially useful for real estate agents, who have to maintain the balancing act between these two groups, and who require as much information as they can get in order to have an advantage in their job. Thus, the primary audience of real estate agents would be likely to benefit from this model.

The coefficient values of the model provide valuable information as well. Several important features affecting the sale price were identified in this study, such as the size of the living area and the neighbourhood of the house. Such information could aid house-owners or developers of houses in knowing what features to construct in a house, so as to fetch a higher sale price. The information could even guide the decision-making regarding where to build a house, and what type of house to build. Thus, other than the ability to predict the price alone, the new visibility on these notable features would also be useful and value-adding to people dealing with real estate.

Further directions in this project could involve expanding the scope of the data analysed to other towns or cities, and also to test out models other than linear regression for the prediction. Also, the temporal aspect of the data could be studied to see if there are any changes or trends year by year. If there are anomalous years or time periods in the data, the model's generalisation ability would have been affected. On the flip side, if the differing patterns in time periods are identified (weather seasons, world events), the model's predictive abilities for market conditions characteristic of a particular time period could be enhanced by focussing on that scope.