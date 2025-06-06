# SpaceX-Landing-Success-Project
Determining if first stage of Falcon9 will successfully land

## Project background and context:
In the modern space race, companies are utilizing cost efficiency through reusability. 
SpaceX dominates the launch market with Falcon 9 at $62M per launch—60% cheaper than competitors ($165M+). Their advantage are reusable first-stage boosters.
For comapny Space Y to compete, predicting SpaceX’s reuse decision is critical. By analyzing public launch data and training machine learning models, Space Y can forecast SpaceX’s costs and strategically price its own services.

Question Space Y wants to answer:
### Will SpaceX be able to successfully land the first stage of its rocket during its next launch?

The repository contains files:
  1. Falcon9CollectingData - data collection using API api.spacexdata.com
  2. Falcon9WebScraping - Comparation of Random Forest and XGBoost modeling performance in predicting house prices in California
  3. Falcon9DataWrangling - evaluation of the random forest regression models using various evaluation metrics and extraction of feature importances
  4. Falcon9SQL
  5. Falcon9FeatureEngineering
  6. Falcon9FoliumGeospatial
  7. Falcon9DASHS
  8. Falcon9MachineLearningPrediction
     
Dataset Sources: 
  1. Data provided by IBM Skills Network - json file can be found [here.](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json)

Technologies Used: python, pandas, matplotlib, scikit-learn, plotly, seaborn, xgboost

Installation: copy and run the code in Jupyter Notebooks or other Python editor of choice. Keep dataset files in the same folder.

![Feature_Importances_in_Random_Forest_Regression](https://raw.githubusercontent.com/natvnu/Machine_Learning/0e4932d49f493e5f633fd70bb80ccc3c65409168/Supervised%20Machine%20Learning%20-%20Regression/3_Feature_Importances_in_Random_Forest_Regression.png)

![Regularization_Linear_regression_coefficients](https://raw.githubusercontent.com/natvnu/Machine_Learning/0e4932d49f493e5f633fd70bb80ccc3c65409168/Supervised%20Machine%20Learning%20-%20Regression/4_Regularization_Linear_regression_coefficients.png)



