# SpaceX-Landing-Success-Project
Determining if first stage of Falcon9 will successfully land

## Project background and context:
In the modern space race, companies are utilizing cost efficiency through reusability. 
SpaceX dominates the launch market with Falcon 9 at $62M per launch—60% cheaper than competitors ($165M+). Their advantage are reusable first-stage boosters.
For company Space Y to compete, predicting SpaceX’s reuse decision is critical. By analyzing public launch data and training machine learning models, Space Y can forecast SpaceX’s costs and strategically price its own services.

Question Space Y wants to answer:
### Will SpaceX be able to successfully land the first stage of its rocket during its next launch?

The repository contains files:
  1. Falcon9CollectingData - data collection using API api.spacexdata.com
  2. Falcon9WebScraping - 
  3. Falcon9DataWrangling - 
  4. Falcon9SQL
  5. Falcon9FeatureEngineering
  6. Falcon9FoliumGeospatial
  7. Falcon9DASHS
  8. Falcon9MachineLearningPrediction
     
Dataset Sources: 
  1. Dataset provided by IBM Skills Network - json file can be found [here.](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json)
  2. [Wikipedia - List of Falcon 9 and Falcon Heavy launches](https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922)
  3. dataset_part_1.csv, the result of Falcon9CollectingData also available [here.](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv)
  4. my_data1.db and Spacex.csv, available in this repository
  5. dataset_part_2.csv, the result of Falcon9DataWrangling, also available [here.](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv)
  6. Dataset provided by IBM Skills Network - spacex_launch_geo.csv, available in this repository
  7. Dataset provided by IBM Skills Network - spacex_launch_dash.csv, available in this repository
  8. Datasets provided by IBM Skills Network - dataset_part_2(for Falcon9Machine Learning).csv and dataset_part_3(for Falcon9Machine Learning).csv, available in this repository

Technologies Used: python, pandas, matplotlib, sklearn, plotly, folium, dash, seaborn

Installation: copy and run the code in Jupyter Notebooks or other Python editor of choice. Keep dataset files in the same folder.

![First_stage_landing](https://github.com/natvnu/SpaceX-Landing-Success-Project/blob/main/landing_1.gif?raw=true)



