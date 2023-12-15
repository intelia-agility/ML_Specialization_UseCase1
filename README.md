
# Taxi Demand Prediction and Management

## Table of Contents
- [3.1.1 Code](#311-code)
  - [3.1.1.1 Code Repository](#3111-code-repository)
  - [3.1.1.2 Code Origin Certification](#3112-code-origin-certification)
- [3.1.2 Data](#312-data)
  - [3.1.2.1 Dataset in Google Cloud](#3121-dataset-in-google-cloud)
- [3.1.3 Whitepaper / Blog](#313-whitepaper--blog)
  - [3.1.3.1 Business Goal and Machine Learning Solution](#3131-business-goal-and-machine-learning-solution)
  - [3.1.3.2 Data Exploration](#3132-data-exploration)
  - [3.1.3.3 Feature Engineering](#3133-feature-engineering)
  - [3.1.3.4 Preprocessing and the Data Pipeline](#3134-preprocessing-and-the-data-pipeline)
  - [3.1.3.5 Machine Learning Model Design(s) and Selection](#3135-machine-learning-model-designs-and-selection)
  - [3.1.3.6 Machine Learning Model Training and Development](#3136-machine-learning-model-training-and-development)
  - [3.1.3.7 Machine Learning Model Evaluation](#3137-machine-learning-model-evaluation)
- [3.1.4 Proof of Deployment](#314-proof-of-deployment)
  - [3.1.4.1 Model/Application on Google Cloud](#3141-modelapplication-on-google-cloud)
  - [3.1.4.2 Callable Library/Application](#3142-callable-libraryapplication)
  - [3.1.4.3 Editable Model/Application](#3143-editable-modelapplication)
- [Conclusion](Conclusion)

- [Resources](Resources)


## 3.1.1 Code

### 3.1.1.1 Code Repository
The code repository for the Taxi Demand Prediction and Management project can be found at [this GitHub link](https://github.com/intelia-agility/ML_Specialization_UseCase1.git). The repository contains all the code used in Demo #1, including scripts for data preprocessing, model training, evaluation, and deployment instructions.

To clone the repository and start exploring the code, use the following command in your terminal:

```bash
git clone https://github.com/intelia-agility/ML_Specialization_UseCase1.git
```

### 3.1.1.1 Code Origin Certification
We, Intelia, confirm that all the code in this case study is original and developed within our organization.

## 3.1.2 Data

### 3.1.2.1 Dataset in Google Cloud

The City of Chicago, known for its commitment to transparency and innovation in urban management, releases the Chicago Taxi Trips dataset, a testament to its rich data-driven culture. This dataset is a treasure trove of insights, meticulously gathered and anonymized to respect privacy while offering a detailed look into the city's bustling taxi ecosystem.

Spanning several years of taxi trip records, the dataset encompasses an array of information that is crucial for both operational analysis and strategic planning. Among the data points included are:

- Precise timestamps of taxi trips, painting a picture of demand throughout the day.
- Geographical coordinates for pick-ups and drop-offs, anonymized to census tract levels, allowing for a granular view of urban travel patterns.
- Trip distances, fares, and payment types, shedding light on economic aspects of the taxi services.
- Taxi identification details, ensuring a layer of operational transparency.

Chicago's progressive approach to open data not only facilitates civic engagement and research but also serves as a foundation for advanced computational studies, such as machine learning and predictive analytics. The dataset's breadth and depth make it an exemplary resource for tackling complex urban challenges like traffic congestion, public transport optimization, and on-demand service allocation.

In our Taxi Demand Prediction and Management project, hosted on Google Cloud Platform (GCP) with the project identifier `mlops`, we harness this dataset to predict taxi demand using machine learning. This project aims at improving taxi fleet efficiency, thereby enhancing the overall transportation network performance.

#### Chicago Taxi Trips Dataset Schema

The schema for the Chicago Taxi Trips dataset is as follows:

| Field Name              | Mode     | Type      | Description |
|-------------------------|----------|-----------|-------------|
| unique_key              | REQUIRED | STRING    | Unique identifier for the trip. |
| taxi_id                 | REQUIRED | STRING    | A unique identifier for the taxi. |
| trip_start_timestamp    | NULLABLE | TIMESTAMP | When the trip started, rounded to the nearest 15 minutes. |
| trip_end_timestamp      | NULLABLE | TIMESTAMP | When the trip ended, rounded to the nearest 15 minutes. |
| trip_seconds            | NULLABLE | INTEGER   | Time of the trip in seconds. |
| trip_miles              | NULLABLE | FLOAT     | Distance of the trip in miles. |
| pickup_census_tract     | NULLABLE | INTEGER   | The Census Tract where the trip began. For privacy, this Census Tract is not shown for some trips. |
| dropoff_census_tract    | NULLABLE | INTEGER   | The Census Tract where the trip ended. For privacy, this Census Tract is not shown for some trips. |
| pickup_community_area   | NULLABLE | INTEGER   | The Community Area where the trip began. |
| dropoff_community_area  | NULLABLE | INTEGER   | The Community Area where the trip ended. |
| fare                    | NULLABLE | FLOAT     | The fare for the trip. |
| tips                    | NULLABLE | FLOAT     | The tip for the trip. Cash tips generally will not be recorded. |
| tolls                   | NULLABLE | FLOAT     | The tolls for the trip. |
| extras                  | NULLABLE | FLOAT     | Extra charges for the trip. |
| trip_total              | NULLABLE | FLOAT     | Total cost of the trip, the total of the fare, tips, tolls, and extras. |
| payment_type            | NULLABLE | STRING    | Type of payment for the trip. |
| company                 | NULLABLE | STRING    | The taxi company. |
| pickup_latitude         | NULLABLE | FLOAT     | The latitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. |
| pickup_longitude        | NULLABLE | FLOAT     | The longitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. |
| pickup_location         | NULLABLE | STRING    | The location of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. |
| dropoff_latitude        | NULLABLE | FLOAT     | The latitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. |
| dropoff_longitude       | NULLABLE | FLOAT     | The longitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. |
| dropoff_location        | NULLABLE | STRING    | The location of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. ||

This schema provides a structured view of the data collected for each taxi trip, offering a solid foundation for comprehensive analysis and predictive modeling.

Our project, hosted on Google Cloud Platform (GCP), leverages the rich dataset of Chicago Taxi Trips. The project identifier for GCP is `mlops`.

The datasets for training, validation, and weather data integration are stored within this GCP project, ensuring scalable storage and efficient data management that is crucial for machine learning workloads.

The datasets are as follows:

- GCP Project: `mlops`
- Training Data: `ChicagoTaxitrips.training_data`
- Validation Data: `ChicagoTaxitrips.validation_data`
- Hourly Weather Data: `ChicagoTaxitrips.weather_hourly`

To complement our trip data, we incorporate weather conditions from the Open-Meteo.com Weather API, which include temperature, humidity, and precipitation. This enrichment aims to reveal the influence of weather on taxi demand, providing a more nuanced view for demand forecasting.

#### Citation for Weather Data
Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. [https://doi.org/10.5281/ZENODO.7970649](https://doi.org/10.5281/ZENODO.7970649)

## 3.1.3 Whitepaper / Blog

### 3.1.3.1 Business Goal and Machine Learning Solution

#### Introduction
In bustling urban environments, the efficiency of taxi services is crucial for both the operators and the city's mobility. This project focuses on leveraging Machine Learning (ML) to predict taxi demand based on time, date, and location factors. By anticipating high-demand areas and times, taxi companies can optimize their fleet management, reduce customer wait times, and improve service availability.

#### Business Question/Goal
The project is anchored on the business goal of increasing the efficiency of taxi distribution within the city to minimize customer wait times. The central question being addressed is: "How can we predict taxi demand more accurately to ensure that taxi fleets are distributed efficiently across the city?" This question is pivotal for enhancing urban mobility and optimizing the taxi service industry's response to fluctuating demand.

#### Machine Learning Use Case
To address this business goal, the project utilizes a machine learning (ML) model that predicts taxi demand based on a combination of historical taxi trip data and weather conditions. The model is designed to recognize patterns and trends from a variety of features, including time of day, date, location, weather conditions, and more.

#### Solution's Impact on Business Goal
The ML solution's efficacy lies in its ability to deliver a comprehensive demand forecast that taxi operators can use to make data-driven decisions for fleet distribution. The end-to-end TensorFlow Extended (TFX) pipeline is the cornerstone of this solution, encompassing several stages:

1. **Data Enrichment**: Augmenting the Chicago Taxi Trips dataset with weather data to capture additional factors that impact taxi demand.
2. **Exploratory Data Analysis (EDA)**: Conducting a thorough examination of the data to identify key variables and patterns that influence taxi demand.
3. **Data Preprocessing**: Utilizing BigQuery to clean and prepare the data, setting the stage for effective machine learning. This preprocessing pipeline is integral to transforming raw data into a format suitable for ML modeling.
4. **End-to-End TFX Pipeline**: Orchestrating the entire ML workflow, from data validation and model training to deployment and serving.

Through this structured ML workflow, the solution directly contributes to the goal of optimizing taxi fleet distribution, thereby reducing customer wait times and improving the overall quality of taxi services in the city.

### 3.1.3.2 Data exploration
**Note:** For a comprehensive walkthrough of our data exploration process, please refer to the [Exploratory Data Analysis Notebook](Notebooks/Exploratory%20Data%20Analysis.ipynb).

In this project, data exploration is a multi-stage process that begins with data enrichment and is followed by a thorough exploratory analysis.

#### Overview
To deepen our analysis, we have enriched the Chicago Taxi Trip dataset with additional weather data. This process is key to exploring the impact of various weather conditions on taxi trip patterns.

#### Integrating Weather Data
**Note:** The complete code for this section is available in the [Data Enrichment Notebook](Notebooks/Data%20Enrichment.ipynb).

Utilizing the [Open-Meteo.com Weather API](https://open-meteo.com/), we have integrated data such as temperature, humidity, and precipitation. This additional information is vital for comprehending how weather conditions influence taxi demand in Chicago. We fetch weather data with both hourly and daily granularity to perform a thorough correlation analysis between weather changes and taxi trip frequency.

#### Code Snippet:
```python

# Fetch the data from the API
response = requests.get(''https://archive-api.open-meteo.com/v1/archive?latitude=41.85&longitude=-87.65&start_date=2013-01-11&end_date=2023-09-10&hourly=temperature_2m,relativehumidity_2m,precipitation,rain,snowfall,weathercode,windspeed_10m&models=best_match&daily=weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean,shortwave_radiation_sum,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max&timezone=America%2FChicago'')
data = response.json()

# Create DataFrames from the 'hourly' and 'daily' data
df_hourly = pd.DataFrame(data['hourly'])
df_daily = pd.DataFrame(data['daily'])

```
After fetching and structuring the weather data, our next steps involve analyzing this data in conjunction with taxi trip records to identify patterns and insights

####  Data Granularity and Analysis
Our analysis emphasizes hourly data granularity. This approach helps us capture the dynamic nature of both weather conditions and taxi trip demand, allowing for a more nuanced understanding of their interplay. Hourly data provides the detail necessary for accurate demand forecasting, crucial for operational planning and resource allocation.
#### Code Snippet:
```python
from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Convert time columns to datetime and reset the index
df_hourly.index = pd.to_datetime(df_hourly.index)
df_hourly = df_hourly.reset_index()

# Upload the hourly DataFrame to a new test table
table_id_new = 'your_project.your_dataset.weather_hourly_test'
try:
    client.load_table_from_dataframe(df_hourly, table_id_new).result()
    print("Data uploaded successfully to the test table.")
except Exception as e:
    print(f"Error during upload: {e}")

# If the test table upload is successful and you want to overwrite the main table:
table_id = 'your_project.your_dataset.weather_hourly'
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",  # overwrite existing table
)
try:
    client.load_table_from_dataframe(df_hourly, table_id, job_config=job_config).result()
    print("Data uploaded successfully to the main table.")
except Exception as e:
    print(f"Error during upload to the main table: {e}")
```

#### Citation
Zippnfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649

#### Exploratory Data Analysis (EDA)
**Note:** For a comprehensive walkthrough of our data exploration process, please refer to the [Exploratory Data Analysis Notebook](Notebooks/Exploratory%20Data%20Analysis.ipynb).
####  Overview
The Exploratory Data Analysis section provides insights into the dataset through various angles and techniques. Each subsection below corresponds to a key aspect of the EDA, complemented by visualizations that highlight our findings.

#### Descriptive Statistical Analysis
The initial step of our EDA involved an in-depth statistical examination of the dataset using the `describe()` function. This provided valuable insights into the central tendencies, dispersion, and shapes of various features, crucial for our understanding of the taxi demand landscape.

- **Key Statistical Metrics**:
  - **Count**: Represented the number of non-missing values in each column, crucial for assessing data completeness.
  - **Mean and Standard Deviation**: These measures offered insights into the average trends and variability in the data, such as the average trip duration (`trip_seconds`) and distance (`trip_miles`).
  - **Minimum and Maximum Values**: Highlighted the range within each feature. Notably, some features like `trip_seconds` and `trip_miles` showed extremely high maximum values, suggesting the presence of outliers.
  - **Percentiles (25%, 50%, 75%)**: Gave an understanding of the distribution of data, particularly identifying the spread and central tendency of each feature.
  - **Weather-related Features**: Analysis of `temperature_2m`, `relativehumidity_2m`, `precipitation`, etc., revealed patterns in weather conditions during taxi trips.

- **Insights Derived**:
  - The dataset displayed significant variability in trip-related features, with certain values indicating potential data entry errors or unique cases.
  - Weather data analysis was instrumental in understanding the potential influence of environmental factors on taxi demand.


This comprehensive statistical overview was pivotal in guiding our subsequent analyses, enabling us to identify areas requiring deeper investigation and to hypothesize about various factors influencing taxi demand.

#### Feature Selection and Data Inspection

In our pursuit of a robust predictive model for taxi demand, selecting the most relevant and impactful features was a key step. This careful selection aimed to ensure that our model would be informed by data that directly influences or reflects taxi usage patterns.

- **Selected Features**:
  - **Temporal Features**: `trip_start_timestamp`, `trip_end_timestamp`, and `trip_seconds`. These features are critical in understanding the temporal aspects of taxi demand, including the duration and specific timing of trips.
  - **Spatial Features**: `pickup_community_area`, `pickup_latitude`, `pickup_longitude`. Geospatial data provides insights into popular areas for taxi pickups and drop-offs, crucial for demand prediction.
  - **Trip Details**: `trip_miles`, `trip_total`. These features offer a direct look at the trip distances and financial aspects of taxi usage.
  - **Weather Data**: `temperature_2m`, `relativehumidity_2m`, `precipitation`, `rain`, `snowfall`, `weathercode`. Weather conditions can significantly influence travel behavior, making these features vital for understanding demand variations.
  - **Operational Data**: `unique_key`, `company`. These help in identifying individual trips and understanding operational dynamics.


- **Data Inspection**:
  - The `data.info()` function revealed the structure and data types of our selected features, indicating a combination of continuous, categorical, and temporal data.
  - The dataset's range index and data columns were displayed, giving a clear picture of the dataset post-feature selection.

#### Handling Missing Values and Data Integrity
A critical part of our EDA was to assess and manage missing data, ensuring the integrity and reliability of our analysis.

- **Analysis of Missing Data**:
  - We conducted a thorough investigation to determine whether the missing data was missing at random. This assessment was crucial in deciding our approach to handling these gaps.
  - Notably, a significant portion of missing data in `pickup_community_area`, `pickup_latitude`, and `pickup_longitude` was due to privacy masking.

- **Decision on Data Imputation vs. Removal**:
  - Various strategies, including data imputation, were tested to address the missing values. We explored different imputation techniques and evaluated their impact on our modeling.
  - However, it was observed that removing these missing entries was more beneficial for the model's performance. This approach was chosen after careful consideration and comparative testing.

- **Proportion of Missing Data**:
  - Importantly, the missing data constituted a relatively small portion of the overall dataset. This reassured us that removing these entries would not significantly compromise the dataset's comprehensiveness or our analysis's depth.

- **Data Cleaning and Retention**:
  - Following our analysis, we proceeded with the removal of entries with missing critical information. This step was vital to ensure the quality and accuracy of our predictive models.
  - The percentage of data retained post-cleaning was approximately 92.16%, indicating that our dataset still maintained a substantial volume of valuable information for robust analysis.

- **Visualization Placeholder for Missing Data Analysis**:
  ![Missing Data Analysis Visualization Placeholder](path/to/missing_data_analysis_visual.png)

This meticulous approach to handling missing data ensured that the remaining dataset was both comprehensive and of high quality, forming a solid basis for our subsequent in-depth analyses and predictive modeling.

#### Univariate Analysis Overview

In our comprehensive Exploratory Data Analysis (EDA), univariate analysis played a crucial role in understanding individual features of the dataset. This analysis was segmented into different sections, each focusing on specific types of features and their implications for taxi demand in Chicago.

#### Continuous and Categorical Features Analysis

- **Continuous Features**: Analyzed key continuous features like `trip_seconds`, `trip_miles`, `trip_total`, and various weather-related attributes (temperature_2m, relativehumidity_2m, precipitation, rain, snowfall). Histograms with kernel density estimates provided insights into the distribution of these variables, highlighting trends and anomalies.

- **Categorical Features**: Features such as `company` and `weathercode` were analyzed using count plots to understand their frequency distribution. This helped us grasp the diversity and prominence of different taxi companies and weather conditions during taxi trips.
![Univariate Analysis of Continuous and Categorical Features](assets/Univariate%20Analysis%20of%20Continuous%20and%20Categorical%20Features.png)
#### Observations from Univariate Analysis

- **Continuous Features**: Most trip durations, distances, and fares were on the lower side, with right-skewed distributions, indicating that shorter and less expensive trips were more common.
- **Weather-related Features**: The analysis revealed patterns in climate conditions and their potential impact on taxi demand.
- **Categorical Features**: Distribution of taxi companies and weather conditions provided insights into operational dynamics and the impact of weather on taxi usage.

#### Univariate Analysis of Spatial Features

- **Spatial Features Analysis**: Examined spatial features like `pickup_community_area`, `pickup_latitude`, and `pickup_longitude`. Count plots and histograms were employed to explore these features, identifying popular areas for taxi pickups and potential hotspots.
![Univariate Analysis of Spatial Features](assets/Univariate%20Analysis%20of%20Spatial%20Features.png)

- **Observations**: Some community areas had significantly higher pickup frequencies, suggesting they were key hotspots. Latitude and longitude data highlighted the geographical concentration of taxi pickups in certain areas.

Each step of our univariate analysis provided crucial insights into different aspects of taxi demand, laying a foundation for more detailed multivariate analysis and predictive modeling. The use of visualizations at each step enhanced our understanding of the data and helped in identifying key areas for focused analysis and strategy development.

#### Timestamp Conversion and Time Feature Extraction

- **Extracting Time-Related Features**: Converted the `trip_start_timestamp` to datetime format and extracted various time-related features (year, month, day, hour, weekday, trip date, day of the week). This allowed for an in-depth analysis of taxi demand patterns over different time periods.


#### Bivariate Analysis

In this section, we delve into Bivariate Analysis to explore the relationships between two distinct variables and their combined impact on taxi demand. This approach helps us understand the interactions and dependencies between various factors in our dataset.

#### Temporal Analysis

##### ***Hourly Taxi Demand***
- *Description*: Analysis of taxi demand based on hourly data.
![Analysis of Hourly Taxi Demand](assets/Analysis%20of%20Hourly%20Taxi%20Demand.png)
- *Observations*:
  - Decrease in demand in the early morning hours, with the lowest point around 5 AM.
  - Increase in demand from 6 AM, peaking during late afternoon and early evening, then decreasing throughout the night.
  - Typical urban dynamics with increased demand during peak hours and reduced demand during off-peak periods.

##### ***Taxi Demand by Day of the Week***
- *Description*: Analyzing the distribution of taxi demand across different days of the week.
![Analysis of Taxi Demand by Day of the Week](assets/Analysis%20of%20Taxi%20Demand%20by%20Day%20of%20the%20Week.png)
- *Observations*:
  - Consistent demand from Monday to Friday, with an increase on Fridays.
  - Drop in demand on Saturdays and the lowest on Sundays.
  - Weekday demand driven by work-related commuting, contrasting with quieter weekends.

##### ***Taxi Demand by Month***
- *Description*: Exploration of how taxi demand varies across different months.
![Analysis of Taxi Demand by Month](assets/Analysis%20of%20Taxi%20Demand%20by%20Month.png)
- *Observations*:
  - Steady demand from January through May.
  - Peak in June, followed by a decline in July and stabilization from August to December.
  - June's peak possibly related to seasonal events, holidays, or tourist influx.

##### Categorical Analysis

##### ***Taxi Demand vs. Weather Code***
- *Description*: Investigation of demand variations across different weather conditions as indicated by weather codes.
![Taxi Demand vs. Weather Code Analysis](assets/Taxi%20Demand%20vs.%20Weather%20Code%20Analysis.png)

- *Observations*:
  - Higher demand for lower weather code values (favorable conditions).
  - Decrease in demand with increasing weather code values, suggesting reduced demand in severe or unfavorable weather.

##### ***Taxi Demand by Weather Condition***
- *Description*: Detailed analysis of taxi demand under specific weather conditions.
![Taxi Demand by Weather Condition Analysis](assets/Taxi%20Demand%20by%20Weather%20Condition%20Analysis.png)

- *observations*:
  - Highest demand in sunny and clear weather.
  - Significant demand during less favorable conditions like rain or fog.
  - Lowest demand during extreme conditions like thunderstorms or heavy snow.

##### Spatial Analysis

##### ***Spatial Analysis of Taxi Pickups by Community Area***
- *Description*: Distribution of taxi pickups across different community areas.
![Spatial Analysis of Taxi Pickups by Community Area](assets/Spatial%20Analysis%20of%20Taxi%20Pickups%20by%20Community%20Area.png)

- *Observations*:
  - Variations in pickup frequency across areas, influenced by proximity to key destinations or transport hubs.
  - Disparity in average fares, indicating differences in trip lengths or destination popularity.

##### ***Analysis of Average Fare by Pickup Community Area***
- *Description*: Investigation into average fares in different community areas.
![Analysis of Average Fare by Pickup Community Area](assets/Analysis%20of%20Average%20Fare%20by%20Pickup%20Community%20Area.png)

- *Observations*:
  - Range in average fares, with certain areas having higher fares on average due to longer trips or routes to in-demand locations.
  - These fare dynamics are crucial for efficient service allocation and pricing strategies.


##### Correlation Analysis of Numerical Features

In this section, we explore the intricate associations and dependencies among the numerical features within our dataset through correlation analysis. This approach is pivotal in unraveling the subtle and complex factors that influence taxi demand and the characteristics of taxi trips.

- *Correlation Heatmap*: The heatmap visualizes the correlation coefficients between different numerical features, such as trip miles, trip seconds, trip total, and various weather-related variables like temperature, humidity, and precipitation.
![Correlation Analysis of Numerical Features](assets/Correlation%20Analysis%20of%20Numerical%20Features.png)

- *Findings*:
  - A positive correlation is observed between trip miles and trip seconds, indicating that trips covering more miles typically have longer durations.
  - Trip total shows a positive correlation with both trip miles and trip seconds, suggesting that longer trips, both in distance and duration, generally lead to higher fares.
  - Weather-related variables display very weak correlations with trip details, suggesting minimal direct linear relationships between these weather factors and the specific attributes of taxi trips.
- *Implications*: This analysis is crucial for understanding how different aspects of taxi trips are interconnected. It provides insights crucial for refining fare structuring, service management, and operational strategies in the taxi service industry.

#### Outlier Analysis

The Outlier Analysis section is dedicated to identifying and understanding anomalies within our dataset. This involves examining various numerical features, continuous variables, and spatial data to pinpoint irregularities that could influence our analysis and model accuracy.

##### Outlier Analysis of Numerical Features

- *Box Plots*: Visualization of outliers in various numerical features using box plots.
![Outlier Analysis of Numerical Features](assets/Outlier%20Analysis%20of%20Numerical%20Features.png)

- *Findings*:
  - Trip_seconds and trip_total exhibit significant outliers, indicating the presence of unusually long or expensive trips.
  - Trip_miles also shows outliers, but they are not as pronounced.
  - Temperature_2m shows a relatively normal distribution with minimal outliers.
  - Relativehumidity_2m mostly falls within the 60-90% range, indicating a good spread with no significant anomalies.


##### Percentile Analysis of Continuous Variables

- *Analysis*: Percentile distribution provides insights into the range and spread of continuous variables like trip_seconds, trip_miles, and trip_total.
- *Observations*:
  - Trip_seconds: The majority of trips are shorter than 38 minutes, with a maximum duration significantly higher, suggesting outliers.
  - Trip_miles: Most trips are under 15.72 miles, with the longest trip recorded at 3430.53 miles.
  - Trip_total: Median fare is around $15.50, with the highest recorded fare being significantly higher.
- *Implications*: This percentile analysis helps identify typical trip characteristics and detect anomalies within the dataset.

##### Scatter Plot Analysis of Taxi Pickup Locations

- *Visualization*: A scatter plot representing the geographical distribution of taxi pickups.
![Scatter Plot Analysis of Taxi Pickup Locations](assets/Scatter%20Plot%20Analysis%20of%20Taxi%20Pickup%20Locations.png)

- *Observations*:
  - Dense clusters indicate popular areas or hotspots for taxi pickups.
  - Areas with fewer points suggest less frequent taxi activity, possibly in residential zones or less commercially active areas.


##### Data Cleaning and Outlier Treatment Process

- **Process Description**: 
  - The cleaning process involves a series of meticulous steps designed to refine the taxi dataset for more accurate analysis.
  - Geographical Filtering: Ensures that all taxi pickups and drop-offs are within the defined Chicago city boundaries, filtering out data points that fall outside these parameters.
  - Removing Zero Values: Trips with zero values for critical variables such as trip_seconds, trip_miles, or trip_total are excluded, as they likely represent data recording errors or irrelevant entries, ensuring the integrity of the dataset.
  - 12-Hour Rule: Trips exceeding a 12-hour duration are removed from the dataset. This step adheres to realistic and legal driving limits, eliminating data points that might be the result of data entry errors or other anomalies.
  - Capping Extreme Values: Extreme values for variables like trip duration, miles, fare, and weather-related measures are capped at their 1st and 99th percentiles. This treatment mitigates the influence of extreme outliers that could skew the analysis, ensuring a more balanced and representative dataset.

- *Data Retention*: 
  - Post-cleaning, a considerable percentage of the data is retained, striking a balance between maintaining a robust dataset size and ensuring the quality and reliability of the data.
  - The retention rate is a testament to the effectiveness of the cleaning process, indicating that while it rigorously filters out inaccuracies and anomalies, it preserves the bulk of valuable data.

- *Strategic Importance*: 
  - This cleaning and outlier treatment process is a critical foundation for any subsequent data analysis and modeling. By ensuring the dataset's accuracy and relevance, it lays the groundwork for drawing reliable conclusions and insights.
  - These practices are not just about removing outliers or erroneous data; they are about enhancing the overall quality of the dataset, thereby enabling more precise and meaningful analyses.
  - The process also reflects the importance of data integrity in the field of data science, where the quality of the input data significantly influences the validity of the results.
  
#### Feature Transformations

In our analysis, we delve into the transformation of key numerical features within the taxi dataset. This step is crucial in understanding the underlying data distribution and addressing any skewness or anomalies that may impact our analysis.

- **Analyzing Skewness in Features**: 
  - Our initial observations reveal that many of our features, such as trip_seconds, trip_miles, and trip_total, exhibit skewed distributions. Skewness can significantly affect the performance of various statistical models and machine learning algorithms.
  - To address these issues, we apply specific transformations aimed at normalizing the data distributions.

- **Applied Transformations**: 
  - We focus on two primary transformation techniques: logarithmic and square root transformations. These are applied to features including trip_total, trip_miles, trip_seconds, temperature_2m, relativehumidity_2m, and precipitation.
  - Log Transformation: This is particularly useful for data with long tails or high variability. By transforming data using the logarithmic scale, we aim to reduce right-skewness.
  - Square Root Transformation: This is a milder transformation compared to the logarithmic one and is used to reduce the effect of extreme values.
 
  ![Visualization of Feature Transformations](assets/Visualization%20of%20Feature%20Transformations.png)

- **Visualization and Comparative Analysis**:
  - The impact of these transformations is visualized through histograms and density plots. Each feature's original distribution is compared with its log-transformed and square root-transformed distributions.
  - These visualizations allow us to assess the effectiveness of each transformation in normalizing the data and to choose the most appropriate method for our analysis.



##### Observations from Feature Transformation Analysis

- The transformation techniques significantly alter the shape of the data distributions, bringing them closer to normality in many cases.
- Log Transformation is particularly effective in reducing the skewness of features with large ranges or outliers.
- Square Root Transformation, while milder, also contributes to reducing skewness and is especially useful for data that cannot undergo log transformation.
- These transformations are instrumental in preparing our data for more sophisticated analyses, ensuring that the assumptions of various statistical and machine learning techniques are met.

#### Analysis of Skewness and Kurtosis for Feature Transformations

This analysis focuses on the skewness and kurtosis of various features in our dataset after applying transformations. It's a crucial step in determining how these transformations impact the distribution characteristics of our data.

##### ***Procedure***
- We calculate the skewness and kurtosis for each feature in its original, log-transformed, and square root-transformed states. These metrics provide insights into the symmetry and tail behavior of the distributions.
  - *Skewness*: Measures distribution asymmetry. Positive skew indicates a right tail, while negative skew indicates a left tail.
  - *Kurtosis*: Indicates whether data are heavy-tailed (positive kurtosis) or light-tailed (negative kurtosis) compared to a normal distribution.

##### Transformations Applied
- Applied transformations include original data, log transformation, and square root transformation on features like `trip_total`, `trip_miles`, `trip_seconds`, `temperature_2m`, `relativehumidity_2m`, and `precipitation`.

##### Observations
- Transformations, particularly logarithmic and square root, tend to normalize distributions, reducing skewness and adjusting kurtosis values closer to a normal distribution.
  - Example: Log transformation of `trip_total` significantly reduces its skewness and kurtosis, resulting in a more symmetric and less heavy-tailed distribution.
  - Note: Some transformations may not be suitable for every feature. For instance, log transformation of `temperature_2m` results in NaN values, indicating its unsuitability for this specific feature.

##### Implications
- Understanding these changes in distribution is crucial for data preprocessing, especially for statistical and machine learning models that often assume normally distributed inputs.
- This analysis aids in selecting the most appropriate transformations for each feature, ensuring our data meets the necessary assumptions for advanced analytical techniques.

##### Visualization Placeholder
- **Insert Visualization Here**: Include tables or charts that display skewness and kurtosis values for each feature under different transformations.


#### Conclusion
Our EDA is a comprehensive process that lays the foundation for predictive modeling. It ensures our understanding of the data is robust and our subsequent models are informed by deep insights.

#### 3.1.3.3 Feature Engineering

##### Feature Engineering: Time-based and Cyclic Features

we focus on enriching our dataset with time-based and cyclic features. This approach aims to capture the temporal patterns and cyclic nature of taxi demand more effectively.

##### ***Extraction of Time-based Features***
- *Overview*: Key time-based features are extracted from the `trip_start_timestamp` field to understand the temporal dynamics of taxi usage.
- *Extracted Features*: 
  - *Year, Month, Day*: To identify long-term trends and seasonal variations.
  - *Hour*: Crucial for understanding daily demand cycles.
  - *Weekday*: Differentiates weekdays from weekends, reflecting varying demand patterns.
  - *Trip Date*: Useful for pinpointing specific events or anomalies.

##### ***Creation of Cyclic Features***
- *Rationale*: Time-based features like hour, day, and month inherently follow a cyclical pattern, which traditional numerical or categorical representations do not capture effectively.
- *Transformation Technique*: 
  - Sine and cosine transformations are applied to hour, weekday, and month features.
  - *Hourly Cycles*: `hour_sin` and `hour_cos` capture the 24-hour daily cycle.
  - *Weekly Cycles*: `day_sin` and `day_cos` encapsulate the weekly cycle.
  - *Monthly Cycles*: `month_sin` and `month_cos` represent the annual monthly cycle.

##### Significance in Modeling
- These cyclic features are essential for models where time is a significant factor, such as in predicting taxi demand patterns.
- By accurately representing the cyclic nature of time, we enhance the model's ability to interpret temporal data more realistically.

### 3.1.3.4 Preprocessing and the Data Pipeline
**Note:** For detailed code and methodologies used in this section, please refer to our [Data Preprocessing Notebook](Notebooks/Data%20Preprocessing.ipynb).

##### Data Preprocessing Pipeline Using BigQuery
##### Introduction
Building on our Exploratory Data Analysis, we've developed a comprehensive Data Preprocessing Pipeline. This pipeline transforms raw taxi trip data into a structured and insightful format, focusing on data cleaning, feature extraction, and data segmentation, ensuring our data is clean, reliable, and enriched with meaningful attributes for accurate demand forecasting.

##### Strategic Importance
Our pipeline streamlines and optimizes the taxi demand prediction process, unlocking deeper insights and enabling precise demand predictions. This directly contributes to enhanced fleet management and customer satisfaction.

##### Technical Approach
Our approach encompasses data integration, granularity control, feature engineering, and encapsulation in a callable API. Key steps include:

1. **Data Integration**: Joining taxi trip data with hourly weather data using SQL queries in BigQuery, crucial for understanding the impact of weather on taxi demand.
2. **Granularity Control**: Segmenting data based on time and spatial dimensions.
3. **Feature Engineering**: Using advanced SQL for feature extraction, including time-based features and trigonometric transformations for capturing cyclical patterns in demand.
4. **Callable API**: Encapsulating the entire pipeline within a callable API, ensuring seamless integration and dynamic data feeding into the production model.

##### Data Preprocessing Steps
The detailed steps of our data preprocessing pipeline are as follows:

- **Joining and Cleaning**: Leveraging SQL in BigQuery to integrate taxi trip and weather data, followed by rigorous cleaning to ensure data quality.
- **Outlier Capping**: Utilizing statistical methods to cap outliers and standardize trip duration, distance, and cost data.
- **Feature Extraction**: Extracting critical features and sorting data to lay the groundwork for effective aggregation and analysis.
- **Data Aggregation**: Aggregating data at the community area level to discern local demand trends.
- **Model Training Segmentation**: Segmenting data into distinct sets for training, validation, and testing to prevent data leakage and model overfitting.
- **Exporting for Modeling**: Exporting preprocessed data as CSV files, ready for use in predictive modeling.

**Callable API**:
```sql
CALL `mlops-363723.ChicagoTaxitrips.data_preprocessing_pipeline_chicago_taxi_trips`();
```
### 3.1.3.5 Machine Learning Model Design(s) and Selection
#### Model Overview
The demand for taxi trips within an urban landscape like Chicago presents a multifaceted problem, where the prediction accuracy hinges on the model's capability to understand and interpret a web of spatial-temporal factors. The chosen model for this task is a Deep Neural Network (DNN), also known as a Multilayer Perceptron (MLP). This model stands out due to its powerful ability to recognize and make sense of the complex patterns that emerge from the interplay of various factors such as location, time, weather conditions, and more.

At its core, the DNN is a composition of layers with numerous neurons that work in unison to process input data, learn from it, and make predictive outputs. The strength of a DNN lies in its depth and breadth, which are manifested in the multiple hidden layers and the vast number of neurons that allow the network to perform intricate computations. These capabilities make DNNs particularly suited for tackling the dynamic and non-linear nature of taxi trip demand forecasting.

In the following sections, we will delve into the specifics of why a DNN was the ideal choice for this scenario, focusing on its handling of temporal and spatial features, flexibility, scalability, and more. Through careful feature engineering and model optimization, the DNN is expected to provide valuable insights and accurate predictions that can significantly aid in demand forecasting for taxi trips.
#### Model Selection Criteria for Taxi Demand Prediction

For our taxi demand prediction task in Demo #1, we have selected a deep learning model implemented using TensorFlow and Keras. This model stands out due to its ability to handle the intricacies of spatiotemporal data which is characteristic of taxi trip patterns.

#### Chosen Machine Learning Model/Algorithm

The deep neural network was chosen for its proficiency in capturing non-linear and complex relationships within the data. Deep learning models, particularly those with multiple layers, are known for their feature learning capabilities, which make them highly suitable for tasks where the input data involves intricate interactions between multiple variables, such as time of day, weather conditions, and geospatial information.

#### Criteria for Machine Learning Model Selection

The model was selected based on the following criteria:

- **Capability to Process High-Dimensional Data**: Taxi trip demand prediction involves various input features; our model can integrate and learn from all these features effectively.
- **Ability to Model Non-linear Relationships**: Given the complex nature of factors affecting taxi demand, the chosen model is capable of uncovering and leveraging non-linear relationships within the data.
- **Handling Temporal Dynamics**: The model's structure is adept at incorporating temporal sequences, which is crucial for predicting demand based on historical patterns.
- **Adaptability to Spatial Features**: Taxi demand is heavily influenced by location, and our model can process geospatial inputs to understand and predict demand fluctuations across different urban areas.
- **Robustness to Noisy Data**: The model is designed to be robust against noise and outliers in the data, ensuring reliable predictions even with imperfect inputs.
- **Handling of Temporal and Spatial Features**: MLPs excel at detecting intricate patterns within datasets, vital for managing the complexities in spatial-temporal patterns of taxi trip demands.
- **Flexibility and Customization**: The adaptable architecture of MLPs allows for tailoring layer and neuron counts, matching the specific challenges of taxi demand prediction.
- **Ability to Model Non-linear Relationships**: Essential for capturing the diverse elements like time of day, weather, and location, affecting taxi demand.
- **Scalability**: With high scalability, MLPs are suitable for handling the substantial data volumes typical of urban taxi trip datasets.
- **Generalization Capability**: Known for their effective generalization to new, unseen data when properly regularized and validated.
- **Integration with Feature Engineering**: Effective incorporation of engineered features like trigonometric representations of time, enhancing MLPs' ability to process multidimensional data.

Through a combination of these criteria, the selected model is poised to provide accurate and insightful predictions for taxi demand, leveraging its depth and breadth to address the dynamic and complex nature of urban transportation needs.
#### Model Construction Code Snippet

Here is the code snippet for constructing the deep learning model:

```python
# Function to build the Keras model
def _build_keras_model(hp, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
     feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)
    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key) for key in feature_spec.keys()}

    concatenated_inputs = tf.keras.layers.Concatenate()(list(inputs.values()))

    num_layers = hp.Int('num_layers', 1, 5)
    activation_choice = hp.Choice('activation', ['relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid'])

    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
        concatenated_inputs = tf.keras.layers.Dense(units=units, activation=activation_choice,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_{i}', 1e-5, 1e-2, sampling='log'))
        )(concatenated_inputs)
        if hp.Boolean(f'dropout_{i}'):
            dropout_rate = hp.Float(f'dropout_rate_{i}', 0.1, 0.5)
            concatenated_inputs = tf.keras.layers.Dropout(dropout_rate)(concatenated_inputs)

    output = tf.keras.layers.Dense(1, activation='linear')(concatenated_inputs)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model
```
#### Model Architecture

The model's architecture is designed to efficiently process the various features related to taxi demand. Below is the architecture of the model, as defined in TensorFlow:
#### Architecture Overview

- **Input Layers**: The model starts with multiple input layers, each corresponding to a specific feature such as day cosine (`day_cos_xf`), day sine (`day_sin_xf`), hour cosine (`hour_cos_xf`), etc. These layers are dedicated to handling the diverse set of features that influence taxi demand, including temporal aspects (like time of day), weather conditions, and other relevant factors.

- **Concatenate Layer**: Following the input layers, a concatenation layer (`concatenate_1`) merges the outputs of all input layers into a single unified layer. This approach allows the model to consider all input features simultaneously, enabling it to learn complex interdependencies among them.

- **Dense and Dropout Layers**: After concatenation, the data flows through a series of densely connected (Dense) layers (`dense_4`, `dense_5`, `dense_6`, etc.), which are the core components where most of the model's learning occurs. These layers contain a large number of neurons that enable the model to learn non-linear relationships within the data. Interspersed with these dense layers are Dropout layers (`dropout_3`, `dropout_4`), which help prevent overfitting by randomly setting a fraction of the input units to 0 at each update during training. This regularization technique is crucial for generalizing the model well to new, unseen data.

- **Output Layer**: The architecture culminates in a single neuron with a linear activation function (`dense_9`). This output layer is responsible for producing the final prediction, representing the estimated taxi demand.

#### Significance of the Architecture

- **Handling High-Dimensional Data**: The architecture's ability to process and learn from a high number of input features makes it particularly effective for the taxi demand prediction task, which inherently involves complex and multidimensional data.
- **Modeling Non-linear Relationships**: The use of dense layers enables the model to capture the non-linear relationships often present in real-world data, particularly important in scenarios like taxi demand forecasting where multiple factors interact in complex ways.
- **Balance Between Depth and Efficiency**: While the model is deep enough to learn detailed patterns and relationships in the data, it is also designed to be computationally efficient, ensuring manageable training times and resource usage.

```plaintext
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape          Param #     Connected to                     
==================================================================================================
 day_cos_xf (InputLayer)        [(None, 1)]           0                                            
 day_sin_xf (InputLayer)        [(None, 1)]           0                                            
 hour_cos_xf (InputLayer)       [(None, 1)]           0                                            
 hour_sin_xf (InputLayer)       [(None, 1)]           0                                            
 log_duration (InputLayer)      [(None, 1)]           0                                            
 log_trip_miles (InputLayer)    [(None, 1)]           0                                            
 log_trip_total (InputLayer)    [(None, 1)]           0                                            
 month_cos_xf (InputLayer)      [(None, 1)]           0                                            
 month_sin_xf (InputLayer)      [(None, 1)]           0                                            
 pickup_community_area_xf (InputLayer) [(None, 1)]    0                                            
 public_holiday_xf (InputLayer) [(None, 1)]           0                                            
 rain_xf (InputLayer)           [(None, 1)]           0                                            
 relativehumidity_2m_xf (InputLayer) [(None, 1)]      0                                            
 snowfall_xf (InputLayer)       [(None, 1)]           0                                            
 sqrt_precipitation (InputLayer) [(None, 1)]          0                                            
 temperature_2m_xf (InputLayer) [(None, 1)]           0                                            
 weathercode_xf (InputLayer)    [(None, 1)]           0                                            
 year_xf (InputLayer)           [(None, 1)]           0                                            
 concatenate_1 (Concatenate)    (None, 18)            0           [All input layers]              
 dense_4 (Dense)                (None, 480)           9120        [concatenate_1[0][0]]           
 dense_5 (Dense)                (None, 64)            30784       [dense_4[0][0]]                 
 dense_6 (Dense)                (None, 32)            2080        [dense_5[0][0]]                 
 dense_7 (Dense)                (None, 512)           16896       [dense_6[0][0]]                 
 dropout_3 (Dropout)            (None, 512)           0           [dense_7[0][0]]                 
 dense_8 (Dense)                (None, 288)           147744      [dropout_3[0][0]]               
 dropout_4 (Dropout)            (None, 288)           0           [dense_8[0][0]]                 
 dense_9 (Dense)                (None, 1)             289         [dropout_4[0][0]]               
==================================================================================================
Total params: 206,913
Trainable params: 206,913
Non-trainable params: 0
__________________________________________________________________________________________________
```
#### Model Performance 

#### Performance Analysis

Our model, a sophisticated Multi-Layer Perceptron, has shown commendable efficacy in forecasting taxi demand in Chicago, as indicated by its performance metrics:

- **Mean Absolute Error (MAE)**: 2.1684
- **Root Mean Squared Error (RMSE)**: 3.19

To contextualize these numbers, imagine the task of predicting the number of taxi trips in Chicago each hour. On average, our model's predictions would deviate from the actual number by about 2 to 3 trips. For example, if the actual number of trips in an hour was 20, the model's prediction would likely fall between 17 and 23 trips.

This degree of precision underscores the model's adeptness at striking a crucial balance between bias and variance, an essential factor in predictive modeling. The model is finely tuned to avoid overfitting, where it might otherwise capture random noise in the data, as well as underfitting, where it could fail to discern underlying patterns. As a result, it offers predictions that are not only reliable but also highly adaptable to real-world situations.

### 3.1.3.6 Machine Learning Model Training and Development

**Note:** For an in-depth exploration of our model training and development process, please see our [TFX TaxiDemandInteractivePipeline Notebook](Notebooks/TFX%20TaxiDemandInteractivePipeline.ipynb).


#### Data Preprocessing and Splitting

Our data preprocessing pipeline, integral to the Chicago Taxi Trips dataset, encompasses several key steps as outlined in section [section 3.1.3.4 Preprocessing and the Data Pipeline](#3134-preprocessing-and-the-data-pipeline). This pipeline is critical for preparing the data for effective machine learning model training and evaluation.

#### Key Steps in Data Preprocessing Pipeline:

1. **Data Integration**: Combining taxi trip data with hourly weather data to incorporate environmental factors.
2. **Data Cleaning and Capping**: Applying capping techniques on trip duration, distance, and cost data to mitigate the impact of outliers.
3. **Feature Extraction**: Engineering time-based variables and trigonometric transformations to capture cyclical demand patterns.
4. **Data Aggregation**: Aggregating data at the community area level for more in-depth demand trend analysis.
5. **Data Export**: Exporting processed data sets to CSV files for accessibility during modeling.

#### Dataset Sampling and Justification:

- **Training Data**: Includes data from the years 2020 and 2021.
- **Validation Data**: Composed of data from the year 2022.
- **Test Data**: Contains data from the year 2023 up to April.

This segmentation strategy minimizes data leakage and accurately reflects a realistic scenario for taxi demand prediction. Additionally, we have experimented with various kinds of data splits to determine the most effective approach for our model. These exploratory analyses are documented in our Exploratory Data Analysis notebook.

```sql
-- Data Preprocessing SQL Code
C CREATE OR REPLACE TABLE `mlops-363723.ChicagoTaxitrips.training_data` AS
    SELECT *
    FROM `mlops-363723.ChicagoTaxitrips.aggregated_data`
    WHERE year = 2020 OR year = 2021;

    -- Create Validation Set
    CREATE OR REPLACE TABLE `mlops-363723.ChicagoTaxitrips.validation_data` AS
    SELECT *
    FROM `mlops-363723.ChicagoTaxitrips.aggregated_data`
    WHERE year = 2022;

    -- Create Test Set
    CREATE OR REPLACE TABLE `mlops-363723.ChicagoTaxitrips.test_data` AS
    SELECT *
    FROM `mlops-363723.ChicagoTaxitrips.aggregated_data`
    WHERE year = 2023 AND month <= 4;
  
     EXPORT DATA OPTIONS(
      uri='gs://chicago_taxitrips/DATA_DIRECTORY/training_data/*.csv',
      format='CSV',
      overwrite=true
    ) AS
    SELECT * FROM `mlops-363723.ChicagoTaxitrips.training_data`;

    EXPORT DATA OPTIONS(
      uri='gs://chicago_taxitrips/DATA_DIRECTORY/validation_data/*.csv',
      format='CSV',
      overwrite=true
    ) AS
    SELECT * FROM `mlops-363723.ChicagoTaxitrips.validation_data`;

    EXPORT DATA OPTIONS(
      uri='gs://chicago_taxitrips/DATA_DIRECTORY/test_data/*.csv',
      format='CSV',
      overwrite=true
    ) AS
    SELECT * FROM `mlops-363723.ChicagoTaxitrips.test_data`;
...
```
#### Hyperparameter Tuning and Model Optimization

Our approach to hyperparameter tuning and model optimization employed the Keras Tuner with a RandomSearch strategy, a crucial step in enhancing the model's predictive capabilities. The `tuner_fn` function defines the hypermodel using the `_build_keras_model` function, sets the objective to minimize the validation mean absolute error, and runs a series of trials to find the best model parameters.

We optimized our machine learning model's architecture by implementing a RandomSearch strategy with Keras Tuner. The objective was to minimize the validation mean absolute error (MAE), which aligns with our business goal of achieving high accuracy in predicting taxi trip demand.

Our tuning process is defined in the `tuner_fn` function, which sets up the RandomSearch tuner with the objective of minimizing the validation mean absolute error. A maximum of 25 trials were conducted, with each trial exploring a unique set of hyperparameters to build and evaluate a model.

The model-building function, `_build_keras_model`, dynamically creates a model based on the provided hyperparameters, such as the number of layers, activation functions, units per layer, and regularization rates. This allows for a thorough investigation of the hyperparameter space, ensuring the best model configuration is identified.

The early stopping callback is utilized to prevent overfitting, monitoring the validation mean absolute error and stopping the training process if no improvement is observed after a specified number of epochs.

Here's a snippet of the code that illustrates the hyperparameter tuning and model optimization process:

```python
def _build_keras_model(hp, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    # Define feature specs and create input layers
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)
    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key) for key in feature_spec.keys()}

    # Concatenate all input features
    concatenated_inputs = tf.keras.layers.Concatenate()(list(inputs.values()))

    # Define model architecture dynamically based on hyperparameters
    num_layers = hp.Int('num_layers', 1, 5)
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
        concatenated_inputs = tf.keras.layers.Dense(units=units, activation=hp.Choice('activation', ['relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid']),
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_{i}', 1e-5, 1e-2, sampling='log')))(concatenated_inputs)
        if hp.Boolean(f'dropout_{i}'):
            concatenated_inputs = tf.keras.layers.Dropout(hp.Float(f'dropout_rate_{i}', 0.1, 0.5))(concatenated_inputs)

    # Output layer for regression
    output = tf.keras.layers.Dense(1, activation='linear')(concatenated_inputs)

    # Compile the model with the chosen optimizer, loss function, and metrics
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

# Tuner function that sets up the RandomSearch tuner
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    tuner = RandomSearch(
        hypermodel=lambda hp: _build_keras_model(hp, tf_transform_output),
        objective='val_mean_absolute_error',
        max_trials=25,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='taxi_trips_tuning_RandomSearch'
    )

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output, _BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output, _BATCH_SIZE)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': 1500,  
            'validation_steps': 969,  
            'callbacks': [early_stopping]
        }
    )

# Hyperparameters tuning with the tfx extension for Google Cloud AI Platform
tuner = tfx.extensions.google_cloud_ai_platform.Tuner(
        module_file=_taxi_trainer_module_file,
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=100),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=5),
         custom_config={
        tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
        tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
        tfx.extensions.google_cloud_ai_platform.experimental.TUNING_ARGS_KEY: vertex_job_spec
        # tfx.extensions.google_cloud_ai_platform.experimental.REMOTE_TRIALS_WORKING_DIR_KEY: os.path.join('gs://', GCS_BUCKET_NAME, 'tuner_trials')
    }
   )

This approach allowed us to systematically search through the hyperparameter space and apply early stopping to prevent overfitting. The use of Google Cloud AI Platform's Tuner enabled us to take advantage of scalable infrastructure and manage the tuning process more effectively. Our choice of hyperparameters included the number of layers, types of activation functions, units per layer, regularization rates, and learning rates, all of which were critical in developing a robust model for our taxi demand prediction task.


```
#### Model Evaluation Metric

In our model's assessment, we primarily utilize the Mean Absolute Error (MAE) complemented by the Root Mean Squared Error (RMSE). These metrics are pivotal for gauging performance and are closely aligned with our core business objectives.

##### Why MAE is Optimal:

- **Business Alignment**: MAE provides an intuitive gauge of prediction accuracy in the same unit as our target variable, the number of taxi trips. This direct correlation facilitates effective communication with business stakeholders.
- **Robustness to Outliers**: The MAE metric's insensitivity to outliers ensures our model evaluation is not disproportionately affected by anomalous data, maintaining consistent performance standards.
- **Interpretability**: The straightforwardness of MAE as an average error metric makes it an invaluable tool for operational decision-making, offering a transparent view into the predictive capabilities of our model.

##### Significance of RMSE:

- **Complementing MAE**: While MAE is excellent for general performance measurement, RMSE is crucial for identifying when the model is prone to larger errors, due to its greater sensitivity to substantial deviations in the data.
- **Error Squaring**: The squaring of errors in RMSE means large errors have a disproportionately large effect on the metric, providing insight into the variability of the model's performance, which can be critical for certain business outcomes.

##### Configuration of Model Evaluation:

We leverage the TensorFlow Model Analysis (TFMA) library for a robust and granular evaluation configuration (`eval_config`). This configuration allows us to monitor our model's MAE, setting thresholds that reflect meaningful improvements in prediction accuracy.

```python
# Define the evaluation configuration using TensorFlow Model Analysis
import tensorflow_model_analysis as tfma

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(signature_name='serving_default', label_key='num_taxi_trips')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='MeanAbsoluteError'),
                tfma.MetricConfig(class_name='RootMeanSquaredError')
            ]
        ),
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(
                    class_name='MeanAbsoluteError',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(lower_bound={'value': MAE_LOWER_BOUND}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.LOWER_IS_BETTER,
                            absolute={'value': MAE_CHANGE_THRESHOLD}
                        )
                    )
                )
            ]
        )
    ]
)


```
#### Bias/Variance Trade-off Considerations

In the development of our machine learning model for taxi demand prediction, we meticulously evaluated and managed the bias-variance trade-off. These trade-offs played a pivotal role in designing an architecture that not only meets our predictive performance criteria but also aligns with our business goals.

##### Layer Configuration and Activation Functions

We calibrated the model's architecture by employing Randomsearch techniques to determine the optimal number of layers and neurons. Activation functions were carefully selected for each layer to ensure non-linearity in the learning process. This fine-tuning aimed to mitigate underfitting (high bias) by enhancing the model's complexity, while also preventing overfitting (high variance) by avoiding an excessively intricate structure.

##### Regularization and Dropout

To address overfitting, we integrated L2 regularization into our model, which imposes a penalty on weight magnitudes and encourages the learning of small weights, effectively simplifying the model. Additionally, dropout layers were strategically placed within the network, randomly disabling neurons during training. This not only prevents the model from becoming overly dependent on any particular neurons (reducing variance) but also promotes a distributed representation within the network.

##### Hyperparameter Tuning

The architecture and learning process were further refined through extensive hyperparameter tuning using the Keras Tuner. This systematic approach allowed us to explore the hyperparameter space efficiently, identifying configurations that balance model complexity (to reduce bias) with generalization capabilities (to control variance).

```python
# Trainer component code snippet
trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
    module_file=os.path.abspath(module_file),
    transformed_examples=transform.outputs['transformed_examples'],
    schema=schema_gen.outputs['schema'],
    transform_graph=transform.outputs['transform_graph'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=tfx.proto.TrainArgs(splits=['train'], num_steps=10160),
    eval_args=tfx.proto.EvalArgs(splits=['eval'], num_steps=5716),
    custom_config={
      tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
      tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
      tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_job_spec
    }
)
```
Our commitment to balancing bias and variance reflects in the robustness and reliability of our predictive model. By leveraging RandomSearch for hyperparameter tuning and incorporating L2 regularization and dropout techniques, our model achieves excellent performance while remaining interpretable and relevant for business decisions.
### 3.1.3.7 Machine Learning Model Evaluation
In our model evaluation process, we incorporated advanced techniques to ensure the deployment of a highly accurate and reliable model. One key aspect of this process is the use of model blessing based on the Mean Absolute Error (MAE) metric and its improvement over time.

Our model's performance on the independent validation dataset is evaluated using TensorFlow Model Analysis (TFMA) with a primary focus on Mean Absolute Error (MAE). MAE provides a clear indication of the average prediction error made by the model in units identical to the predicted variable, making it a highly interpretable and relevant metric for our business needs.Our evaluation setup uses thresholds to determine the model's performance and its improvement over previous iterations.

#### Evaluation Configuration
We configured the `eval_config` in TensorFlow Model Analysis (TFMA) to focus on Mean Absolute Error (MAE) as our primary metric. This choice aligns with our goal of minimizing the average prediction error in taxi trip demand forecasting.

```python
   eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(signature_name='serving_default', label_key=taxi_constants.LABEL_KEY)],
        slicing_specs=[tfma.SlicingSpec()], 
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name='MeanAbsoluteError'),
                    tfma.MetricConfig(
                        class_name='MeanAbsoluteError',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.8159}),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.LOWER_IS_BETTER,
                                absolute={'value': 0.02}
                            )
                        )
                    )
                ]
            )
        ]
    )
```
#### Model Resolver
We integrated a model resolver with the `LatestBlessedModelStrategy` to ensure that only models which demonstrate an improvement over previously 'blessed' models are deployed. This approach ensures a consistent enhancement in model performance, aligning with our commitment to delivering the most accurate and reliable predictions.

```python
model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')
```
#### Running the Evaluation
The Evaluator component in TFX is instrumental in executing this rigorous model evaluation process, assessing the model's performance against the set thresholds
```python
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)
```
#### Outcome
Our model achieved a Mean Absolute Error (MAE) of 2.1684, surpassing the established lower bound and demonstrating a significant improvement over previous versions.

Best MAE Achieved: 2.1684

This advanced evaluation methodology, involving meticulous metric configuration and the strategic use of model resolvers, exemplifies our dedication to deploying a model that excels in both accuracy and continuous improvement, ensuring the highest standards of prediction quality in taxi trip demand forecasting.
## 3.1.4 Proof of Deployment
**Note:** For a detailed walkthrough of our deployment process on Vertex AI, including code examples and configurations, refer to our [Vertex AI-Pipeline Notebook](Notebooks/Vertex%20AI-Pipeline.ipynb).

#### End-to-End TFX Pipeline for Taxi Demand Prediction
![Pipeline Runtime Graph](assets/Run%20time%20graph.PNG)

#### Workflow Execution Illustrated by DAG

The DAG image visually represents the sequential execution and interdependencies of each pipeline component. This visual aid is crucial for understanding the flow and interaction of data and processes throughout the pipeline.
#### Introduction

In this project, we have developed a sophisticated end-to-end TensorFlow Extended (TFX) pipeline for Taxi Demand Prediction. This README outlines each component's role in the pipeline, illustrated by a DAG image that provides a comprehensive view of the workflow execution. Additionally, we have also developed an interactive pipeline to further enhance the development process and provide immediate insights at each stage.

#### Interactive Pipeline Development

In addition to the end-to-end TFX pipeline, we have developed an interactive pipeline that provides immediate output at each component stage. This interactive approach allows for more dynamic development and quicker iterations, making the process more efficient and user-friendly.

#### Pipeline Overview

The pipeline comprises several key components, each responsible for a specific function in the machine learning workflow:

#### 1. CsvExampleGen
- **Function**: Manages data ingestion, organizing it into distinct splits for training, evaluation, and testing.
- **Immediate Output**: Automatically partitions the dataset into dedicated datasets for subsequent stages.
  ![CsvExampleGen Output](assets/CsvExampleGen%20Output.png)
  *Screenshot: Output of the CsvExampleGen component showing the partitioned datasets.*
#### 2. StatisticsGen
- **Function**: Generates comprehensive statistics of the dataset to understand its distribution and characteristics.
- **Immediate Output**: Provides descriptive statistics of the dataset, offering initial insights into data quality and features.

#### 3. SchemaGen
- **Function**: Infers a schema from the dataset based on generated statistics, outlining the structure, types, and properties of data fields.
- **Immediate Output**: Produces a schema of the dataset, essential for data validation and understanding.

#### 4. ExampleValidator
- **Function**: Uses the schema to detect anomalies and inconsistencies within the dataset.
- **Immediate Output**: Flags any data anomalies or inconsistencies, ensuring high data quality. E.g., outputting 'No anomalies found' across different data splits.

#### 5. Transform
- **Function**: Performs feature engineering on the dataset, enhancing the data's predictive power for the model.
- **Immediate Output**: Outputs transformed datasets with newly engineered features, optimized for model performance.

#### 6. Tuner
- **Function**: Optimizes the model's hyperparameters to ensure peak performance.
- **Immediate Output**: Delivers the best set of hyperparameters discovered for the model.

#### 7. Trainer
- **Function**: Trains machine learning models using the transformed data.
- **Immediate Output**: Yields a trained model, ready for evaluation and potential deployment.

#### 8. Evaluator
- **Function**: Assesses the trained model against pre-defined metrics and validation criteria.
- **Immediate Output**: Provides key evaluation metrics indicating the model's performance.

#### 9. Resolver
- **Function**: Manages various model versions, selecting the best one based on performance metrics.
- **Immediate Output**: Selects the optimal model version for deployment.

#### 10. Pusher
- **Function**: Responsible for deploying the selected model version, making it ready for practical application.
- **Immediate Output**: Ensures the model is deployed to a specified server or cloud environment, confirming successful deployment.

#### Local Testing with DAG Runner

As part of our development process, we executed our TFX pipeline locally using a DAG runner. This approach allowed us to iteratively test and refine our pipeline in a controlled environment.

#### Executing Local TFX Pipeline

Below is the Python script used for executing the pipeline locally:

```python
import tfx
from tfx import v1 as tfx
import kfp
from tfx.orchestration.metadata import sqlite_metadata_connection_config

tfx.orchestration.LocalDagRunner().run(
    _create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_DIRECTORY,
        module_file=_taxi_trainer_module_file,
        serving_model_dir=SERVING_MODEL_DIR
    )
)
```
##### Output from the pipeline execution
```plaintext
-----------------------------------
Trial 10 Complete [00h 00m 12s]
val_mean_absolute_error: 0.7877678871154785

Best val_mean_absolute_error So Far: 0.7770861387252808
Total elapsed time: 00h 02m 08s

Model Summary
-------------
Model: "model_1"
_________________________________________________________________
Layer (type)                       Output Shape            Param #   
=================================================================
(Model layers and their connections)
_________________________________________________________________
Total params: 206,913
Trainable params: 206,913
Non-trainable params: 0
_________________________________________________________________
100/100 [==============================] - ETA: 0s - loss: 87.7152 - mean_absolute_error: 4.3698 - root_mean_squared_error: 9.3574
Epoch 1: val_mean_absolute_error improved from inf to 0.77404, saving model to gs://chicago_taxitrips/pipeline_root/c-prediction/Trainer/model_run/37/model/best_model
100/100 [==============================] - 9s 71ms/step - loss: 87.7152 - mean_absolute_error: 4.3698 - root_mean_squared_error: 9.3574 - val_loss: 1.5186 - val_mean_absolute_error: 0.7740 - val_root_mean_squared_error: 1.1683
Exporting the serving model to gs://chicago_taxitrips/pipeline_root/c-prediction/Trainer/model/37/Format-Serving
Model exported successfully.
```
#### Conclusion of Local Testing

The local testing phase, conducted using the DAG runner, has been an essential step in our development process. It allowed us to thoroughly validate each component of our TFX pipeline in a controlled and iterative manner. This approach ensured that any issues could be promptly identified and rectified, thereby enhancing the robustness and reliability of our pipeline.

Through this rigorous testing, we have gained valuable insights into the performance and efficacy of our model. The immediate feedback and detailed outputs obtained from the local execution have not only affirmed the pipeline's functionality but also provided a foundation for further optimizations. 

As we progress from this successful local testing phase to deploying our pipeline in a production environment on Vertex AI, we are confident in the stability and readiness of our solution for real-world applications. The transition from a local DAG runner to a vertex AI environment signifies a pivotal step towards delivering a scalable and impactful machine learning model for Taxi Demand Prediction.

#### Deployment on Vertex AI

Our TFX pipeline, initially tested locally, has been adapted for deployment on Vertex AI. While the components remain consistent with our interactive pipeline, the setup for Vertex AI necessitates specific configurations and the use of additional features offered by Vertex AI.

#### Pipeline Adaptation for Vertex AI

In the adaptation process for Vertex AI, we retain the same components as in our interactive pipeline. However, we incorporate configurations specific to Vertex AI environments, such as specifying the Vertex AI runner, setting up the necessary GCP project details, and configuring the pipeline's data paths and storage locations.

#### Trainer Component for Vertex AI

The trainer component is adapted for Vertex AI with specific hyperparameter tuning and model training strategies. We use TensorFlow, Keras, and additional TensorFlow Transform for feature engineering.

```python
# Trainer component setup with Keras and TensorFlow Transform
_taxi_trainer_module_file = 'taxi_trainer.py'
# Trainer code includes model definition, training configuration, and 
# hyperparameter tuning setup.

# Example snippet of the trainer module:
%%writefile {_taxi_trainer_module_file}

_LABEL_KEY = taxi_constants.LABEL_KEY


_BATCH_SIZE = 64

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', RandomSearch), ('fit_kwargs', Dict[Text, Any])])  # Changed to Randomsearch


early_stopping = EarlyStopping(
    monitor='val_mean_absolute_error',
    patience=10,
    restore_best_weights=True
)

def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = _BATCH_SIZE) -> tf.data.Dataset:
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
        tf_transform_output.transformed_metadata.schema).repeat()


def _build_keras_model(hp, tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    # Define feature specs and create input layers
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    # Remove the label feature
    feature_spec.pop(_LABEL_KEY)
    inputs = {
        key: tf.keras.layers.Input(shape=(1,), name=key)
        for key in feature_spec.keys()
    }

    # Concatenate all input features
    concatenated_inputs = tf.keras.layers.Concatenate()(list(inputs.values()))

    num_layers = hp.Int('num_layers', 1, 5)
    activation_choice = hp.Choice('activation', ['relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid'])

    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
        concatenated_inputs = tf.keras.layers.Dense(
            units=units,
            activation=activation_choice,
            kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_{i}', 1e-5, 1e-2, sampling='log'))
        )(concatenated_inputs)
        if hp.Boolean(f'dropout_{i}'):
            dropout_rate = hp.Float(f'dropout_rate_{i}', 0.1, 0.5)
            concatenated_inputs = tf.keras.layers.Dropout(dropout_rate)(concatenated_inputs)

    # Output layer for regression
    output = tf.keras.layers.Dense(1, activation='linear')(concatenated_inputs)

    # Create and compile the Keras model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='mean_squared_error',
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    model.summary()
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    tuner = RandomSearch(
        hypermodel=lambda hp: _build_keras_model(hp, tf_transform_output),
        objective='val_mean_absolute_error',
        max_trials=25,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='taxi_trips_tuning_RandomSearch'
    )

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output, _BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output, _BATCH_SIZE)
    )
def _get_transform_features_signature(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        return transformed_features

    return transform_features_fn

def export_serving_model(tf_transform_output, model, output_dir):
    print(f"Exporting the serving model to {output_dir}")
    model.tft_layer = tf_transform_output.transform_features_layer()
    signatures = {
        'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(output_dir, save_format='tf', signatures=signatures)
    print("Model exported successfully.")

    
def run_fn(fn_args: FnArgs):
    print(f"Model run directory: {fn_args.model_run_dir}")
    
    model_dir = os.path.join(fn_args.model_run_dir, 'model')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_absolute_error', 
        mode='min', 
        patience=3,
        restore_best_weights=True
    )
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    # Fit the model with the callbacks
    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=fn_args.train_steps // steps_per_epoch,
        callbacks=[
            # tensorboard_callback,
            early_stopping_callback,
            model_checkpoint_callback
        ]
    )
  
    signatures = {
      'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output),
    }
    

    # At the end of the run_fn function
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)
```
#### Pipeline Definition for Vertex AI
The pipeline definition for Vertex AI involves creating a pipeline object that integrates all the TFX components, configured to run in the Vertex AI environment.
# Example snippet of the trainer module:
```python
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str) -> tfx.dsl.Pipeline:
    # Component setup (CsvExampleGen, StatisticsGen, etc.)
    ...
    # Integration of components into the pipeline
    components = [example_gen, statistics_gen, ..., trainer, pusher]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components
    )
```
#### Vertex AI Pipeline Execution

After configuring the TFX components and adapting the trainer for Vertex AI, the next step involves setting up the `KubeflowV2DagRunner`. This runner is utilized for executing the pipeline in the Vertex AI environment.

#### Setting Up KubeflowV2DagRunner

To deploy our pipeline on Vertex AI, we use the `KubeflowV2DagRunner`. This runner facilitates the translation of our TFX pipeline into a format compatible with Vertex AI Pipelines. 

```python

# Configure the KubeflowV2DagRunner
runner = KubeflowV2DagRunner(
    config=KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE
)

# Run the pipeline using the runner
_ = runner.run(
    _create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_DIRECTORY,
        module_file=_taxi_trainer_module_file,
        endpoint_name=ENDPOINT_NAME,
        project_id=GOOGLE_CLOUD_PROJECT,
        region=GOOGLE_CLOUD_REGION,
        serving_model_dir=SERVING_MODEL_DIR
    )
)
```
#### Submitting the Pipeline to Vertex AI
With the pipeline compiled by the KubeflowV2DagRunner, the next step is to submit this pipeline to Vertex AI for execution. This process is handled through the Vertex AI SDK.

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

# Initialize the AI Platform
aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

# Create and submit a pipeline job
job = pipeline_jobs.PipelineJob(
    template_path=PIPELINE_DEFINITION_FILE,
    display_name=PIPELINE_NAME
)
job.submit()
```
#### Pipeline Job Submission Output
```plaintext
Creating PipelineJob
INFO:google.cloud.aiplatform.pipeline_jobs:Creating PipelineJob
PipelineJob created. Resource name: projects/75674212269/locations/us-central1/pipelineJobs/taxi-demand-prediction-management-20231213030329
INFO:google.cloud.aiplatform.pipeline_jobs:PipelineJob created. Resource name: projects/75674212269/locations/us-central1/pipelineJobs/taxi-demand-prediction-management-20231213030329
To use this PipelineJob in another session:
INFO:google.cloud.aiplatform.pipeline_jobs:To use this PipelineJob in another session:
pipeline_job = aiplatform.PipelineJob.get('projects/75674212269/locations/us-central1/pipelineJobs/taxi-demand-prediction-management-20231213030329')
INFO:google.cloud.aiplatform.pipeline_jobs:pipeline_job = aiplatform.PipelineJob.get('projects/75674212269/locations/us-central1/pipelineJobs/taxi-demand-prediction-management-20231213030329')
View Pipeline Job:
https://console.cloud.google.com/vertex-ai/locations/us-central1/pipelines/runs/taxi-demand-prediction-management-20231213030329?project=75674212269
INFO:google.cloud.aiplatform.pipeline_jobs:View Pipeline Job:
https://console.cloud.google.com/vertex-ai/locations/us-central1/pipelines/runs/taxi-demand-prediction-management-20231213030329?project=75674212269
```
This submission triggers the execution of our pipeline in the cloud, leveraging the powerful resources and managed services provided by Vertex AI. It marks a critical step in deploying our Taxi Demand Prediction model in a scalable and robust cloud environment.

#### 3.1.4.1 Model/Application on Google Cloud

#### Overview

The deployment of our Taxi Demand Prediction model on Google Cloud marks a significant milestone in showcasing our project's adaptability and innovation in machine learning. Highlighting our agile development approach, we successfully deployed two iterations of the model: an initial version following training and a second, customized version, reflecting our ability to refine and enhance our solutions based on evolving requirements and feedback.

#### Initial Deployment of the Trained Model

The first deployment phase involved deploying the trained model directly onto Google Cloud. This step was crucial to test the model's performance in a cloud environment and to ensure seamless integration with Google Cloud's services.

- **Deployment Process**: The trained model was deployed using Vertex AI, which provided the necessary infrastructure and scalability.
- **Verification**: Post-deployment, the model's functionality and performance were verified to ensure it met our expected standards.

#### Customization and Second Deployment

To demonstrate the versatility of our solution, we modified the model based on initial feedback and specific requirements. This phase highlights our model's customizability and our ability to iterate rapidly.

- **Model Editing**: Adjustments were made to the model, improving its accuracy and efficiency based on the insights gathered from the initial deployment.
- **Second Deployment**: The updated model was redeployed on Google Cloud, showcasing our ability to respond to changes and enhance our solution promptly.

#### Demonstrating Customizability

These two deployments underline our model's capacity for customization and adaptability. It proves our commitment to providing a solution that is not only effective but also flexible enough to evolve according to changing needs and insights.

- **Images of Deployed Models**: _[Add images or screenshots of the deployed models on Google Cloud]_
- **Continuous Improvement**: This process exemplifies our approach to continuous improvement and adaptation in the rapidly evolving field of machine learning and data science.
#### 3.1.4.2 Callable Library/Application

#### Overview of Prediction Endpoints

In our Taxi Demand Prediction project, we have established robust online prediction endpoints on Google Cloud. These endpoints, named "DNN-Customizable-endpoint" and "DNN-Demand-Endpoint," are integral for providing real-time predictions and interacting with our deployed models.

#### Endpoints Details

#### DNN-Customizable-Endpoint

- **Endpoint ID**: 8334344318030446592
- **Status**: Deployed
- **Deployment Resource Pool**: To be updated
- **Region**: us-central1
- **Monitoring**: No alerts yet
- **Last Updated**: 15 Dec 2023, 05:08:20
- **Description**: This endpoint is linked to the "DNN-Customizable" model, showcasing the model's ability to adapt and provide tailored predictions based on varying inputs.

#### DNN-Demand-Endpoint

- **Endpoint ID**: 4013140475568455680
- **Status**: Deployed
- **Deployment Resource Pool**: [Add deployment resource pool if applicable]
- **Region**: us-central1
- **Monitoring**: No alerts yet
- **Last Updated**: 15 Dec 2023, 05:06:38
- **Description**: Associated with the "DNN-Demand" model, this endpoint serves real-time predictions regarding taxi demand based on the provided data.
#### Online Prediction with Deployed Endpoints

Our project demonstrates the practical application of the deployed models by enabling online predictions through the Vertex AI endpoints. We use a Python script to serialize feature data and interact with the model endpoints, fetching real-time predictions based on various input features.

#### Example of Online Prediction Process

Here's how we prepare and send a prediction request to our model's endpoint:

```python


#### Example raw features for prediction
raw_features = {
    'day': 26, 'year': 2022, 'month': 1, 'hour': 1, 'pickup_community_area': 16,
    'duration': 2.0, 'trip_miles': 9.225, 'trip_total': 2.35,
    'temperature_2m': 9.25, 'relativehumidity_2m': 84, 'precipitation': 0.0,
    'rain': 0.0, 'snowfall': 0.0, 'weathercode': 3, 'public_holiday': 0,
    'hour_sin': -0.51958395003511026, 'hour_cos': 0.85441940454668519,
    'day_sin': 0.0, 'day_cos': 1.0, 'month_sin': 0.50000000000002986,
    'month_cos': 0.8660254037
}
#### Make the prediction request
response = make_prediction(
    project_id="75674212269",
    endpoint_id="547902037283569664",
    location="us-central1",
    instance=instance
)

#### Print the prediction response
print("Prediction response:", response)
```
#### Prediction Response from DNN-Customizable

The response received from the "DNN-Customizable" endpoint after submitting the prediction request is as follows:

```plaintext
Prediction response: predictions {
  list_value {
    values {
      number_value: 1.1016165
    }
  }
}
deployed_model_id: "7298687927549165568"
model: "projects/75674212269/locations/us-central1/models/5446485823769804800"
model_display_name: "DNN-Customizable"
model_version_id: "1"
```
#### Prediction Response from DNN-Demand Endpoint
```plaintext
Prediction response: predictions {
  list_value {
    values {
      number_value: 1.39575148
    }
  }
}
deployed_model_id: "8119926356474593280"
model: "projects/75674212269/locations/us-central1/models/5538528141154189312"
model_display_name: "DNN-Demand"
model_version_id: "1"
```
### 3.2.4.3 Editable Model/Application

#### Overview

In our Taxi Demand Prediction project, we have developed an editable and customizable version of our Deep Neural Network (DNN) model, "DNN-Customizable." This version demonstrates our commitment to flexibility and adaptability in our machine learning solutions.

#### Training Differences

#### Initial DNN Model
- **Features Used**: The initial version of our DNN model was trained using both transformed and raw features, leveraging a comprehensive dataset to predict taxi demand.
- **Tuning Method**: Utilized a standard approach for hyperparameter tuning, ensuring optimal model performance based on the given data.

#### DNN-Customizable Model
- **Features Used**: In contrast, the "DNN-Customizable" model was trained exclusively with transformed features. This approach focuses on the most impactful data representations, allowing for more streamlined and efficient predictions.
- **Tuning Method Changes**: The hyperparameter tuning for the customizable model was adjusted to include a higher maximum number of trials. This change was implemented to explore a broader range of hyperparameter configurations, enhancing the model's ability to adapt to varying data patterns and complexities.

#### Customizability and Adaptation

The "DNN-Customizable" model represents a significant step towards creating machine learning solutions that can be easily adapted and refined. By training the model with different sets of features and employing a more extensive hyperparameter search, we've built a system that can adjust more readily to new data, evolving requirements, and specific prediction scenarios.

This level of customizability ensures that our model remains relevant and effective, even as the underlying data or business objectives change. It underscores our project's focus on creating not just powerful predictive tools, but also versatile and dynamic solutions that can grow and evolve in line with real-world demands.

The development of the "DNN-Customizable" model is a testament to our project's innovative approach to machine learning. It showcases our ability to not only create robust predictive models but also to engineer solutions that are flexible and adaptable, catering to the ever-changing landscape of data-driven decision making.

### Conclusion: Realizing Our Vision in Taxi Demand Management
Our project in Taxi Demand Prediction and Management transcends the realms of advanced analytics and machine learning to align with a specific, mission-driven business goal. This endeavor is a fusion of data science with practical application, all geared towards transforming taxi services in urban landscapes.
#### Impact on Taxi Services
- **Insightful Data Analysis**: Our detailed EDA, emphasizing hourly data granularity, has revealed critical insights into the dynamics of taxi demand, significantly influencing time, date, and geographic factors.
- **Enhanced Operational Efficiency**: Our project's core aligns with the operational goal of optimizing taxi distribution efficiency. Accurate demand predictions allow for better resource allocation, ensuring high availability during peak demand periods.

#### Achieving the Business Goal
- **Streamlined Taxi Distribution**: Our project's focus has always been to improve taxi distribution efficiency across the city, aiming to reduce customer wait times through precise demand predictions.
- **Reduced Customer Wait Times**: Our predictive models lay the groundwork for more responsive taxi services, enabling taxi companies to proactively meet demand fluctuations and improve service quality.

#### Use Case Realization
- **Practical Application of Predictions**: Our models realize the goal of predicting taxi demand based on key variables, thus empowering taxi companies to optimize their fleet distribution effectively.
- **Contribution to Urban Mobility**: Beyond improving taxi services, our project plays a role in enhancing the broader urban transportation ecosystem, contributing to the revolution of urban mobility.

#### Future Directions
- **Continuous Improvement**: We anticipate refining and adapting our models with new data, ensuring they remain cutting-edge in predictive accuracy.
- **Expanding Applications**: The methodologies and technologies used in this project have the potential to be adapted for various industries, showcasing the versatility and broad applicability of our approach.
- **Enhancing User Accessibility**: Future efforts will focus on developing more user-friendly interfaces and integration points, making our models more accessible to a wider audience.

#### Final Reflections
- **Innovation at the Core**: This project highlights the importance of innovation in addressing complex urban challenges like taxi demand.
- **Empowering Decision-makers**: By providing actionable insights, we demonstrate how AI can be a powerful tool for informed decision-making in urban planning and transportation management.
- **Foundation for Future Exploration**: We've set a robust foundation for future advancements in AI and machine learning, particularly for real-world problem-solving.

In conclusion, the Taxi Demand Prediction and Management project marks a significant stride in our endeavor to merge technological innovation with practical utility. Our focused approach, grounded in real-world challenges and aimed at achieving tangible improvements, showcases how data science can be a driving force in reshaping industries and enhancing everyday experiences.

### Resources

#### Evalution critera

| Item | Requirement | Description |
| --- | --- | --- |
| **3.1.1 Code** | | |
| 3.1.1.1 Code repository | Partners must provide a link to the code repository. | The repository must contain a ReadMe file with code descriptions and detailed instructions for running the model/application. |
| 3.1.1.2 Code origin certification | Partners must certify the origin of the code. | Evidence must include a certification by the partner organization for the code origin scenarios. |
| **3.1.2 Data** | | |
| 3.1.2.1 Dataset in Google Cloud | Partners must provide documentation of data location. | Evidence must include the project name and project ID for the Google Cloud Storage bucket or BigQuery dataset. |
| **3.1.3 Whitepaper / Blog** | | |
| 3.1.3.1 Business goal and machine learning solution | Partners must describe the business goal and machine learning use case. | Evidence must include a top-line description of the business goal and the proposed machine learning solution. |
| 3.1.3.2 Data exploration | Partners must describe the data exploration performed. | Evidence must include a description of the tools used and types of data exploration performed, along with code snippets. |
| 3.1.3.3 Feature engineering | Partners must describe the feature engineering performed. | Evidence must include a description of the feature engineering and feature selection steps, with code snippets. |
| 3.1.3.4 Preprocessing and the data pipeline | Partners must describe the data preprocessing pipeline. | Evidence must include a description of how data preprocessing is accomplished using callable APIs. |
| 3.1.3.5 Machine learning model design(s) and selection | Partners must describe the machine learning model selection. | Evidence must describe the selection criteria and the specific machine learning model algorithms with code snippets. |
| 3.1.3.6 Machine learning model training and development | Partners must document the use of Vertex AI or Kubeflow for training. | Evidence must describe the machine learning model training and development points with code snippets. |
| 3.1.3.7 Machine learning model evaluation | Partners must describe the model evaluation. | Evidence must include records/data of the model performance on an independent test dataset. |
| **3.1.4 Proof of Deployment** | | |
| 3.1.4.1 Model/application on Google Cloud | Partners must provide proof of model/application deployment. | Evidence must include the Project Name and Project ID of the deployed model and client. |
| 3.1.4.2 Callable library/application | Partners must demonstrate the model as a callable library/application. | Evidence must include a demonstration of the served model making a prediction via an API call. |
| 3.1.4.3 Editable model/application | Partners must demonstrate that the deployed model is customizable. | Evidence must include a demonstration of the model fully functional after code modification. |

#### References

1. Zhao, Q., Yang, G., Zhao, K., Yin, J., Rao, W., Chen, L. (2023). Multivariate Time-Series Forecasting Model: Predictability Analysis and Empirical Study. IEEE Transactions on Big Data, 9(6), 1536-1548.

2. Bagal, N. M., Gabhane, M. D., Mahamuni, C. V. (2023). Rideshare Transportation Fare Prediction using Deep Neural Networks. 2023 International Conference on Disruptive Technologies (ICDT), 643-649.

3. Wu, Z., Zheng, L., Zhang, C. J., Zhu, H., Yin, J., Jiang, D. (2023). Opponent-aware Order Pricing towards Hub-oriented Mobility Services. 2023 IEEE 39th International Conference on Data Engineering (ICDE), 1874-1886.

4. Gao, P., Yang, X., Zhang, R., Huang, K., Goulermas, J. Y. (2023). Explainable Tensorized Neural Ordinary Differential Equations for Arbitrary-Step Time Series Prediction. IEEE Transactions on Knowledge and Data Engineering, 35(6), 5837-5850.

5. Zhang, C., Zhao, K., Chen, M. (2023). Beyond the Limits of Predictability in Human Mobility Prediction: Context-Transition Predictability. IEEE Transactions on Knowledge and Data Engineering, 35(5), 4514-4526.

6. Venkata Ramana, A. Dr., Batool, A., Ramavath, M., & Viveka, P. (2022). Taxi Demand Prediction using ML. International Journal of Research in Applied Science and Engineering Technology (IJRASET). [https://doi.org/10.22214/ijraset.2022.43912](https://doi.org/10.22214/ijraset.2022.43912)

7. Gong, L., Liu, X., Wu, L., & Liu, Y. (2015). Inferring trip purposes and uncovering travel patterns from taxi trajectory data. Cartography and Geographic Information Science, 103-114. [https://doi.org/10.1080/15230406.2015.1014424](https://doi.org/10.1080/15230406.2015.1014424)
