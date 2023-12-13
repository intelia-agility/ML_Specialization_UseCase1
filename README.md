
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

In this project, data exploration is a multi-stage process that begins with data enrichment and is followed by a thorough exploratory analysis.

#### Overview
To deepen our analysis, we have enriched the Chicago Taxi Trip dataset with additional weather data. This process is key to exploring the impact of various weather conditions on taxi trip patterns.

#### Integrating Weather Data
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

#####  Data Granularity and Analysis
Our analysis emphasizes hourly data granularity. This approach helps us capture the dynamic nature of both weather conditions and taxi trip demand, allowing for a more nuanced understanding of their interplay. Hourly data provides the detail necessary for accurate demand forecasting, crucial for operational planning and resource allocation.
##### Code Snippet:
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

##### Citation
Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649

#### Exploratory Data Analysis (EDA)

#####  Overview
The Exploratory Data Analysis section provides insights into the dataset through various angles and techniques. Each subsection below corresponds to a key aspect of the EDA, complemented by visualizations that highlight our findings.

##### Descriptive Statistical Analysis
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

- **Visualization Placeholder for Descriptive Statistics**:
  ![Descriptive Statistics Visualization Placeholder](path/to/descriptive_statistics_visual.png)

This comprehensive statistical overview was pivotal in guiding our subsequent analyses, enabling us to identify areas requiring deeper investigation and to hypothesize about various factors influencing taxi demand.

##### Feature Selection and Data Inspection

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

##### Handling Missing Values and Data Integrity
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

##### Continuous and Categorical Features Analysis

- **Continuous Features**: Analyzed key continuous features like `trip_seconds`, `trip_miles`, `trip_total`, and various weather-related attributes (temperature_2m, relativehumidity_2m, precipitation, rain, snowfall). Histograms with kernel density estimates provided insights into the distribution of these variables, highlighting trends and anomalies.

- **Categorical Features**: Features such as `company` and `weathercode` were analyzed using count plots to understand their frequency distribution. This helped us grasp the diversity and prominence of different taxi companies and weather conditions during taxi trips.

##### Observations from Univariate Analysis

- **Continuous Features**: Most trip durations, distances, and fares were on the lower side, with right-skewed distributions, indicating that shorter and less expensive trips were more common.
- **Weather-related Features**: The analysis revealed patterns in climate conditions and their potential impact on taxi demand.
- **Categorical Features**: Distribution of taxi companies and weather conditions provided insights into operational dynamics and the impact of weather on taxi usage.

##### Univariate Analysis of Spatial Features

- **Spatial Features Analysis**: Examined spatial features like `pickup_community_area`, `pickup_latitude`, and `pickup_longitude`. Count plots and histograms were employed to explore these features, identifying popular areas for taxi pickups and potential hotspots.
- **Observations**: Some community areas had significantly higher pickup frequencies, suggesting they were key hotspots. Latitude and longitude data highlighted the geographical concentration of taxi pickups in certain areas.

Each step of our univariate analysis provided crucial insights into different aspects of taxi demand, laying a foundation for more detailed multivariate analysis and predictive modeling. The use of visualizations at each step enhanced our understanding of the data and helped in identifying key areas for focused analysis and strategy development.

##### Timestamp Conversion and Time Feature Extraction

- **Extracting Time-Related Features**: Converted the `trip_start_timestamp` to datetime format and extracted various time-related features (year, month, day, hour, weekday, trip date, day of the week). This allowed for an in-depth analysis of taxi demand patterns over different time periods.


##### Bivariate Analysis

In this section, we delve into Bivariate Analysis to explore the relationships between two distinct variables and their combined impact on taxi demand. This approach helps us understand the interactions and dependencies between various factors in our dataset.

##### Temporal Analysis

##### Hourly Taxi Demand
- **Description**: Analysis of taxi demand based on hourly data.
- **Observations**:
  - Decrease in demand in the early morning hours, with the lowest point around 5 AM.
  - Increase in demand from 6 AM, peaking during late afternoon and early evening, then decreasing throughout the night.
  - Typical urban dynamics with increased demand during peak hours and reduced demand during off-peak periods.

##### Taxi Demand by Day of the Week
- **Description**: Analyzing the distribution of taxi demand across different days of the week.
- **Observations**:
  - Consistent demand from Monday to Friday, with an increase on Fridays.
  - Drop in demand on Saturdays and the lowest on Sundays.
  - Weekday demand driven by work-related commuting, contrasting with quieter weekends.

##### Taxi Demand by Month
- **Description**: Exploration of how taxi demand varies across different months.
- **Observations**:
  - Steady demand from January through May.
  - Peak in June, followed by a decline in July and stabilization from August to December.
  - June's peak possibly related to seasonal events, holidays, or tourist influx.

##### Categorical Analysis

##### Taxi Demand vs. Weather Code
- **Description**: Investigation of demand variations across different weather conditions as indicated by weather codes.
- **Observations**:
  - Higher demand for lower weather code values (favorable conditions).
  - Decrease in demand with increasing weather code values, suggesting reduced demand in severe or unfavorable weather.

##### Taxi Demand by Weather Condition
- **Description**: Detailed analysis of taxi demand under specific weather conditions.
- **Observations**:
  - Highest demand in sunny and clear weather.
  - Significant demand during less favorable conditions like rain or fog.
  - Lowest demand during extreme conditions like thunderstorms or heavy snow.

##### Spatial Analysis

##### Spatial Analysis of Taxi Pickups by Community Area
- **Description**: Distribution of taxi pickups across different community areas.
- **Observations**:
  - Variations in pickup frequency across areas, influenced by proximity to key destinations or transport hubs.
  - Disparity in average fares, indicating differences in trip lengths or destination popularity.

##### Analysis of Average Fare by Pickup Community Area
- **Description**: Investigation into average fares in different community areas.
- **Observations**:
  - Range in average fares, with certain areas having higher fares on average due to longer trips or routes to in-demand locations.
  - These fare dynamics are crucial for efficient service allocation and pricing strategies.

##### Visualization Placeholders for Bivariate Analysis

- **Insert Graph 1**: Bar chart for Hourly Distribution of Taxi Demand.
- **Insert Graph 2**: Bar chart for Distribution of Taxi Demand by Day of the Week.
- **Insert Graph 3**: Bar chart for Distribution of Taxi Demand by Month.
- **Insert Graph 4**: Scatter plot for Taxi Demand vs. Weather Code.
- **Insert Graph 5**: Bar chart for Taxi Demand by Weather Condition.
- **Insert Graph 6**: Bar chart for Taxi Pickups by Community Area.
- **Insert Graph 7**: Bar chart for Average Fare by Pickup Community Area.

##### Correlation Analysis of Numerical Features

In this section, we explore the intricate associations and dependencies among the numerical features within our dataset through correlation analysis. This approach is pivotal in unraveling the subtle and complex factors that influence taxi demand and the characteristics of taxi trips.

- **Correlation Heatmap**: The heatmap visualizes the correlation coefficients between different numerical features, such as trip miles, trip seconds, trip total, and various weather-related variables like temperature, humidity, and precipitation.
- **Findings**:
  - A positive correlation is observed between trip miles and trip seconds, indicating that trips covering more miles typically have longer durations.
  - Trip total shows a positive correlation with both trip miles and trip seconds, suggesting that longer trips, both in distance and duration, generally lead to higher fares.
  - Weather-related variables display very weak correlations with trip details, suggesting minimal direct linear relationships between these weather factors and the specific attributes of taxi trips.
- **Implications**: This analysis is crucial for understanding how different aspects of taxi trips are interconnected. It provides insights crucial for refining fare structuring, service management, and operational strategies in the taxi service industry.

##### Visualization Placeholder

- **Insert Correlation Heatmap Here**: Visualization showing the correlation matrix among various numerical features.

##### Outlier Analysis

The Outlier Analysis section is dedicated to identifying and understanding anomalies within our dataset. This involves examining various numerical features, continuous variables, and spatial data to pinpoint irregularities that could influence our analysis and model accuracy.

##### Outlier Analysis of Numerical Features

- **Box Plots**: Visualization of outliers in various numerical features using box plots.
- **Findings**:
  - Trip_seconds and trip_total exhibit significant outliers, indicating the presence of unusually long or expensive trips.
  - Trip_miles also shows outliers, but they are not as pronounced.
  - Temperature_2m shows a relatively normal distribution with minimal outliers.
  - Relativehumidity_2m mostly falls within the 60-90% range, indicating a good spread with no significant anomalies.
- **Visualization Placeholder**: (Insert box plots for each numerical feature)

##### Percentile Analysis of Continuous Variables

- **Analysis**: Percentile distribution provides insights into the range and spread of continuous variables like trip_seconds, trip_miles, and trip_total.
- **Observations**:
  - Trip_seconds: The majority of trips are shorter than 38 minutes, with a maximum duration significantly higher, suggesting outliers.
  - Trip_miles: Most trips are under 15.72 miles, with the longest trip recorded at 3430.53 miles.
  - Trip_total: Median fare is around $15.50, with the highest recorded fare being significantly higher.
- **Implications**: This percentile analysis helps identify typical trip characteristics and detect anomalies within the dataset.

##### Scatter Plot Analysis of Taxi Pickup Locations

- **Visualization**: A scatter plot representing the geographical distribution of taxi pickups.
- **Observations**:
  - Dense clusters indicate popular areas or hotspots for taxi pickups.
  - Areas with fewer points suggest less frequent taxi activity, possibly in residential zones or less commercially active areas.
- **Visualization Placeholder**: (Insert scatter plot for pickup latitudes and longitudes)

##### Data Cleaning and Outlier Treatment Process

- **Process Description**: 
  - The cleaning process involves a series of meticulous steps designed to refine the taxi dataset for more accurate analysis.
  - Geographical Filtering: Ensures that all taxi pickups and drop-offs are within the defined Chicago city boundaries, filtering out data points that fall outside these parameters.
  - Removing Zero Values: Trips with zero values for critical variables such as trip_seconds, trip_miles, or trip_total are excluded, as they likely represent data recording errors or irrelevant entries, ensuring the integrity of the dataset.
  - 12-Hour Rule: Trips exceeding a 12-hour duration are removed from the dataset. This step adheres to realistic and legal driving limits, eliminating data points that might be the result of data entry errors or other anomalies.
  - Capping Extreme Values: Extreme values for variables like trip duration, miles, fare, and weather-related measures are capped at their 1st and 99th percentiles. This treatment mitigates the influence of extreme outliers that could skew the analysis, ensuring a more balanced and representative dataset.

- **Data Retention**: 
  - Post-cleaning, a considerable percentage of the data is retained, striking a balance between maintaining a robust dataset size and ensuring the quality and reliability of the data.
  - The retention rate is a testament to the effectiveness of the cleaning process, indicating that while it rigorously filters out inaccuracies and anomalies, it preserves the bulk of valuable data.

- **Strategic Importance**: 
  - This cleaning and outlier treatment process is a critical foundation for any subsequent data analysis and modeling. By ensuring the dataset's accuracy and relevance, it lays the groundwork for drawing reliable conclusions and insights.
  - These practices are not just about removing outliers or erroneous data; they are about enhancing the overall quality of the dataset, thereby enabling more precise and meaningful analyses.
  - The process also reflects the importance of data integrity in the field of data science, where the quality of the input data significantly influences the validity of the results.
  











### Conclusion
Our EDA is a comprehensive process that lays the foundation for predictive modeling. It ensures our understanding of the data is robust and our subsequent models are informed by deep insights.

## Data Preprocessing Pipeline 

### Introduction
Building on our Exploratory Data Analysis, we've developed a comprehensive Data Preprocessing Pipeline. This pipeline transforms raw taxi trip data into a structured and insightful format, focusing on data cleaning, feature extraction, and data segmentation, ensuring our data is clean, reliable, and enriched with meaningful attributes for accurate demand forecasting.

### Strategic Importance
Our pipeline streamlines and optimizes the taxi demand prediction process, unlocking deeper insights and enabling precise demand predictions. This directly contributes to enhanced fleet management and customer satisfaction.

### Technical Approach
Our approach encompasses data integration, granularity control, feature engineering, and encapsulation in a callable API. Key steps include:

1. **Data Integration**: Joining taxi trip data with hourly weather data using SQL queries in BigQuery, crucial for understanding the impact of weather on taxi demand.
2. **Granularity Control**: Segmenting data based on time and spatial dimensions.
3. **Feature Engineering**: Using advanced SQL for feature extraction, including time-based features and trigonometric transformations for capturing cyclical patterns in demand.
4. **Callable API**: Encapsulating the entire pipeline within a callable API, ensuring seamless integration and dynamic data feeding into the production model.

### Data Preprocessing Steps
The pipeline includes:

- **Data Joining and Cleaning**: SQL queries to integrate and clean the data.
- **Capping Outliers**: Applying techniques to mitigate outliers in trip duration, distance, and cost.
- **Feature Extraction and Sorting**: Extracting informative attributes and organizing data systematically.
- **Data Aggregation**: Grouping data to analyze trends at the community area level.
- **Data Segmentation for Model Training**: Creating training, validation, and test sets based on different years.
- **Data Export**: Exporting datasets to CSV files for modeling.

**Callable API**:
```sql
CALL `mlops-363723.ChicagoTaxitrips.data_preprocessing_pipeline_chicago_taxi_trips`();
```
# TFX Taxi Demand Interactive Pipeline

## Introduction
Our project leverages TensorFlow Extended (TFX) to build an interactive pipeline, expertly tailored for the end-to-end process of taxi demand prediction. This pipeline forms an integral component of our workflow, efficiently automating data ingestion, processing, model training, evaluation, and deployment.

## Pipeline Overview
The TFX pipeline is intricately structured with multiple components, each contributing significantly to different stages of the machine learning lifecycle:

- **ExampleGen**: Ingests data and splits it into distinct training and evaluation sets.
- **StatisticsGen**: Generates essential statistics for initial data analysis and further validation.
- **SchemaGen**: Infers a schema from the data statistics, offering insights into the datasets structure and format.
- **ExampleValidator**: Detects anomalies and missing values, ensuring high data quality.
- **Transform**: Conducts feature engineering, transforming raw data into a machine learning-compatible format.
- **Trainer**: Develops and trains the machine learning model using various algorithms.
- **Evaluator**: Assesses the models performance against established baselines.
- **Pusher**: Deploys the trained model to a serving infrastructure for real-world applications.

## Need for an Interactive Pipeline
Utilizing an interactive pipeline prior to large-scale deployment is crucial for:

1. **Iterative Development**: Enables rapid model iterations with immediate feedback.
2. **Experimentation**: Facilitates testing of diverse features, structures, and hyperparameters.
3. **Data Quality Assurance**: Guarantees the integrity and reliability of the data.
4. **Debugging**: Allows identification and resolution of issues in data processing and model training.

## Pipeline Execution
Our pipeline execution combines complex data processing with machine learning tasks, executed locally for enhanced control and transparency.

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
    ))

```
## Transitioning to Vertex AI
After refining our model through the interactive pipeline, we transition to Vertex AI. This platform offers automated and scalable ML workflows, enhanced performance and efficiency, and robust MLOps capabilities, making it ideal for deploying and managing our models at scale.

## Conclusion
Our TFX Taxi Demand Interactive Pipeline encapsulates the complexity of the machine learning process, offering a streamlined and scalable approach to taxi demand prediction. It automates repetitive tasks and ensures consistency and quality in our model development lifecycle, setting a strong foundation for deploying sophisticated ML models in a production environment.

# TFX Taxi Demand Production Pipeline

## Introduction
The TFX Taxi Demand Production Pipeline is our advanced solution for taxi demand prediction, leveraging the full power of TensorFlow Extended (TFX). This pipeline is engineered for high-performance, large-scale data processing and model deployment, marking the transition from development to production.

## Pipeline Enhancement for Production
Building upon the foundation established in our interactive pipeline, the production pipeline incorporates additional features and optimizations:

- **Efficient Data Handling**: Optimized for processing large datasets efficiently and reliably.
- **Advanced Model Training**: Utilizing complex training strategies to improve model accuracy and robustness.
- **Comprehensive Model Evaluation**: Implementing thorough evaluation metrics to ensure the model's readiness for real-world scenarios.
- **Automated Model Deployment**: Seamless deployment of the trained model to Vertex AI for real-time predictions.

## Scalability and Automation
Our production pipeline is designed for scalability, capable of handling vast amounts of data and complex model training scenarios. Automation plays a key role in ensuring consistent and error-free operations throughout the machine learning lifecycle.

## Monitoring and Maintenance
In a production environment, continuous monitoring and maintenance are crucial. Our pipeline is integrated with tools that facilitate ongoing supervision and timely updates to the model, ensuring it remains effective and relevant.

## Pipeline Execution on Vertex AI
The pipeline is executed on Google Cloud's Vertex AI, a platform known for its robust machine learning capabilities, which is ideal for deploying and managing models at scale.

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging

logging.getLogger().setLevel(logging.INFO)

aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                display_name=PIPELINE_NAME)
job.submit()

```
## Conclusion
Our TFX Taxi Demand Production Pipeline stands as a testament to our commitment to delivering scalable, efficient, and reliable machine learning solutions. By harnessing the capabilities of Vertex AI, we ensure that our model performs optimally in a real-world context, driving forward the innovation in taxi demand prediction.
## Deployment
*(To be added)*

## Usage
*(To be added)*

## Contributing
*(To be added)*

## License
*(To be added)*

## Contact
*(To be added)*

