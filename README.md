
# Taxi Demand Prediction and Management

## Table of Contents
1. [Introduction](#introduction)
2. [Business Goal and Machine Learning Solution](#business-goal-and-machine-learning-solution)
3. [Data Enrichment Process](#data-enrichment-process)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development and Evaluation](#model-development-and-evaluation)
7. [Deployment](#deployment)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

## Introduction
In bustling urban environments, the efficiency of taxi services is crucial for both the operators and the city's mobility. This project focuses on leveraging Machine Learning (ML) to predict taxi demand based on time, date, and location factors. By anticipating high-demand areas and times, taxi companies can optimize their fleet management, reduce customer wait times, and improve service availability.

## Business Goal and Machine Learning Solution
The primary business goal is to "Increase the efficiency of taxi distribution throughout the city to reduce customer wait time by predicting demand." This involves analyzing historical taxi usage patterns to predict future demand and enable a more dynamic allocation of taxi resources.

### Machine Learning Use Case
The use case for ML in this context is to develop a predictive model that can accurately forecast taxi demand. By using historical data and identifying patterns associated with different times, dates, and locations, the model can make informed predictions about future needs.

### Solution's Impact on Business Goal
The ML solution incorporates an end-to-end TensorFlow pipeline that processes the Chicago taxi trips dataset. The workflow includes data enrichment, exploratory data analysis, data preprocessing, and model training and evaluation. This comprehensive approach ensures accurate demand forecasting, facilitating efficient taxi distribution and reducing customer wait times.

## Data Enrichment Process

### Overview
In the Taxi Demand Prediction and Management project, we enrich the Chicago Taxi Trip dataset with additional data to enhance our analysis. A key component of this process involves integrating weather data to explore its impact on taxi trip patterns.

### Integrating Weather Data
Using the Open-Meteo.com Weather API, we fetch data such as temperature, humidity, and precipitation. This information is crucial in understanding how weather conditions influence taxi demand in Chicago. The data is fetched in both hourly and daily granularity, enabling a comprehensive analysis of the correlation between weather variations and taxi trip frequency.
#### Code Snippet:
```python

# Fetch the data from the API
response = requests.get('[API URL]')
data = response.json()

# Create DataFrames from the 'hourly' and 'daily' data
df_hourly = pd.DataFrame(data['hourly'])
df_daily = pd.DataFrame(data['daily'])

```
After fetching and structuring the weather data, our next steps involve analyzing this data in conjunction with taxi trip records to identify patterns and insights

### Data Granularity and Analysis
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

### Citation
Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649

## Exploratory Data Analysis (EDA)

### Overview
The Exploratory Data Analysis section provides insights into the dataset through various angles and techniques. Each subsection below corresponds to a key aspect of the EDA, complemented by visualizations that highlight our findings.

### Univariate Analysis
Here we look at each variable individually to understand its distribution, presence of outliers, and other statistical properties.
![Univariate Analysis]("/assets/Univariate_Analysis_of_Continuous_and_Categorical_Features.png")



### Conclusion
Our EDA is a comprehensive process that lays the foundation for predictive modeling. It ensures our understanding of the data is robust and our subsequent models are informed by deep insights.

## Data Preprocessing Pipeline for Taxi Demand Prediction

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

```sql
CREATE OR REPLACE PROCEDURE `mlops-363723.ChicagoTaxitrips.data_preprocessing_pipeline_chicago_taxi_trips`()
BEGIN
  
  -- Joining the table with weather_hourly data
  CREATE OR REPLACE TABLE mlops-363723.ChicagoTaxitrips.joined_table AS
  SELECT 
      t.*,
      w.*
  FROM 
      mlops-363723.ChicagoTaxitrips.taxi_trips_demand_1 AS t
  JOIN 
      mlops-363723.ChicagoTaxitrips.weather_hourly AS w
  ON 
      EXTRACT(DATE FROM t.trip_start_timestamp) = EXTRACT(DATE FROM w.time) AND 
      EXTRACT(HOUR FROM t.trip_start_timestamp) = EXTRACT(HOUR FROM w.time)
  WHERE 
      EXTRACT(YEAR FROM w.time) BETWEEN 2020 AND 2023;
    CREATE OR REPLACE TABLE mlops-363723.ChicagoTaxitrips.cleaned_data AS
    WITH CappedData AS (
      SELECT *,
         CASE 
            WHEN trip_seconds/60 < PERCENTILE_CONT(trip_seconds/60, 0.01) OVER() THEN 
              PERCENTILE_CONT(trip_seconds/60, 0.01) OVER()
            WHEN trip_seconds/60 > PERCENTILE_CONT(trip_seconds/60, 0.99) OVER() THEN 
              PERCENTILE_CONT(trip_seconds/60, 0.99) OVER()
            ELSE trip_seconds/60
          END AS duration,
        CASE 
            WHEN trip_miles < PERCENTILE_CONT(trip_miles, 0.01) OVER() THEN       PERCENTILE_CONT(trip_miles, 0.01) OVER()
            WHEN trip_miles > PERCENTILE_CONT(trip_miles, 0.99) OVER() THEN PERCENTILE_CONT(trip_miles, 0.99) OVER()
            ELSE trip_miles
        END AS capped_trip_miles,
        CASE 
            WHEN trip_total < PERCENTILE_CONT(trip_total, 0.01) OVER() THEN PERCENTILE_CONT(trip_total, 0.01) OVER()
            WHEN trip_total > PERCENTILE_CONT(trip_total, 0.99) OVER() THEN PERCENTILE_CONT(trip_total, 0.99) OVER()
            ELSE trip_total
        END AS capped_trip_total
      FROM mlops-363723.ChicagoTaxitrips.joined_table
      WHERE 
        pickup_longitude BETWEEN -87.9401 AND -87.5241
        AND pickup_latitude BETWEEN 41.6445 AND 42.0231
        AND trip_seconds > 0
        AND trip_miles > 0
        AND trip_total > 0
        AND pickup_community_area>0
    )
    SELECT * FROM CappedData;
    -- -- 2. Feature Extraction
    CREATE OR REPLACE TABLE mlops-363723.ChicagoTaxitrips.feature_extracted_data AS
    SELECT 
        *,
        CASE WHEN public_holiday = TRUE THEN 1 ELSE 0 END AS encoded_public_holiday,
        EXTRACT(YEAR FROM trip_start_timestamp) AS year,
        EXTRACT(MONTH FROM trip_start_timestamp) AS month,
        EXTRACT(DAY FROM trip_start_timestamp) AS day,
        EXTRACT(HOUR FROM trip_start_timestamp) AS hour,
        EXTRACT(DAYOFWEEK FROM trip_start_timestamp) - 1 AS weekday,  -- Subtracting 1 to get Monday as 0 and Sunday as 6
        DATE(trip_start_timestamp) AS trip_date,
        SIN(2 * 3.14159265359 * EXTRACT(HOUR FROM trip_start_timestamp) / 23.0) AS hour_sin,
        COS(2 * 3.14159265359 * EXTRACT(HOUR FROM trip_start_timestamp) / 23.0) AS hour_cos,
        SIN(2 * 3.14159265359 * (EXTRACT(DAYOFWEEK FROM trip_start_timestamp) - 1) / 6.0) AS day_sin, 
        COS(2 * 3.14159265359 * (EXTRACT(DAYOFWEEK FROM trip_start_timestamp) - 1) / 6.0) AS day_cos,
        SIN(2 * 3.14159265359 * EXTRACT(MONTH FROM trip_start_timestamp) / 12.0) AS month_sin,
        COS(2 * 3.14159265359 * EXTRACT(MONTH FROM trip_start_timestamp) / 12.0) AS month_cos
    FROM 
        mlops-363723.ChicagoTaxitrips.cleaned_data;
    CREATE OR REPLACE TABLE mlops-363723.ChicagoTaxitrips.sorted_feature_extracted_data AS
  SELECT 
    * 
  FROM 
    mlops-363723.ChicagoTaxitrips.feature_extracted_data
  ORDER BY 
    pickup_community_area, 
    year, 
    month, 
    day, 
    hour;

    -- 3. Data Aggregation
    CREATE OR REPLACE TABLE mlops-363723.ChicagoTaxitrips.aggregated_data AS
    SELECT 
        pickup_community_area,
        year,
        month,
        hour,
        day,
        -- encoded_public_holiday AS public_holiday,
        COUNT(unique_key) AS demand,
        AVG(duration) AS duration,
        AVG(capped_trip_miles) AS trip_miles,
        AVG(capped_trip_total) AS trip_total,
        AVG(temperature_2m) AS temperature_2m,
        AVG(relativehumidity_2m) AS relativehumidity_2m,
        AVG(precipitation) AS precipitation,
        AVG(rain) AS rain,
        AVG(snowfall) AS snowfall,
        AVG(weathercode) AS weathercode,
        MAX(encoded_public_holiday) AS public_holiday,
        MAX(hour_sin) AS hour_sin,
        MAX(hour_cos) AS hour_cos,
        MAX(day_sin) AS day_sin,
        MAX(day_cos) AS day_cos,
        MAX(month_sin) AS month_sin,
        MAX(month_cos) AS month_cos
    FROM 
        mlops-363723.ChicagoTaxitrips.sorted_feature_extracted_data
    GROUP BY 
        pickup_community_area, year,
        month,
        hour,
        day;
        
    CREATE OR REPLACE TABLE `mlops-363723.ChicagoTaxitrips.training_data` AS
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
END;


## Model Development and Evaluation
*(To be added)*

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

