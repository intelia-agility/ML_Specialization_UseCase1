
# Taxi Demand Prediction and Management

## Table of Contents
1. [Introduction](#introduction)
2. [Business Goal and Machine Learning Solution](#business-goal-and-machine-learning-solution)
3. [Data Enrichment Process](#data-enrichment-process)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#Data Preprocessing Pipeline )
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

