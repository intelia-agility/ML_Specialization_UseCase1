
# Taxi Demand Prediction and Management

## Table of Contents
1. [Code](#code)
    1.1. [Code Repository](#code-repository)
    1.2. [Code Origin Certification](#code-origin-certification)
2. [Data](#data)
    2.1. [Dataset in Google Cloud](#dataset-in-google-cloud)
3. [Solution](#solution)
    3.1. [Business Goal and Machine Learning Solution](#business-goal-and-machine-learning-solution)
    3.2. [Data Exploration](#data-exploration)
    3.3. [Feature Engineering](#feature-engineering)
    3.4. [Preprocessing and the Data Pipeline](#preprocessing-and-the-data-pipeline)
    3.5. [Machine Learning Model Design(s) and Selection](#machine-learning-model-designs-and-selection)
    3.6. [Machine Learning Model Training and Development](#machine-learning-model-training-and-development)
    3.7. [Machine Learning Model Evaluation](#machine-learning-model-evaluation)
4. [Deployment](#deployment)
    4.1. [Model/Application on Google Cloud](#modelapplication-on-google-cloud)
    4.2. [Callable Library/Application](#callable-libraryapplication)
    4.3. [Editable Model/Application](#editable-modelapplication)
5. [Conclusion](#conclusion)
6. [Resources](#resources)
7. [Evaluation Criteria](#evaluation-criteria)

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
# TFX Taxi Demand Interactive Pipeline

## Introduction
Our project leverages TensorFlow Extended (TFX) to build an interactive pipeline, expertly tailored for the end-to-end process of taxi demand prediction. This pipeline forms an integral component of our workflow, efficiently automating data ingestion, processing, model training, evaluation, and deployment.

## Pipeline Overview
The TFX pipeline is intricately structured with multiple components, each contributing significantly to different stages of the machine learning lifecycle:

- **ExampleGen**: Ingests data and splits it into distinct training and evaluation sets.
- **StatisticsGen**: Generates essential statistics for initial data analysis and further validation.
- **SchemaGen**: Infers a schema from the data statistics, offering insights into the dataset’s structure and format.
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

