# Clustering-Project-Zillow

### Description 
- Project examining drivers of Zestimate logerror using the Zillow database from Codeup
- Data is subset from "single unit" properties sold in 2017 from LA, Ventura, and Orange counties in Southern California

### Goals
- The goals are to find drivers of logerror, cluster the data into relevant groups, and to predict logerror using a machine learning model for each cluster

---------------------------------
### Data Dictionary
---
| Column | Definition | Data Type |
| ----- | ----- | ----- |
|parcelid| unique id for the lot| int|
|transactiondate| date the property sold | object|
|bathrooms | number of bathrooms in home including half baths| float|
|bedrooms| number of bedrooms in home| float|
|buildingqualitytypeid| assessment of condition of the property from best(lowest) to worst(highest)| float
|square_feet| total finished living area of home| float|
|fips| federal county code| float|
|latitude| latitude of the middle of the parcel | float|
|longitude| longitude of the middle of the parcel| float|
|lotsizesquarefeet| area of the lot in sqft| float|
|rawcensustractandblock| Census tract and block ID combined| float|
|regionidcity| code for the city in which the property is located| float|
|regionidcounty| code for the county in which the property is located|float|
|regionidzip| zip code the property is located in| float|
|roomcnt| total number of rooms | float|
|unitcnt| number of units the structure is built into|float|
|yearbuilt| year the property was built| float|
|structuretaxvaluedollarcnt| the assessed value of the structure built on the lot| float|
|tax_value|total taxed assessed value of the parcel| float |
|assessmentyear| year of property tax assessment| float|
|landtaxvaluedollarcnt| assessed value of the land on the lot| float|
|taxes| taxes paid for assement year| float|
|heatingorsystemdesc| type of heating system| float|
|county| name of the county the property is located| object|
|logerror_absolute| absolute value of logerror| float|
|age_in_years| houses age from yearbuilt to 2021| float|
|Bathrooms_cat| categorical bathrooms from 0,1,2,3,4+|object|
|Bedrooms_cat| categorical bedrooms from 0,1,2,3,4+| object|
|tax_rate| taxes paid divided by assessed tax value perctenage| float|

---------------------------------------------------
| Target | Definition | Data Type |
| ----- | ----- | ----- |
|logerror| log of the error between Zestimate and saleprice log(Zestimate) - log(Saleprice)| float|

--------------------------------------------------
### Hypotheses
**1. Is there a significant difference in the average logerror of Orange County and the total average error?**
- null_hypothesis = There is no difference in the average logerror of Orange county and the overall average logerror
- alternative_hypothesis = There is a difference in the logerror of Orange county when compared to the overall average logerror

**2. Is there a difference in the average log error between houses that are greater than 2500 feet and those that are less?**
- null_hypothesis = There is no difference in the average logerror of houses greater than 2500 square feet and those that have 2500 square feet or less
- alternative_hypothesis = There is a difference in the average logerror of houses greater than 2500 square feet and those that have 2500 square feet or less

**3. Is there a difference in the average log error between houses that have more than 4 bathrooms and those that have 4 or less?**
- null_hypothesis = There is no difference in the average logerror between houses that have greater than 4 bathrooms and those with 4 or less
- alternative_hypothesis = There is a difference in the average logerror of houses that have greater than 4 bathrooms and those with 4 or less

--------------------------------------------------

### Project Plan
1. Acquire data from codeup zillow database using correct joins
2. Prepare data by removing unnecessary/redundant columns, dealing with nulls, and encoding variables for use with ML models
3. Explore data using functions in explore.py as well as jointplot, pairplot, and heatmap.
4. Cluster the data into relevant groups
5. Create ML models based on clusters and choose best performing for test data
6. Present findings on logerror drivers, clusters, and model performance

---------------------------------------------------
### Project Takeaways
- Clusters created were minimally effective at assisting in regression predictions of logerror
- Even clusters created out of features most associated with the target variable were only slightly effective
- Logerror was not easy to predict using a linear model, would like to see if there was another model type that may be more effective at predicting logerror
- Would also like to use the zestimate and actual selling price as target variables to see if a better prediction model could be made for error in estimates

--------------------------------------------------
### How to re-create
- All necessary files minus env.py are in this repository so the best method would be to git clone and add your own env.py if you want to create your own zillow.csv
- Run Cluster_project_explore and Final_clustering_project notebooks
- Adjust exploration and modeling to your liking


