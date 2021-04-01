# Clustering-Project-Zillow

### Description 
- Project examining drivers of Zestimate logerror using the Zillow database from Codeup
- Data is subset from "single unit" properties sold in 2017 from LA, Ventura, and Orange counties in Southern California

### Goals
- The goals are to find drivers of logerror, cluster the data into relevant groups, and to predict logerror using a machine learning models for each cluster

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
**1. Is the mean significantly different between?**
- null_hypothesis = 
- alternative_hypothesis = 

**2. Is there a correlation between?**
- null_hypothesis = 
- alternative_hypothesis =

**3. Is there a correlation between tax value and square footage?**
- null_hypothesis = 
- alternative_hypothesis = 

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
-

--------------------------------------------------
### How to re-create
- All necessary files minus env.py are in this repository so the best method would be to git clone and add your own env.py if you want to create your own zillow.csv
- Run .ipynb
- Adjust exploration and modeling to your liking


