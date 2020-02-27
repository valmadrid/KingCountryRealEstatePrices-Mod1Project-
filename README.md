Status: WIP

# King Country Real Estate Prices 
## Regression (Module 1) Project

<center><img src="https://www.seattlemag.com/sites/default/files/field/image/0517_Home2ariel.jpg" height=500x width=1000x /></center>

## Objective 

The main goal of this project is to build a regression model to predict the price of the King County properties.  Final model should have at most 3 features with p-values < 0.05.

## Dataset

Dataset<sup>(1)</sup> consists of 21,597 properties across 24 cities in King County sold between 2014 and 2015:
* **id** - unique identifier for the property
* **date** - date property was sold
* **price** - property selling price
* **bedrooms** - number of bedrooms
* **bathrooms** - number of bathrooms
* **sqft_living** - floor area in square feet
* **sqft_lot** - lot area in square feet
* **floors** - number of floor levels
* **waterfront** - 1 has waterfront access, 0 none
* **view** - grade corresponding to the view the property has (Mt.Rainier, Olympic, Puget Sound, lake, river, creek, skyline), coded 0-4
* **condition** - building condition relative to age and grade, coded 1-5 <a href="https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#b">(source)</a> 
* **grade** - construction quality of improvements, coded 1-13 <a href="https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#b">(source)</a> 
* **sqft_above** - floor area excluding basement
* **sqft_basement** - floor area of the basement
* **yr_built** - year built
* **yr_renovated** - year renovated
* **zipcode** - zipcode
* **lat** - latitude coordinate
* **long** - longitude coordinate
* **sqft_living15** - square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - square footage of the land lots of the nearest 15 neighbors

## Process & Results

### EDA

Data analysis was performed to understand the relationshiop between variables.  Here are some of the interesting findings:

 - Properties built in 1905 and 1917 have the highest average condition. After the Wall Street Crash of 1929, the average condition and grade and the number of properties built dropped.
 - 78% of properties built from 2000 have above average (>7) grade in terms of contruction quality.  99% of these properties were in average (3) conditon.
- 16% of properties that were at least a century old at the time of acquisition were in above average (>3) condition and (>7) grade.
- There are 176 duplicates in the dataset, these properties properties were resold within a year.  95% of these properties were resold at profit. None of them were upgraded or renovated. A worn-out property in Seattle was resold at 321% profit.
- The most expensive houses are located in Seattle, Bellevue, Medina, Mercer Island and Kirkland.
- The features with the highest correlation with price are grade, sqft_living, sqft_living 15 and lat.

### Preprocessing

**Missing Values**
* view - Upon checking the King County website for a handful of properties, values for these features are indeed missing. Since there is only 0.2% missing values, these were defaulted to 0.
* waterfront - Similar to view, these values are missing on the website too.  11% of this column were defaulted to 0 (no waterfront access right).
* sqft_basement - This was derived by getting the different between sqft_living and sqft_above.
* year_renovated - 97% of this column are zeroes and missing values.  It was assumed that the properties were not renovated.

**Outliers**
Z-score was use to tackle outliers.  All datapoints with absolute z-score of more than 3 were removed from the dataset.

### Modeling

Three techniques were used to select 3 features:
1. Highest correlation coefficient (multicollinearity removed)
2. Recursive Feature Elimination (Scikit Learn)
3. Select From Model (Scikit Learn)

Interactions and polynomial features were also introduced.

| Features | R<sup>2</sup> | MSE |
|------------------------------------------|--------------|--------------|
|linear: grade, lat_log, bedrooms|0.6218|0.0160|
|interaction: sqft_living_log\*grade, lat_log\*yr_sold_log, floors\*condition|0.6610|0.0144|
|polynomial: grade<sup>2</sup>, lat_log<sup>2</sup>, bedrooms<sup>2</sup>|0.6204|0.0161|

## Important libraries used
* Scikit Learn
* Statsmodels
* Folium

## Files

## Conclusion

**Market Value**

The average price per square foot of the 15 closest neighbours of lot and improvement, seperately. Prices per square foot change from one area to another and it would improve our model by subsetting over zipcodes that have similar values for these average square foots.

**Lot Type**

**Restrictions and Encumbrances**

A categorical variable that will tell whether the property has a clean title. Buyer and real estate agents should be aware of the limitations and threats to property ownership. These will reduce the price and make it less marketable.

**More information on location and neighborhood**

* Unemployment rate, average/median household income, crime rate, air quality 
* Distance to famous landmarks, lakes, rivers, mountains
* Proximity to church, schools, cemetary, shopping malls etc. 

## Contributor
Grace Valmadrid

Tom Ribaroff

## Credit

Image by <a href="https://www.coldwellbanker.com">Coldwell Banker</a> 

## Footnotes
1 The dataset has been modified by our instructors for this project.

