Status: WIP

# King Country Real Estate Prices 
## Regression (Module 1) Project

<center><img src="https://www.seattlemag.com/sites/default/files/field/image/0517_Home2ariel.jpg" height=500x width=1000x /></center>

## Objective 

The main goal of this project is to build a regression model to predict the price of the King County properties.  Final model should have at most 3 features with p-values less than 0.05. 

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
* waterfront - As this has a very weak correlation with price, missing values were defaulted to 0.
* view - Since only 0.2% is missing and correlation with price is weak, these were also defaulted to 0. 
* sqft_basement - This was derived by getting the difference between sqft_living and sqft_above.
* year_renovated - 97% of this column are zeroes and missing values.  It was assumed that the properties were not renovated.

**Outliers**

All datapoints with z-score greater than 3 or less than -3 were removed from the dataset.

### Modeling

A baseline model was created using all features which yielded an R<sup>2</sup> of 0.8728.  However, 9 features have p-values of greater than 0.05.

Three techniques were used to select 3 features:
1. Highest correlation coefficient (multicollinearity removed)
2. Recursive Feature Elimination (Scikit Learn)
3. Select From Model (Scikit Learn)

Below table shows that the first technique gave the highest R<sup>2</sup>:

|Technique| ŷ(price_log) | Test R<sup>2</sup> | Test MSE |
|---------|------------------------------------------|--------------|--------------|
|1|-116.1136 + 0.1081\*grade + 72.0351\*lat_log + 0.0352\*bedrooms|0.6218|0.0160|
|2|-186.4715 + 84.4719\*lat_log - 73.7738\*long_log + 61.8658\*yr_sold_log|0.2604|0.0313|
|3|-4.1042 - 2.1174e14\*yr_built_log + 2.1174e14\*yr_renovated__log + 0.0\*renovated|-0.132|0.0479|


Interactions and polynomial features were also introduced.  Again, the first feature selection technique works best:

| ŷ(price_log) | R<sup>2</sup> | MSE |
|------------------------------------------|--------------|--------------|
|interaction: -114.9193 + 0.028 sqft_living_log\*grade + 21.6215 lat_log\*yr_sold_log + 0.0086 floors\*condition  |0.6610|0.0144|
|polynomial: -55.054 + 0.0068 grade<sup>2</sup> + 21.4129 lat_log<sup>2</sup> + 0.0052 bedrooms<sup>2</sup>|0.6204|0.0161|

## Conclusion

## Important libraries used
* Scikit Learn
* Statsmodels
* Folium

## Files

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

