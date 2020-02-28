# King Country Real Estate Prices 
## Regression (Module 1) Project

<center><img src="https://www.seattlemag.com/sites/default/files/field/image/0517_Home2ariel.jpg" height=500x width=1000x /></center>
 
## Objective 

The main goal of this project is to build a regression model to predict the price of the King County properties.  Final model should have at most 3 features with p-values less than 0.05. 

Topics covered: Pandas, Data Visualisation, Regression, Interactions & Polynomial Features

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
* **condition**<sup>(2)</sup> - building condition relative to age and grade, coded 1-5
* **grade**<sup>(2)</sup> - construction quality of improvements, coded 1-13
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
 - 78% of properties built from 2000 have above average (>7) grade in terms of contruction quality.  99% of these properties were in average (3) conditon only.
 
<center><img src="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/images/condition.png"/></center>
 
- 16% of properties that were at least a century old at the time of acquisition were in above average (>3) condition and (>7) grade.

<center><img src="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/images/100.png"/></center>


- 176 properties were resold within a year.  
   - 95% of which were resold at profit. None of them were upgraded or renovated.
   - A worn-out property in Seattle was resold at 321% profit.
   
<center><img src="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/images/resell.png"/></center>
 
- The most expensive houses are located in Seattle, Bellevue, Medina, Mercer Island and Kirkland.

<center><img src="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/images/expensive.png"/></center>

- There is one property with 33 bedrooms in a neighborhood where majority of the properties have 1-6 bedrooms only.

<center><img src="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/images/33.png"/></center>

- The features with the highest correlation with price are grade, sqft_living_log, sqft_living15_log and lat_log.

<center><img src="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/images/corr.png"/></center>

### Preprocessing

**Missing Values**
* waterfront - As this has a very weak correlation with price, missing values (11%) were defaulted to 0.
* view - Since only 0.2% is missing and correlation with price is weak, these were also defaulted to 0. 
* sqft_basement - This was derived by getting the difference between sqft_living and sqft_above.
* year_renovated - 97% of this column are zeroes and missing values.  It was assumed that the properties were not renovated.

**Outliers**

All datapoints with z-score greater than 3 or less than -3 were removed from the dataset.

### Modeling

A baseline model was created using all features which yielded an R<sup>2</sup> of 0.8728.  However, 9 features have p-values of greater than 0.05.

Three techniques were then used to select 3 features:
1. Highest correlation coefficient (multicollinearity removed)
2. Recursive Feature Elimination (Scikit Learn)
3. Select From Model (Scikit Learn)

Below table shows that the first technique gave the highest R<sup>2</sup>:

|Technique| ŷ(price_log) | Test R<sup>2</sup> | Test MSE |
|---------|------------------------------------------|--------------|--------------|
|1|-116.1136 + 0.1081\*grade + 72.0351\*lat_log + 0.0352\*bedrooms|0.6218|0.0160|
|2|-186.4715 + 84.4719\*lat_log - 73.7738\*long_log + 61.8658\*yr_sold_log|0.2604|0.0313|
|3|-4.1042 - 2.1174e14\*yr_built_log + 2.1174e14\*yr_renovated__log + 0.0\*renovated|-0.132|0.0479|

Interactions and polynomial features (degree=2) were also introduced.  Again, the first feature selection technique works best:

| ŷ(price_log) | Test R<sup>2</sup> | Test MSE |
|------------------------------------------|--------------|--------------|
|interaction: -114.9193 + 0.028 sqft_living_log\*grade + 21.6215 lat_log\*yr_sold_log + 0.0086 floors\*condition  |0.6610|0.0144|
|polynomial: -55.054 + 0.0068 grade<sup>2</sup> + 21.4129 lat_log<sup>2</sup> + 0.0052 bedrooms<sup>2</sup>|0.6204|0.0161|

## Conclusion
The best linear model explains 62% of the variation in price (*price_log*).  *lat_log* has the biggest effect on *price_log*.

Adding interaction features improved the score by 4%. It can still be further improved with more features.  LASSO, Ridge and Elastic Net can then be used to regularised the model.

Also, the following can useful addition to the dataset as these are taken into considerations when appraising a property:
- Market value per sqft. of lot and improvement per zipcode:  Prices change from one area to another.
- Lot type: Premiums are added depending on the location of the lot. 
- Retrictions and encumbrances on the land title: Limitations and threats to property ownership can reduce the property value.
- More information about the neighborhoud:
   - Unemployment rate, average/median household income, crime rate, air quality 
   - Proximity to church, schools, cemetary, shopping malls etc. 

## Important libraries used
* Scikit Learn
* Statsmodels
* Folium

## Files

Main notebook: <a href="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/kchousing.ipynb">kchousing.ipynb</a>

Functions: <a href="https://github.com/valmadrid/KingCountryRealEstatePrices-Mod1Project-/blob/master/functions.py">functions.py</a>

## Contributors
<a href="https://www.linkedin.com/in/valmadrid/">Grace Valmadrid</a>

Tom Ribaroff

## Credit

Image by <a href="https://www.coldwellbanker.com">Coldwell Banker</a> 

## Footnotes
1 The dataset has been modified by our instructors for this project.

2 https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#b

