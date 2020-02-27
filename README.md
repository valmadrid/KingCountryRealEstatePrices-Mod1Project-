Status: WIP

# King Country Real Estate Prices 
## Regression (Module 1) Project

<center><img src="https://www.seattlemag.com/sites/default/files/field/image/0517_Home2ariel.jpg" height=500x width=1000x /></center>

## Summary

A regression model was developed to predict the price of the King County properties sold between 2014 and 2015.  Through data exploration we have discovered the following:

 - Properties built in 1904, 1905, 1909, 1911, 1916, 1917 and 1931 have the highest average condition. After the Wall Street Crash of 1929, the number of properties built and average condition both dropped.
 
 - 78% of properties built from 2000 have above <a href="https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#b">average</a> (>7) grade in terms of contruction quality.  99% of these properties were in <a href="https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#b">average</a> (3) conditon.

- 16% of properties that were at least a century old at the time of acquisition were in above average (>3) condition and (>7) grade.

- 

- 176 properties appeared more than once in the dataset, 95% of which were resold at profit. None of the resold properties were upgraded or renovated. A worn-out property in Seattle was resold at 321% profit.



* Where are the top 10 most expensive properties located?

* Is it worth upgrading the improvement to resell at a higher price?

The following features were also introduced:
* 

## Dataset

Dataset<sup>1</sup> consists of 21,597 properties across 24 cities in King County:
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

