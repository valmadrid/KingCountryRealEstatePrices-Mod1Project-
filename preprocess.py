import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set_context("paper", font_scale=1.8)
sns.set_palette("GnBu_d")

import folium
from folium.plugins import MarkerCluster, HeatMap

from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup
import re

from IPython.display import Image



def get_zipcodes(zipcode_list, df):
    
    zipcodes = pd.DataFrame()
    for i in range(len(zipcode_list)):
        z = zipcode_list[i]
        zipcodes.at[i, "zipcode"] = int(z)
        r = requests.get("https://www.zipdatamaps.com/"+str(z))
        soup = BeautifulSoup(r.content, "html.parser")
        table = (soup.find("table")).find_all("td")
        table = [t.text for t in table]
        for j in range(len(table)):
            if j == table.index("Official Zip Code Name"):
                zipcodes.at[i, "zipcode_name"] = table[j+1].strip()
            elif j == table.index("Coordinates(Y,X)"):
                zipcodes.at[i, "coordinates"] = table[j+1]
        zipcodes.at[i, "coordinates"] = [float(c) for c in zipcodes.coordinates.iloc[i].split(", ")]
        zipcodes.at[i, "property_count"] = df.query("zipcode == @z").price.count()
        for col in ["price", "grade", "condition", "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors"]:
            zipcodes.at[i, col+"_max"] = df.query("zipcode == @z")[col].max()
            zipcodes.at[i, col+"_min"] = df.query("zipcode == @z")[col].min()
            zipcodes.at[i, col+"_mean"] = df.query("zipcode == @z")[col].mean()
            zipcodes.at[i, col+"_median"] = df.query("zipcode == @z")[col].median()
            zipcodes.at[i, col+"_mode"] = df.query("zipcode == @z")[col].mode()[0]
            
    zipcodes.zipcode = zipcodes.zipcode.astype("object")
    zipcodes.property_count = zipcodes.property_count.astype("int64")
    for col in ["grade", "condition"]:
        zipcodes[col+"_max"] = zipcodes[col+"_max"].astype("int64")
        zipcodes[col+"_min"] = zipcodes[col+"_min"].astype("int64")
        zipcodes[col+"_mean"] = zipcodes[col+"_mean"].astype("int64")
        zipcodes[col+"_median"] = zipcodes[col+"_median"].astype("int64")
        zipcodes[col+"_mode"] = zipcodes[col+"_mode"].astype("int64")
    

    return zipcodes


def get_image(image_filename):
    """
    Takes in 
    """
    
    display(Image(image_filename))