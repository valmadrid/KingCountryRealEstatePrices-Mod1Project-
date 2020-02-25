import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set_context("paper", font_scale=1.8)
sns.set_palette("GnBu_d")

import folium
from folium.plugins import MarkerCluster, HeatMap


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup
import re

from IPython.display import Image

import folium
from folium.plugins import MarkerCluster, HeatMap



def get_interaction(df, degree=3):
    """
    Takes in a df and returns a new df with interaction features
    """
    
    inter = PolynomialFeatures(degree=degree,
                              interaction_only=True,
                              include_bias=False)
    X_interaction = inter.fit_transform(df)
    new_df = pd.DataFrame(
        X_interaction,
        columns=[
            feat.replace(" ", "_x_")
            for feat in inter.get_feature_names(df.columns)
        ],
        index=df.index)

    return new_df


def get_poly(df, degree=3):
    """
    Takes in a df and returns a new df with polynomial features
    """
    
    poly = PolynomialFeatures(degree=degree,
                              interaction_only=False,
                              include_bias=False)
    X_poly = poly.fit_transform(df)
    new_df = pd.DataFrame(
        X_poly,
        columns=[
            feat.replace(" ", "_x_").replace("^","_")
            for feat in poly.get_feature_names(df.columns)
        ],
        index=df.index)

    return new_df


def preprocess(X, y):
    """
    Takes in: X, y (dataframes)
    Returns: train and test sets, X scaled
    
    """
    
    # split X and y between train and test
    X_train, X_test, y_train, y_test = split(X,y)
    
    # scale X
    X_train_scaled, X_test_scaled = scale(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def scale(X_train, X_test):
    """
    Takes in: X_train, X_test (dataframes)
    Returns: scaled X_train, X_test using MinMaxScaler
    
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled,
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled,
                                 columns=X_test.columns,
                                 index=X_test.index)
    
    return X_train_scaled, X_test_scaled


def split(X,y):
    """
    Takes in: X, y (dataframes)
    Returns: X_train, X_test, y_train, y_test (test set = 20%)
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    return X_train, X_test, y_train, y_test


def get_corr(features_list, df):
    """
    Takes in: features_list (list), df (dataframe)
    Returns: list of features ranked based on correlation coefficient (features causing multicollinearity excluded)
    
    """
    
    top_features = []
    top_features.append(features_list[0])

    for i in range(1, len(features_list)):
        for col in top_features:
            if col == features_list[i]:
                continue
            if (abs(df[col].corr(df[features_list[i]])) < 0.5) & (features_list[i] not in top_features):
                top_features.append(features_list[i])
            elif (abs(df[col].corr(df[features_list[i]])) >= 0.5) & (features_list[i] in top_features):
                top_features.remove(features_list[i])
                break
            else:
                break

    return top_features


def run_linreg(X, y, df, show_formula = False, plot_regressor=False):
    """
    Takes in: X (subset of df), y(subset of df), df (dataframe)
    Returns: linear regression object
    
    """
    
    X_train, X_test, y_train, y_test = split(X,y)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    
    # prints y_hat formula, R^2, MSE, p-values, qqplot and plot of each feature in X
    evaluate_model(linreg, X_train, X_test, y_train, y_test, df, show_formula = show_formula, plot_regressor=plot_regressor)
    
    return linreg


def evaluate_model(reg_model, X_train, X_test, y_train, y_test, df, show_formula = False, plot_regressor=False):
    """
    Takes in: reg_model (object), X_train, X_test, y_train, y_test (all subset of df), df (dataframe)
    Prints: y_hat formula, R^2, MSE, p-values, qqplot and plot of each feature in X
    
    """
    
    # prints y_hat formula based on reg_model
    beta_x = ["{}*{}".format(round(beta,4), x) for beta, x in dict(zip(reg_model.coef_ , X_train.columns)).items()]
    yhat = "{} + {}".format(round(reg_model.intercept_,4), " + ".join(beta_x))
    if show_formula:
        print("y\u0302({}) = ".format(y_train.name), yhat, "\n")
    
    # prints R^2 and MSE
    y_train_hat = reg_model.predict(X_train)
    y_test_hat = reg_model.predict(X_test)
    
    print("Train R\u00b2 = ", round(reg_model.score(X_train, y_train),4))
    print("Train MSE = ", round(mean_squared_error(y_train, y_train_hat),4))
    print("Test  R\u00b2 = ", round(reg_model.score(X_test, y_test),4))
    print("Test MSE = ", round(mean_squared_error(y_test, y_test_hat),4), "\n")
    
    # prints features with p-values > 0.05, using statsmodels
    model = ols(formula = y_train.name+"~"+"+".join(X_train.columns), data = df).fit()

    print("Features with p-values > 0.05: ")
    display(pd.DataFrame(data=model.pvalues.reset_index().values,  columns=["features", "p_values"]).query("p_values>0.05"))
    
    # prints regression plot of each feature in X using statsmodels 
    if plot_regressor:
        for x in X_train.columns:
            fig = plt.figure(figsize=(12,10))
            sm.graphics.plot_regress_exog(model, x, fig=fig)
            fig.show()
            
    # prints qqplot using statsmodels     
    sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)
    plt.title("QQ-Plot")
    plt.show()
    

def run_rfe(model, X, y, n_features_to_select=3):
    """
    Takes in: model (object), X , y (dataframes), n_features_to_select (number of features to select)
    Returns: dataframe of selected features using RFE
    
    """
    
    X_train, X_test, y_train, y_test = split(X, y)
    selector = RFE(model, n_features_to_select)
    selector.fit(X_train, y_train)
    top_features = pd.DataFrame({"features": X.columns, "ranking": selector.ranking_}).sort_values("ranking")
    
    return top_features.query("ranking == 1")


def run_sfm(model, X, y, max_features=3):
    """
    Takes in: model (object), X , y (dataframes), max_features (number of features to select)
    Returns: dataframe of selected features using SelectFromModel
    
    """
    
    X_train, X_test, y_train, y_test = split(X, y)
    selector = SelectFromModel(model, max_features=3)
    selector.fit(X_train, y_train)
    top_features = pd.DataFrame({"features": X.columns, "include": selector.get_support()}).sort_values("include", ascending=False)
    
    return top_features.query("include == True")


def get_zipcodes(zipcode_list, df):
    """
    Takes in: zipcode_list (list of zipcodes), df (dataframe)
    Returns: zipcodes (dataframe with descriptive statistics per zipcode of selected features from df)
    
    """
    
    zipcodes = pd.DataFrame()
    for i in range(len(zipcode_list)):
        z = zipcode_list[i]
        zipcodes.at[i, "zipcode"] = int(z)
        
        # get city and coordinates from zipdatamaps.com
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
        
        # descriptive statistics per zipcode
        zipcodes.at[i, "property_count"] = df.query("zipcode == @z").price.count()
        for col in ["price", "grade", "condition", "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors"]:
            zipcodes.at[i, col+"_max"] = df.query("zipcode == @z")[col].max()
            zipcodes.at[i, col+"_min"] = df.query("zipcode == @z")[col].min()
            zipcodes.at[i, col+"_mean"] = df.query("zipcode == @z")[col].mean()
            zipcodes.at[i, col+"_median"] = df.query("zipcode == @z")[col].median()
            zipcodes.at[i, col+"_mode"] = df.query("zipcode == @z")[col].mode()[0]
    
    # fix dtypes
    zipcodes.zipcode = zipcodes.zipcode.astype("object")
    zipcodes.property_count = zipcodes.property_count.astype("int64")
    for col in ["grade", "condition"]:
        zipcodes[col+"_max"] = zipcodes[col+"_max"].astype("int64")
        zipcodes[col+"_min"] = zipcodes[col+"_min"].astype("int64")
        zipcodes[col+"_mean"] = zipcodes[col+"_mean"].astype("int64")
        zipcodes[col+"_median"] = zipcodes[col+"_median"].astype("int64")
        zipcodes[col+"_mode"] = zipcodes[col+"_mode"].astype("int64")
    

    return zipcodes
    

def add_CircleMarker(map_, df, color="green"):
    """
    Takes in: map_ (folium map object), df (dataframe with lat and long columns), color
    Returns: updated map_ with markers based on the coordinate list
    
    """
    
    # create a coordinate list using df
    coord_list = df.coordinates.values.tolist()
    
    # add markets to map_
    for point in range(len(coord_list)):
        folium.CircleMarker(
            coord_list[point],
            radius=df.price_mean.iloc[point] / 80000,
            color='b',
            fill=True,
            fill_opacity=0.7,
            fill_color=color,
            popup={
                "zipcode": int(df.zipcode.iloc[point]),
                "zipcode_name": df.zipcode_name.iloc[point],
                "# of properties": int(df.property_count.iloc[point]),
                "max price": round(df.price_max.iloc[point], 0),
                "min price": round(df.price_min.iloc[point], 0),
                "mean price": round(df.price_mean.iloc[point], 0),
                "mode price": round(df.price_mode.iloc[point], 0),
                "median price": round(df.price_median.iloc[point], 0)
            }).add_to(map_)
    
    return map_
    
    
    
def add_CircleMarker2(map_, df, color="red"):
    """
    Takes in: map_ (folium map object), df (dataframe with lat and long columns), color
    Returns: updated map_ with markers based on the coordinate list
    
    """
    # create a coordinate list using df
    coord_list = df[["lat", "long"]].values.tolist()
    
    # add markets to map_
    for point in range(len(coord_list)):
        folium.CircleMarker(
            coord_list[point],
            radius=df.price.iloc[point]/500000,
            color='b',
            fill=True,
            fill_opacity=0.7,
            fill_color=color,
            popup={
                "id": int(df.id.iloc[point]),
                "waterfront": df.waterfront_.iloc[point],
                "view": df.view_.iloc[point],
                "zipcode": int(df.zipcode.iloc[point]),
                "city": df.zipcode_name.iloc[point],
                "price": round(df.price.iloc[point], 0)
            }).add_to(map_)
    
    return map_


def add_CircleMarker3(map_, df, size, color="red"):
    """
    Takes in: map_ (folium map object), df (dataframe with lat and long columns), size (df column to be used for the size of the circle maker), color
    Returns: updated map_ with markers based on the coordinate list
    
    """
    
    # create a coordinate list using df
    coord_list = df[["lat", "long"]].values.tolist()
    
    # add markets to map_
    for point in range(len(coord_list)):
        folium.CircleMarker(
            coord_list[point],
            radius=df[size].iloc[point],
            color='b',
            fill=True,
            fill_opacity=0.7,
            fill_color=color,
            popup={
                "id": int(df.id.iloc[point]),
                "zipcode": int(df.zipcode.iloc[point]),
                "city": df.zipcode_name.iloc[point],
                "lot area": df.sqft_lot.iloc[point],
                "floor area": df.sqft_living.iloc[point],
                "price": round(df.price.iloc[point], 0)
            }).add_to(map_)
    
    return map_


def add_CircleMarker4(map_, df, color="red"):
    """
    Takes in: map_ (folium map object), df (dataframe with lat and long columns), color
    Returns: updated map_ with markers based on the coordinate list
    
    """
    
    # create a coordinate list using df
    coord_list = df[["lat", "long"]].values.tolist()
    
     # add markets to map_
    for point in range(len(coord_list)):
        folium.CircleMarker(
            coord_list[point],
            radius=2.5,
            color='b',
            fill=True,
            fill_opacity=0.7,
            fill_color=color,
            popup={
                "id": int(df.id.iloc[point]),
                "zipcode": int(df.zipcode.iloc[point]),
                "city": df.zipcode_name.iloc[point],
                "lot area": df.sqft_lot.iloc[point],
                "floor area": df.sqft_living.iloc[point],
                "price": round(df.price.iloc[point], 0)
            }).add_to(map_)
    
    return map_


