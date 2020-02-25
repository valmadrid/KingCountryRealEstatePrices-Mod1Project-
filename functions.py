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
    

def add_CircleMarker(map_, df, color="green"):
    
    coord_list = df.coordinates.values.tolist()
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
    
    coord_list = df[["lat", "long"]].values.tolist()
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
    i = 10**(len(str(df[size].max()))-2)
    coord_list = df[["lat", "long"]].values.tolist()
    for point in range(len(coord_list)):
        folium.CircleMarker(
            coord_list[point],
            radius=df[size].iloc[point]/i,
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
    coord_list = df[["lat", "long"]].values.tolist()
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


def get_interaction(df, degree=3):
    """
    Takes in a df and returns a new df with interaction columns
    """
    
    poly = PolynomialFeatures(degree=degree,
                              interaction_only=False,
                              include_bias=False)
    X_interaction = poly.fit_transform(df)
    new_df = pd.DataFrame(
        X_interaction,
        columns=[
            feat.replace(" ", "_")
            for feat in poly.get_feature_names(df.columns)
        ],
        index=df.index)

    return new_df


def preprocess(X, y):
    """
    
    """
    
    X_train, X_test, y_train, y_test = split(X,y)
    X_train_scaled, X_test_scaled = scale(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def scale(X_train, X_test):
    """
    
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
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    return X_train, X_test, y_train, y_test


def get_corr(features_list, df):
    
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
    
    X_train, X_test, y_train, y_test = split(X,y)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    evaluate_model(linreg, X_train, X_test, y_train, y_test, df, show_formula = show_formula, plot_regressor=plot_regressor)
    
    return linreg


def run_lasso(X, y, df, alpha, show_formula = False, plot_regressor=False):
    
    X_train, X_test, y_train, y_test = split(X,y)
    
    if isinstance(alpha, np.ndarray):
        df_lasso = pd.DataFrame()
        for i in range(len(alpha)):
            lasso = Lasso(alpha = alpha[i])
            lasso.fit(X_train, y_train)
            
            y_train_hat = lasso.predict(X_train)
            y_test_hat = lasso.predict(X_test)
            
            df_lasso.at[i, "alpha"] = alpha[i]
            df_lasso.at[i, "train_r2"] = lasso.score(X_train, y_train)
            df_lasso.at[i, "train_mse"] = mean_squared_error(y_train, y_train_hat)
            df_lasso.at[i, "test_r2"] = lasso.score(X_test, y_test)
            df_lasso.at[i, "test_mse"] = mean_squared_error(y_test, y_test_hat)
            df_lasso = df_lasso.sort_values("test_r2", ascending=False)
            
        best_alpha = df_lasso.alpha.iloc[0]
        lasso = Lasso(alpha = best_alpha)
        lasso.fit(X_train, y_train)
        display(df_lasso.head(10))
        
    else:
        lasso = Lasso(alpha = alpha)
        lasso.fit(X_train, y_train)
        evaluate_model(lasso, X_train, X_test, y_train, y_test, df, show_formula = show_formula, plot_regressor=plot_regressor)
    
    
    return lasso



def run_ridge(X, y, df, alpha, show_formula = False, plot_regressor=False):
    
    X_train, X_test, y_train, y_test = split(X,y)
    
    if isinstance(alpha, np.ndarray):
        df_ridge = pd.DataFrame()
        for i in range(len(alpha)):
            ridge = Ridge(alpha = alpha[i])
            ridge.fit(X_train, y_train)
            
            y_train_hat = ridge.predict(X_train)
            y_test_hat = ridge.predict(X_test)
            
            df_ridge.at[i, "alpha"] = alpha[i]
            df_ridge.at[i, "train_r2"] = ridge.score(X_train, y_train)
            df_ridge.at[i, "train_mse"] = mean_squared_error(y_train, y_train_hat)
            df_ridge.at[i, "test_r2"] = ridge.score(X_test, y_test)
            df_ridge.at[i, "test_mse"] = mean_squared_error(y_test, y_test_hat)
            df_ridge = df_ridge.sort_values("test_r2", ascending=False)
            
        best_alpha = df_ridge.alpha.iloc[0]
        ridge = Ridge(alpha = best_alpha)
        ridge.fit(X_train, y_train)
        display(df_ridge.head(10))
    
    else:
    
        ridge = Ridge(alpha = alpha)
        ridge.fit(X_train, y_train)
        evaluate_model(ridge, X_train, X_test, y_train, y_test, df, show_formula = show_formula, plot_regressor=plot_regressor)
    
    return ridge

def evaluate_model(reg_model, X_train, X_test, y_train, y_test, df, show_formula = False, plot_regressor=False):
    
    beta_x = ["{}*{}".format(round(beta,4), x) for beta, x in dict(zip(reg_model.coef_ , X_train.columns)).items()]
    yhat = "{} + {}".format(round(reg_model.intercept_,4), " + ".join(beta_x))
    if show_formula:
        print("y\u0302({}) = ".format(y_train.name), yhat, "\n")
    

    y_train_hat = reg_model.predict(X_train)
    y_test_hat = reg_model.predict(X_test)
    
    print("Train R\u00b2 = ", round(reg_model.score(X_test, y_test),4))
    print("Train MSE = ", round(mean_squared_error(y_train, y_train_hat),4))
    print("Test  R\u00b2 = ", round(reg_model.score(X_test, y_test),4))
    print("Test MSE = ", round(mean_squared_error(y_test, y_test_hat),4), "\n")
    
    model = ols(formula = y_train.name+"~"+"+".join(X_train.columns), data = df).fit()
    
    print("Features with p-values > 0.05: ")
    display(pd.DataFrame(data=model.pvalues.reset_index().values,  columns=["features", "p_values"]).query("p_values>0.05"))
    
    if plot_regressor:
        for x in X_train.columns:
            fig = plt.figure(figsize=(15,10))
            sm.graphics.plot_regress_exog(model, x, fig=fig)
            fig.show()
            
    sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)
    plt.title("QQ-Plot")
    plt.show()
    

def run_rfe(model, X, y, n_features_to_select=3):
    
    X_train, X_test, y_train, y_test = split(X, y)
    selector = RFE(model, n_features_to_select)
    selector.fit(X_train, y_train)
    top_features = pd.DataFrame({"features": X.columns, "ranking": selector.ranking_}).sort_values("ranking")
    
    return top_features.query("ranking == 1")


def run_sfm(model, X, y, max_features=3):
    
    X_train, X_test, y_train, y_test = split(X, y)
    selector = SelectFromModel(model, max_features=3)
    selector.fit(X_train, y_train)
    top_features = pd.DataFrame({"features": X.columns, "include": selector.get_support()}).sort_values("include", ascending=False)
    
    return top_features.query("include == True")