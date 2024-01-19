import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit


def read(file):
    """
    This method returns two dataframes:
    df_year (with years as columns) and df_country (with countries as columns)
    """
    data = pd.read_excel(file, header=None)
    data = data.iloc[3:]
    print(data)
    var = data.rename(columns=data.iloc[0]).drop(data.index[0])
    list_col = ['Country Code', 'Indicator Name', 'Indicator Code']
    var = var.drop(list_col, axis=1)
    df_year = var.set_index("Country Name")

    df_country = df_year.transpose()
    df_year.index.name = None
    df_country.index.name = None
    df_year = df_year.fillna(0)
    df_country = df_country.fillna(0)
    return df_year, df_country


def filter_countries(df):
    """
    The function filters the dataframe by selecting only rows with
    country names found in the list 'countries'
    """
    df = df[df.index.isin(countries)]
    return df


def norm(array):
    """
    Function normalizes the values present in the array passed to the function
    and returns the normalized values
    """
    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array - min_val) / (max_val - min_val)

    return scaled


def norm_df(df, first=0, last=None):
    """
    This function is used to normalize a dataframe by calling the norm function
    """
    # Iterate over all numerical columns
    for col in df.columns[first:last]:  # Excluding the first column
        df[col] = norm(df[col])
    return df


# Define fitting function
def polynomial(x, a, b, c):
    """
    Calculate the value of a quadratic polynomial for a given input
    """
    return a * x**2 + b * x + c


def plot_country_data(country_name):
    # Read population and undernourishment data
    pop_df, pop_df_trnsps = read('API_SP.POP.TOTL_DS2_en_excel_v2_6299418.xls')
    underNourish_df, underNourish_df_transpose = read(
        'API_SN.ITK.DEFC.ZS_DS2_en_excel_v2_6298709.xls')

    # Filter population data for the selected country
    pop_filter_df = pd.DataFrame()
    pop_filter_df = pop_df[pop_df.index == country_name]

    pop_filter_df.columns.astype(int)

    # Create a DataFrame for population from 2000 to 2014
    df = pd.DataFrame({'2000': pop_filter_df[2000.0],
                       '2002': pop_filter_df[2002.0],
                       '2004': pop_filter_df[2004.0],
                       '2006': pop_filter_df[2006.0],
                       '2008': pop_filter_df[2008.0],
                       '2010': pop_filter_df[2010.0],
                       '2012': pop_filter_df[2012.0],
                       '2014': pop_filter_df[2014.0]}, index=pop_filter_df.index)

    # Plot bar chart for population
    df.plot.bar()
    plt.xlabel('Countries', size=12)
    plt.ylabel('Population', size=12)
    plt.title(f'Population in {country_name}', size=16)

    # Show the graph
    plt.show()

    # Normalize population data
    df_fit_trial = pd.DataFrame()
    df_fit_trial['2000'] = pop_df[2000]
    df_fit_trial['2014'] = pop_df[2014]
    df_fit_trial = norm_df(df_fit_trial)
    print(df_fit_trial.describe())
    print()

    # Perform clustering using KMeans
    for ic in range(2, 7):
        # Set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(df_fit_trial)

        # Extract labels and calculate silhouette score
        labels = kmeans.labels_
        print(ic, skmet.silhouette_score(df_fit_trial, labels))

    # Plot scatter plot with color-coded clusters
    kmeans = cluster.KMeans(n_clusters=5)
    kmeans.fit(df_fit_trial)

    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    plt.figure(figsize=(6.0, 6.0))

    scatter = plt.scatter(df_fit_trial['2000'], df_fit_trial['2014'],
                          c=labels, cmap="Accent")

    # Show cluster centers
    for ic in range(3):
        xc, yc = cen[ic, :]
        plt.plot(xc, yc, "dk", markersize=10)

    plt.xlabel("2000", size=12)
    plt.ylabel("2014", size=12)
    plt.title(f"Population year in (2000 vs 2014) - {country_name}", size=16)
    plt.show()

    # Update cluster information in the population dataframe
    cluster_df = pop_df
    cluster_df['Cluster'] = labels

    pop_df = filter_countries(cluster_df)

    # Fitting using curve fit
    underNourish_df_updated = pd.DataFrame()
    updated_final = pd.DataFrame()
    underNourish_df_transpose['years'] = underNourish_df_transpose.index.values
    year = underNourish_df_transpose['years'].tail(22)
    underNourish_df_updated[country_name] = underNourish_df_transpose[country_name]

    updated_final['Country'] = underNourish_df_updated.tail(22)

    xdata = updated_final['Country']

    # Perform curve fit
    params, cov = curve_fit(polynomial, year, xdata)

    # Plot data and fitted curve
    plt.scatter(year, xdata, label="data")
    plt.plot(year, polynomial(year, *params), label="fit", color='g')

    # Predict values for 2025 and 2030
    year_2025 = 2025
    year_2030 = 2030
    pred_2025 = polynomial(year_2025, *params)
    pred_2030 = polynomial(year_2030, *params)

    # Annotate the plot with predicted values
    plt.scatter([year_2025, year_2030], [pred_2025, pred_2030],
                color='r', label='Predictions')
    plt.annotate(f'2025: {pred_2025:.2f}', (year_2025, pred_2025),
                 textcoords="offset points", xytext=(-10, 10), ha='center', fontsize=8, color='r')
    plt.annotate(f'2030: {pred_2030:.2f}', (year_2030, pred_2030),
                 textcoords="offset points", xytext=(-10, 10), ha='center', fontsize=8, color='r')

    plt.title(f'Under Nourishment in {country_name} from 2000 to 2020')
    plt.xlabel("Year")
    plt.ylabel("Prevalence of undernourishment (% of population)")
    plt.legend()
    plt.show()


countries = ["India", "China"]

for country_name in countries:
    plot_country_data(country_name=country_name)
