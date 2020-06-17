import os
from os import path
import shutil
import platform
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import geopandas as gpd

class COVIDCharts(object):
    
    def stackedBar(self, 
                   df, 
                   y1 = 'Confirmed Per Capita', 
                   y2 = 'Recovered Per Capita', 
                   y3 = 'Deaths Per Capita', 
                   ylabel='Cases per capita',
                   countries=['US', 'China', 'Italy', 'Spain', 'Korea, South', 'United Kingdom', 'Germany', 'Singapore', 'Taiwan']):
        
        sns.set_style("white")
        sns.set_context({"figure.figsize": (24, 10)})

        df = df[df.index.isin(countries)]

        #Plot 1 - background - "total" (top) series
        sns.barplot(x = df.index, y = df[y1], color = "#0000A3", linewidth = 0)
        #Plot 2 - next layer 
        sns.barplot(x = df.index, y = df[y2] + df[y3], color = "red", linewidth = 0)
        #Plot 3
        bottom_plot = sns.barplot(x = df.index, y = df[y2], color = "green", linewidth = 0)


        topbar = plt.Rectangle((0,0),0,0,fc="#0000A3", edgecolor = 'none', linewidth = 0)
        middlebar = plt.Rectangle((0,0),0,0,fc="red", edgecolor = 'none', linewidth = 0)
        bottombar = plt.Rectangle((0,0),0,0,fc='green',  edgecolor = 'none', linewidth = 0)
        l = plt.legend([bottombar, middlebar, topbar], ['Recovered', 'Deaths', 'Active'], loc=2, ncol = 1, prop={'size':20})
        l.draw_frame(False)

        #Optional code - Make plot look nicer
        sns.despine(left=True)
        bottom_plot.set_ylabel(ylabel)
        bottom_plot.set_xlabel("")

        #Set fonts to consistent 20pt size
        for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                    bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
            item.set_fontsize(20)
        return bottom_plot

    def timeSeriesLines(self, df, startDate, geos, ylabel = 'Confirmed Cases'):
        sns.set_style("white")
        sns.set_context({"figure.figsize": (10, 10)})
        time_plot = df[startDate:][geos].plot()
        time_plot.set_ylabel(ylabel)
        time_plot.set_xlabel("")
        #Set fonts to consistent 16pt size
        for item in ([time_plot.xaxis.label, time_plot.yaxis.label] + time_plot.get_xticklabels() + time_plot.get_yticklabels()):
            item.set_fontsize(16)
        time_plot.legend(fontsize=14)
        time_plot.tick_params(axis='both', which='minor', labelsize=16)
        return time_plot
    
    def timeSeriesBar(self, df, startDate, column, ylabel='Confirmed Cases per 100,000'):
        sns.set_style("white")
        sns.set_context({"figure.figsize": (15, 10)})
        
        x=df[startDate:].index
        labels = np.arange(start=1, stop=len(x.values)+1) 

        y = df[startDate:][column].values

        time_plot = sns.barplot(x=labels, y=y, color='blue')
        
        time_plot.set_ylabel(ylabel)
        time_plot.set_xlabel('Days since ' + startDate)
        #Set fonts to consistent 16pt size
        for item in ([time_plot.xaxis.label, time_plot.yaxis.label] + time_plot.get_yticklabels()):
            item.set_fontsize(16)
        time_plot.tick_params(axis='both', which='minor', labelsize=16)
        return time_plot
                        
    def timeSeriesTwoBars(self, df, startDate, state1, state2, ylabel = 'New cases per 100,000'):
        x=df[startDate:].index
        y1=df[startDate:][state1].values
        y2=df[startDate:][state2].values
        labels = np.arange(start=1, stop=len(x.values)+1) 
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(x - width/2, y1, width, label=state1)
        ax.bar(x + width/2, y2, width, label=state2)
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Days since ' + startDate)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc=2, fontsize=14)

        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):
            item.set_fontsize(16)
        
        fig.tight_layout()

        plt.show()
        return fig

    def timeSeriesUSBar(self, df, startDate, ylabel = "New cases per 100,000", title_text = "New Cases Per 100,000"):
        x=df[startDate:].index
        y1=df[startDate:][0].values

        labels = np.arange(start=1, stop=len(x.values)+1) 
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(x - width/2, y1, width)
        #ax.plot(x - width/2, y1)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Days since ' + startDate)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title_text)
        ax.title.set_fontsize(14)

        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):
            item.set_fontsize(16)
        
        fig.tight_layout()

        plt.show()
        return fig
    
    def timeSeriesBarAndLine(self, df_bar, df_line, startDate,
                             column=0, 
                             ylabel='Confirmed Cases per 100,000', 
                             title_text = "New Cases Per 100,000"):
        
        sns.set_style("white")
        sns.set_context({"figure.figsize": (15, 10)})

        x=df_bar[startDate:].index
        labels = np.arange(start=1, stop=len(x.values)+1) 

        y = df_bar[startDate:][column].values
        time_plot = sns.barplot(x=labels, y=y, color='blue')
        
        if df_line.empty:
            y1 = None
        else:
            y1 = df_line[startDate:][column].values
            #plt.plot(x,y1)
            ax = sns.pointplot(x=labels, y=y1, color='red')
            plt.setp(ax.collections, sizes=[0])
        
        time_plot.set_ylabel(ylabel)
        time_plot.set_xlabel('Days since ' + startDate)
        time_plot.set_title(title_text)

        #Set fonts to consistent 16pt size
        for item in ([time_plot.xaxis.label, time_plot.yaxis.label, time_plot.title] + time_plot.get_yticklabels()):
            item.set_fontsize(16)
        time_plot.tick_params(axis='both', which='minor', labelsize=16)
        return time_plot

    def mapWithColoredGeos(self, gdf, geo='STUSPS', cm='Reds', column='Cases', title = 'COVID-19 cases per 100,000 population in each state'):

        fig, ax = plt.subplots(1, figsize=(30,10))
        base = gdf.plot(ax=ax, color='GRAY') 
        max_cases = max(gdf[column]) + 1.0
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=max_cases))
        for g in gdf[geo]:
            cases = gdf.loc[gdf[geo]==g, column]
            gdf.loc[gdf[geo].isin([g]) == True].plot(ax=base, color=plt.get_cmap(cm)(cases/max_cases))

        ax.axis('off')

        # Create colorbar as a legend
        sm._A = [] # empty array for the data range
        cb = fig.colorbar(sm)
        cb.ax.fontsize = 20
        _ = ax.set_title(title, fontdict={'fontsize': '25', 'fontweight' : '3'})
        return fig

class COVIDCountryTotals(object):
    ''' COVIDCountryTotals  
    Inputs: 
        df
        df_pop
    Outputs: 
        df_sub_pc_sort
    '''

    def __init__(self, df, df_pop):
        self.df = df
        self.df_pop = df_pop

    def mergeCountriesWithPop(self):
        ''' 
        Take in a df for daily worldwide update data
        and another one (df_pop) for country popuplations
        '''

        df = self.df
        df_pop = self.df_pop

        # Group by Country and add
        df_sum = df.groupby(['Country_Region']).sum()
        
        # rename a couple columns to match between dfs
        df_pop_replaced = df_pop.replace(df_pop.loc[df_pop['Country Name'] == 'United States', ['Country Name']], 'US')
        df_pop_replaced = df_pop_replaced.replace(df_pop_replaced.loc[df_pop['Country Name'] == 'Korea, Rep.', ['Country Name']], 'Korea, South')
        
        # merge dfs
        df_merged = df_sum.merge(df_pop_replaced, how='left', left_on='Country_Region', right_on='Country Name')
        df_merged = df_merged.set_index('Country Name')

        return df_merged

    def countriesPerCapita(self):
        df = self.mergeCountriesWithPop()
        df_per_capita = df.copy()
        df_per_capita['Active Per Capita'] = df_per_capita['Active']/df_per_capita['2018']
        df_per_capita['Deaths Per Capita'] = df_per_capita['Deaths']/df_per_capita['2018']
        df_per_capita['Recovered Per Capita'] = df_per_capita['Recovered']/df_per_capita['2018']
        df_per_capita['Confirmed Per Capita'] = df_per_capita['Confirmed']/df_per_capita['2018']

        return df_per_capita


class COVIDTimeSeries(object):
    def __init__(self, df, df_pop):
        self.df = df
        self.df_pop = df_pop

    def mergeStatesWithPop(self):
        '''
        Merge the state time series data with the 
        population data per state
        '''

        df = self.df
        df_pop = self.df_pop

        df_sum = df.groupby(['Province_State']).sum()
        df_sum = df_sum.drop(['UID', 'code3', 'FIPS', 'Lat', 'Long_'], axis=1)
        if 'Population' in df_sum.columns:
            df_sum.drop('Population', axis=1)
        
        df_pop['States'] = df_pop['States'].replace(to_replace=r'\.', value='', regex=True)
        df_merged = df_sum.merge(df_pop, how='right', left_on='Province_State', right_on='States')
        
        df_merged = df_merged.set_index('States')

        return df_merged

    def statesPerCapita(self):
        df = self.mergeStatesWithPop()
        df_pc = (df.iloc[:,1:-1].div(df[['2018']].values, axis='columns')
                               .mul(100000))
        return df_pc

    def transposeStatesTimeSeries(self, df):
        '''
        Transpose the df so that it's easier to graph
        '''
        df_t = df.transpose()
        df_t['Date'] = pd.to_datetime(df_t.index)
        df_t = df_t.set_index(['Date'])

        return df_t
    
    def dropPopulation(self):
        df = self.mergeStatesWithPop()
        df = df.iloc[:,1:-1]
        return df

    def stateTimeSeries(self):
        df = self.dropPopulation()
        df_pc = self.statesPerCapita()
        
        df_t = self.transposeStatesTimeSeries(df)
        df_pc_t = self.transposeStatesTimeSeries(df_pc)

        return df_t, df_pc_t 

    def lagDaysAndSubtract(self, df, lag=1):
        df_lagged = df.shift(lag, axis='rows')
        df_diff = df - df_lagged
        return df_diff
    
    def stateDailyNewCases(self):
        df, df_pc = self.stateTimeSeries()

        df_daily = self.lagDaysAndSubtract(df, lag=1)
        df_daily_pc = self.lagDaysAndSubtract(df_pc, lag=1)

        return df_daily, df_daily_pc

    def usDailyNewCases(self): 
        df, df_pc = self.stateDailyNewCases()
        df_daily_us = pd.DataFrame(df.sum(axis=1))
        df_daily_us_pc = pd.DataFrame(df_pc.sum(axis=1))

        return df_daily_us, df_daily_us_pc

    def rollingAve(self):
        df_daily, df_daily_pc = self.stateDailyNewCases()
        df_daily_us, df_daily_us_pc = self.usDailyNewCases()

        df_list = []
        for df in [df_daily, df_daily_pc, df_daily_us, df_daily_us_pc]:
            df_rolling = df.rolling(window=7).mean()
            df_list.append(df_rolling)
        
        return tuple(df_list)

class COVIDCounties(object):
    def __init__(self, df, df_pop):
        self.df = df
        self.df_pop = df_pop

    def mergeCountiesWithPop(self):
        df = self.df
        df_pop = self.df_pop

        df_pop['County'] = (df_pop['County'].str.replace('County', '')
                                            .str.replace('.', '')
                                            .str.strip())
        df_pop['State'] = df_pop['State'].str.strip()
        df_pop['2019'] = (df_pop['2019'].str.replace(',','')
                                        .str.strip())  
        df_pop = pd.DataFrame(df_pop)
        df_merged = df.merge(df_pop, left_on=['Admin2', 'Province_State'], right_on=['County', 'State'])
        
        return df_merged

    def countiesPerCapita(self):
        df = self.mergeCountiesWithPop()

        df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'], axis=1)
        df = df.set_index(['State','County'])

        df_pc = (df.iloc[:,:-1]
                   .div(df[['2019']]
                   .astype(float)
                   .values, axis='columns')
                   .mul(100000))
        
        df = (df.iloc[:,:-1]
                .astype(float))

        return df, df_pc

    def transposeCountiesTimeSeries(self, df):
        '''
        Transpose the df so that it's easier to graph
        '''
        df_t = df.transpose()
        df_t['Date'] = pd.to_datetime(df_t.index)
        df_t = df_t.set_index(['Date'])

        return df_t
    
    def countiesTimeSeries(self):
        df, df_pc = self.countiesPerCapita()
        df_t = self.transposeCountiesTimeSeries(df)
        df_pc_t = self.transposeCountiesTimeSeries(df_pc)

        return df_t, df_pc_t

    def lagDaysAndSubtract(self, df, lag=1):
        df_lagged = df.shift(lag, axis='rows')
        df_diff = df - df_lagged
        return df_diff
    
    def countiesDailyNewCases(self):
        df, df_pc = self.countiesTimeSeries()
        df_daily = self.lagDaysAndSubtract(df, lag=1)
        df_daily_pc = self.lagDaysAndSubtract(df_pc, lag=1)
        
        return df_daily, df_daily_pc

    def rollingAve(self, win=7):
        df_daily, df_daily_pc = self.countiesDailyNewCases()
        df_rolling = df_daily.rolling(window=win).mean()
        df_rolling_pc = df_daily_pc.rolling(window=win).mean()
        
        return df_rolling, df_rolling_pc