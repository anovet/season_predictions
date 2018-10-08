'''
This script pulls the probabilities of each team as they are calculated each day
and plots them over time splitting the results up into the different divisions
'''
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sqlalchemy import create_engine

def draw_graph(df, filename):
    '''
    this function takes a dataframe of probabilities by time and creates a line
    graph and labels it and saves to a file to tweet out later
    '''

    plt.figure()
    df = df[df.division == filename]
    df = df.sort_values(by='date')
    df = df.rename({'abbrev': 'Teams'}, axis='columns')
    ax = sns.lineplot(x='date', y="playoff_probs", hue="Teams", data=df, markers=True)
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    x_dates = df['date'].dt.strftime('%Y-%m-%d').sort_values().unique()
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
    ax.set_xticklabels(df['date'].dt.strftime('%m-%d-%Y').unique())
    #ax.figure.set_size_inches(12.5, 12.5)
    for x in df['Teams'].unique():
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text = f"{x}: {round(df[(df.date == df.date.max()) & (df.Teams == x)].loc[:, 'playoff_probs'].values[0] *100, 2)}%"
        ax.annotate(text, xy=(df.date.max(),
                              df[(df.date == df.date.max()) & (df.Teams == x)].loc[:, 'playoff_probs']),
                    bbox=props
                   )
    ax.figure.tight_layout()
    ax.figure.savefig(filename)

def main():

#this pulls in all the season predictions
    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    season_query = f"SELECT * FROM nhl_tables.season_predictions predict LEFT JOIN nhl_tables.nhl_teams team ON team.name=predict.team"
    season_df = pd.read_sql(season_query, engine)
    season_df.loc[:, ('date')] = season_df['date'].astype('datetime64[ns]')

    divisions = ['Central', 'Metropolitan', 'Pacific', 'Atlantic']

    for division in divisions:
        draw_graph(season_df, division)


if __name__ == '__main__':
    main()
