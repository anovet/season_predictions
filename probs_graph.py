'''
This script pulls the probabilities of each team as they are calculated each day
and plots them over time splitting the results up into the different divisions
'''
import os
from twython import Twython
import sys
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sqlalchemy import create_engine
from predict_season import get_yest_games

def get_twitter_keys(key_file):
    '''
    Funciton to get the twitter bot's api keys from a text file

    Input:
    key_file - text file holding api auth keys

    Output:
    keys_dict - dictionary with key names mapped to keys
    '''

    twitter_keys = key_file
    keys_dict = {}

    with open(twitter_keys, 'r') as keys:
        api_keys = []

        for line in keys:
            api_keys.append(line)

        api_keys = list(map(str.strip, api_keys))

        for key in api_keys:
            name_list = key.split(':')
            name_list = list(map(str.strip, name_list))
            key_name, key_value = name_list[0], name_list[1]
            keys_dict[key_name] = key_value
        return keys_dict

def tweet_results(file_names, date):
    '''
    this function tweets out the results of the prediciton model

    Inputs:
    df - dataframe of the results of the prediction model

    Outputs:
    None
    '''

    twitter_keys = get_twitter_keys(sys.argv[1])

    #set twitter API
    twitter = Twython(twitter_keys['Consumer Key'], twitter_keys['Consumer Secret Key'],
                      twitter_keys['Access Key'], twitter_keys['Access Secret Key'])

    tweet_string = f'{date} updated playoff probabilities.'

    media = []
    for name in file_names:
        photo = open(f'{name}.png', 'rb')
        media.append(twitter.upload_media(media=photo)['media_id'])
        photo.close()

    twitter.update_status(status=tweet_string, media_ids=media)

    media = []
    for name in file_names:
        photo = open(f'{name}barplot.png', 'rb')
        media.append(twitter.upload_media(media=photo)['media_id'])
        photo.close()

    twitter.update_status(status = f"Expected Point Amounts. Updated: {date}", media_ids=media)

def draw_bar_graph(df, filename):

    plt.figure()
    df = df[df.division == filename]
    df = df.sort_values(by='avg_points', ascending=False)
    df = df.rename({'abbrev': 'Teams'}, axis='columns')
    ax = sns.barplot(y="avg_points", x="Teams", data=df)
    ax.figure.set_size_inches(10.5, 8.5)
    ax.set_title(f"{filename} Division Expected Points")
    plt.xlabel('Team')
    plt.ylabel('Expected Points')
    ax.figure.tight_layout()
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % (p.get_height()),
                fontsize=12, color='black', ha='center', va='bottom')
    ax.figure.savefig(f'{filename}barplot.png')

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
    ax.set_title(f"{filename} Division Playoff Probabilities")
    x_dates = df['date'].dt.strftime('%Y-%m-%d').sort_values().unique()
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Probability of Making Playoffs')
    ax.set_xticklabels(df['date'].dt.strftime('%m-%d-%Y').unique())
    ax.figure.set_size_inches(10.5, 8.5)
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

    date = datetime.datetime.now().strftime('%Y-%m-%d')
    yest_date = datetime.datetime.now() - datetime.timedelta(1)
    yest_date = yest_date.strftime('%Y-%m-%d')
    print(date)

    yest_games = get_yest_games(yest_date)
    print(yest_games)

    if yest_games == None:
        return
#this pulls in all the season predictions
    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    season_query = f"SELECT * FROM nhl_tables.season_predictions predict LEFT JOIN nhl_tables.nhl_teams team ON team.name=predict.team"
    season_df = pd.read_sql(season_query, engine)
    season_df.loc[:, ('date')] = season_df['date'].astype('datetime64[ns]')

    divisions = ['Central', 'Metropolitan', 'Pacific', 'Atlantic']

    for division in divisions:
        draw_graph(season_df, division)
        draw_bar_graph(season_df, division)

    #tweet_results(divisions, date)


if __name__ == '__main__':
    main()
