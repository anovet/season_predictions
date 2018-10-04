'''
this scripts gets the standings of the nhl at the start of each day and inserts
them into an SQL database table for future predictions
'''
import os
import logging
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine


def standings_insert(df):
    '''
    function to insert that days standings into the database
    '''
    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    df.to_sql('standings', schema='nhl_tables', con=engine,
              if_exists='append', index=False)

    logging.info('Standings inserted into database')

def parse_standings(standings_dict, date):
    '''
    this function parses the NHL standings API and returns a dataframe

    Inputs:
    standings_dict - dictionary created from the standings NHL api

    Outputs:
    standings_df - pandas dataframe of the NHL standings
    '''

    #pulls actual standings list out of the API dictionary
    standings = standings_dict['records']
    standings_list = []
    for division in standings:
        div = division['division']['name']
        div_id = division['division']['id']
        conference = division['conference']['name']
        conf_id = division['conference']['id']
        for teams in division['teamRecords']:
            team = teams['team']['name']
            team_id = teams['team']['id']
            wins = teams['leagueRecord']['wins']
            losses = teams['leagueRecord']['losses']
            ot = teams['leagueRecord']['ot']
            points = 2*wins + ot
            standings_list.append([team, team_id, div, div_id, conference, conf_id,
                                   wins, losses, ot, points, date])

    standings_columns = ['team', 'team_id', 'division', 'division_id', 'conference',
                         'conference_id', 'wins', 'losses', 'ot', 'points', 'date']

    standings_df = pd.DataFrame(standings_list, columns=standings_columns)

    logging.info('Standings parsed from NHL api json')

    return standings_df

def get_standings(date):
    '''
    This function gets the NHL schedule from the NHL api and
    returns a dictionary

    Inputs:
    date - string of today's date

    Outputs:
    standings_dict - dictionary created from api JSON
    '''

    api_url = ('https://statsapi.web.nhl.com/api/v1/standings?'
               'date={}').format(date)

    req = requests.get(api_url)
    standings_dict = req.json()

    logging.info('Standings pulled from NHL api')

    return standings_dict

def main():

#setup logging
    logging.basicConfig(filename='standings_scraper.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.DEBUG)
#get todays date
    date = datetime.now().strftime('%Y-%m-%d')
    standings = get_standings(date)
    standings_df = parse_standings(standings, date)
    standings_insert(standings_df)

if __name__ == '__main__':
    main()
