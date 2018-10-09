'''
This script will predict the results of the NHL season given the current date
and determine the point distribution and playoff probabilities of each team
'''
import json
import multiprocessing as mp
import os
import pprint
import logging
import math
from datetime import datetime, date, timedelta
import requests
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sqlalchemy import create_engine

def pandas_clean_results(row, team):
    '''
    This funciton cleans the results dataframe for each team to it can be
    processed by the prediction model
    '''
    if row.home_team == team:
        new_row = row[['game_id', 'game_type', 'season', 'game_date', 'home_team_id', 'home_team', 'home_abbrev',
                       'home_score', 'away_score', 'ot_flag', 'shootout_flag', 'seconds_in_ot']]

        new_row.index = ['game_id', 'game_type', 'season', 'game_date', 'team_id', 'team', 'team_abbrev',
                       'goals_for', 'goals_against', 'ot_flag', 'shootout_flag', 'seconds_in_ot']

        new_row['is_home'] = 1

        return new_row

    elif row.away_team == team:
        new_row = row[['game_id', 'game_type', 'season', 'game_date', 'away_team_id', 'away_team', 'away_abbrev',
                       'away_score', 'home_score', 'ot_flag', 'shootout_flag', 'seconds_in_ot']]
        new_row.index = ['game_id', 'game_type', 'season', 'game_date', 'team_id', 'team', 'team_abbrev',
                       'goals_for', 'goals_against', 'ot_flag', 'shootout_flag', 'seconds_in_ot']

        new_row['is_home'] = 0

        return new_row

def game_predict(home_results, away_results):
    '''
    taking the results of each team this function predicts whether a team wins
    in Regulation, OT, or SO

    Inputs:
    home_results - results of the last 82 games of the home team
    away_results - results of the last 82 games of the away team

    Outputs:

    results - the amount of points won by each team in the simulation
    '''

    results ={}

#create the lambdas for the poisson distribution for regulation by adding the
#home team's goals for and away teams goals against and average. I reverse the
#process for the away team. This lambada is for the interval of one regulation
#game therefore 60 minutes. These are then adjusted for home ice since the avg
#home team wins 55% of the time I increase the home lambad by 5% to account for
#that and decrase the away lambda.
    home_lambda = ((home_results['non_ot_goals_for'].mean()
                    + away_results['non_ot_goals_against'].mean()) / 2) * 1.05

    away_lambda = ((home_results['non_ot_goals_against'].mean()
                    + away_results['non_ot_goals_for'].mean()) / 2) / 1.05

#This creates OT lambdas the same way as above but adjusts for a five minute
#interval which is the length of OT in the NHL by dividing by the total seconds
#in OT to get goal per second and then multiplyting by 300 to get goals per 5
#minutes. I adjust for home and away with the same method as I do above for
#regulation time
    home_ot_lambda = ((((home_results['ot_goals'].sum()/home_results['seconds_in_ot'].sum()) * 300) + \
                      (away_results['ot_goals_against'].sum()/away_results['seconds_in_ot'].sum())*300)/2) * 1.05

    away_ot_lambda = ((((away_results['ot_goals'].sum()/away_results['seconds_in_ot'].sum()) * 300) + \
                      (home_results['ot_goals_against'].sum()/home_results['seconds_in_ot'].sum())*300)/2) / 1.05

#calculate home and away OT win percentages for the Terry-Bradley models that
#determine the probabilites of the binomial flip in OT
    home_ot_win_percent = np.where((home_results.ot_flag == 1) &
                                   (home_results.shootout_flag == 0) &
                                   (home_results.goals_for > home_results.goals_against), 1, 0)\
                                   .sum()/home_results[(home_results.ot_flag == 1)
                                                       & (home_results.shootout_flag == 0)].shape[0]

    away_ot_win_percent = np.where((away_results.ot_flag == 1) &
                                   (away_results.shootout_flag == 0) &
                                   (away_results.goals_for >
                                       away_results.goals_against), 1, 0)\
                                   .sum()/away_results[(away_results.ot_flag == 1) &
                                                       (away_results.shootout_flag == 0)].shape[0]

#calculate shootout percentages for the Bradley-Terry models that determine the
#winner of the Shootout Binomial flip
    home_so_win_percent = np.where((home_results.shootout_flag == 1) &
                                   (home_results.goals_for > home_results.goals_against),
                                   1, 0).sum()/home_results[home_results.shootout_flag == 1].shape[0]

    away_so_win_percent = np.where((away_results.shootout_flag == 1) &
                                   (away_results.goals_for > away_results.goals_against),
                                   1, 0).sum()/away_results[away_results.shootout_flag == 1].shape[0]

#these lines fill any nan results from the OT and SO win percentage calculations
#with an even coin flip
    if math.isnan(home_ot_win_percent):
        home_ot_win_percent = .5

    if math.isnan(away_ot_win_percent):
        away_ot_win_percent = .5

    if math.isnan(home_so_win_percent):
        home_so_win_percent = .5

    if math.isnan(away_so_win_percent):
        away_so_win_percent = .5

#Here we draw one sample from the poisson distributions for home and away
#teams using the lambdas calculated earlier for regulation goals scored
    home_reg_goals = np.random.poisson(home_lambda, 1)
    away_reg_goals = np.random.poisson(away_lambda, 1)

#just do one comparison of each game from the drawing of the poisson dist
#and determine the winner of each game and whether the game is
#decided by home OT and Shootout probability determined by a
#Bradley-Terry Model

    if home_reg_goals > away_reg_goals:
        results[home_results['team'].unique()[0]] = [1, 0, 0]
        results[away_results['team'].unique()[0]] = [0, 1, 0]
    elif away_reg_goals > home_reg_goals:
        results[home_results['team'].unique()[0]] = [0, 1, 0]
        results[away_results['team'].unique()[0]] = [1, 0, 0]
    else:
        prob_of_zero_goals = (math.exp(-home_ot_lambda) * math.exp(-away_ot_lambda))

        if np.random.binomial(1, prob_of_zero_goals) == 1:
            try:
                prob_of_home_so_win = home_so_win_percent/(home_so_win_percent + away_so_win_percent)
            except:
                logging.exception('Shootout probability calculation messed up')
                prob_of_home_so_win = .5
            if np.random.binomial(1, prob_of_home_so_win) == 1:
                results[home_results['team'].unique()[0]] = [1, 0, 0]
                results[away_results['team'].unique()[0]] = [0, 0, 1]
            else:
                results[home_results['team'].unique()[0]] = [0, 0, 1]
                results[away_results['team'].unique()[0]] = [1, 0, 0]

        else:
            try:
                prob_of_home_ot_win = home_ot_win_percent/(home_ot_win_percent + away_ot_win_percent)
            except:
                prob_of_home_ot_win = .5
                logging.exception('OT win probability calculation messed up')
            if np.random.binomial(1, prob_of_home_ot_win) == 1:
                results[home_results['team'].unique()[0]] = [1, 0, 0]
                results[away_results['team'].unique()[0]] = [0, 0, 1]
            else:
                results[home_results['team'].unique()[0]] = [0, 0, 1]
                results[away_results['team'].unique()[0]] = [1, 0, 0]


    return results

def clean_results(results_df, team, date):
    '''
    this function cleans the results dataframe and just strips out the wanted
    team results and creates a column for OT goals as well
    '''
    results_df = results_df[(results_df.game_date < date) & (results_df.game_type == 'R')]
    results_df = results_df[(results_df.home_team == team) | (results_df.away_team == team)]

    #looping through the results_df to pull out only the games the team variable played in.
    cleaned_df = results_df.apply(pandas_clean_results, args=(team,), axis=1)

    #cleaned_df = pd.concat(cleaned_results, axis=1).T

    #calculating non ot goals by seeing if the game went to ot or shootout and if so whether the team won or not.
    #if they did then they score one less goals than their final total if not then they scored their same goals for
    #amount
    cleaned_df['non_ot_goals_for'] = np.where(((cleaned_df.shootout_flag == 1) | (cleaned_df.ot_flag == 1)) &
                                          (cleaned_df.goals_for > cleaned_df.goals_against), cleaned_df.goals_for - 1,
                                          cleaned_df.goals_for)
    cleaned_df['non_ot_goals_against'] = np.where(((cleaned_df.shootout_flag == 1) | (cleaned_df.ot_flag == 1)) &
                                          (cleaned_df.goals_for < cleaned_df.goals_against), cleaned_df.goals_against - 1,
                                          cleaned_df.goals_against)
    cleaned_df['ot_goals'] = np.where(cleaned_df.shootout_flag == 0,
                                      cleaned_df.goals_for - cleaned_df.non_ot_goals_for, 0)
    cleaned_df['ot_goals_against'] = np.where(cleaned_df.shootout_flag == 0,
                                              cleaned_df.goals_against - cleaned_df.non_ot_goals_against,0)

    cleaned_df = cleaned_df.reset_index(drop=True)

    #only return the last two seasons of games
    cleaned_df = cleaned_df.sort_values(by=['game_date'], ascending=False).iloc[:83, :]

    return cleaned_df

def get_avg_df(df):
    '''
    This is to create an average of the time frame of the past results. This
    was only implemented because of Vegas who had no past results as an expansion
    team. I'm hoping to use this most likely with the new Seatle expansion team
    in the next couple years.

    Inputs:
    df - dataframe of past results

    Outputs:
    avg_df - a dataframe of avg results for games with no OT, OT and no Shootout,
             and games that goto shootout
    '''

#creates three different dataframes based on whether the game went to OT, SO,
#or ended in regulation
    reg = df[df.ot_flag != 1]
    ot = df[(df.ot_flag == 1) & (df.shootout_flag != 1)]
    shootout = df[df.shootout_flag == 1]

#creates averages for the results of the three dataframes which reduces them to
#one row
    reg_avg = reg[['home_score', 'away_score','seconds_in_ot']].mean()
    ot_avg = ot[['home_score', 'away_score', 'seconds_in_ot']].mean()
    shootout_avg = shootout[['home_score', 'away_score', 'seconds_in_ot']].mean()

#creates the ot_flag and shootout_flag that is in the table the data is pulled
#from. Will be neccesary for future calculations
    reg_avg['ot_flag'] = 0
    ot_avg['ot_flag'] = 1
    shootout_avg['ot_flag'] = 1

    reg_avg['shootout_flag'] = 0
    ot_avg['shootout_flag'] = 0
    shootout_avg['shootout_flag'] = 1

#combines the three series from average the three dataframes of the three types
#of game outcomes: Regulation, Overtime, Shootout
    avg_df = pd.concat([reg_avg, ot_avg, shootout_avg], axis=1).T

    #rename avg_df columns
    avg_df.columns = ['goals_for', 'goals_against', 'seconds_in_ot', 'ot_flag', 'shootout_flag']

#create columsn for ot and non ot_goals which will be used in the monte carlo
#simulations
    avg_df['non_ot_goals_for'] = np.where(((avg_df.shootout_flag == 1) | (avg_df.ot_flag == 1)) &
                                              (avg_df.goals_for > avg_df.goals_against), avg_df.goals_for - 1,
                                              avg_df.goals_for)
    avg_df['non_ot_goals_against'] = np.where(((avg_df.shootout_flag == 1) | (avg_df.ot_flag == 1)) &
                                              (avg_df.goals_for < avg_df.goals_against), avg_df.goals_against - 1,
                                              avg_df.goals_against)
    avg_df['ot_goals'] = np.where(avg_df.shootout_flag == 0,
                                          avg_df.goals_for - avg_df.non_ot_goals_for, 0)
    avg_df['ot_goals_against'] = np.where((avg_df.ot_flag == 1) & (avg_df.shootout_flag != 1), 1, 0)

    return avg_df

def get_remaining_sched(date):
    '''
    This function pulls the remaining NHL schedule and converts it
    to a pandas data frame

    Inputs:
    date - the date which the script is run

    Ouputs:
    schedule_df - dataframe of reamining games in the NHL season
    '''
    #pull NHL schedule
    schedule_url = f'https://statsapi.web.nhl.com/api/v1/schedule?startDate={date}&endDate=2019-04-06'
    req = requests.get(schedule_url)
    schedule_dict = req.json()

    today_games = {}

    #if there are no games exit the function with an empty dataframe
    if not schedule_dict['dates']:
        return pd.DataFrame()

    #parse the schedule to create a dataframe to feed to the prediction model
    for x in schedule_dict['dates']:
        for game in x['games']:
            today_games[game['gamePk']] = {}
            today_games[game['gamePk']]['date'] = date
            today_games[game['gamePk']]['home_team'] = game['teams']['home']['team']['name']
            today_games[game['gamePk']]['home_team_id'] = game['teams']['home']['team']['id']
            today_games[game['gamePk']]['away_team'] = game['teams']['away']['team']['name']
            today_games[game['gamePk']]['away_team_id'] = game['teams']['away']['team']['id']

    #turn dictionary of daily games to a dataframe:
    schedule_df = pd.DataFrame.from_dict(today_games, orient='index')
    schedule_df = schedule_df.reset_index()
    schedule_df.columns = ['game_id', 'game_date', 'home_team', 'home_team_id',
                              'away_team', 'away_team_id']
    schedule_df = schedule_df[schedule_df.game_id > 2018020000]

    logging.info('Remaining schedule parsed')

    return schedule_df

def get_standings(date):
    '''
    This function gets the current standings of the NHL from the date
    passed to it

    Inputs:
    date - date script is run

    Outputs:
    standings_df - dataframe of the NHL standings snapshot
    '''

    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    standings_query = f"SELECT * from nhl_tables.standings WHERE date = '{date}'"
    standings_df = pd.read_sql(standings_query, engine)

    logging.info('Standings queried and returned succesfully')

    return standings_df

def compile_predictions():
    '''
    this function takes all the results of the season simulations and
    calculates the playoff probabilites and points distribution of each team
    '''
#TODO finish this function probably need to add arguments at some point when
#I know what the outputs of predict_rest_season are going to be

def get_results(date):
    '''
    this function returns the results of all teams before a given date

    Inputs:
    date - date of script running

    Ouputs:
    results_df - dataframe of all teams results
    '''

    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    sql_query = 'SELECT * from nhl_tables.nhl_schedule'
    results_df = pd.read_sql(sql_query, con=engine,
                             parse_dates = {'game_date': '%Y-%m-%d'})
    results_df = results_df[results_df.game_date < date]

    return results_df

def predict_rest_season(standings_df, schedule_df, team_results_dict, date):
    '''
    This function predicts the rest of the season and tabulates a final points
    standings and then calculates which teams make the playoffs
    '''

#get results of teams that happened before the date which the script is run

    new_standings_df = standings_df.copy()
    for index, row in schedule_df.iterrows():
        game_id = row.game_id

        home_team = row.home_team
        away_team = row.away_team

#getting the sample of results for the home and away teams the try and excepts
#are incase there is an expansion team in the future i.e. Seattle
        home_results = team_results_dict[home_team]
        away_results = team_results_dict[away_team]

        game_result = game_predict(home_results, away_results)

        for key, value in game_result.items():
            for result, columns in zip(value, ['wins', 'losses', 'ot']):
                new_standings_df.loc[new_standings_df.team == key, (columns)] += result


    new_standings_df['points'] = new_standings_df['wins']*2 + new_standings_df['ot']

    return new_standings_df

def clean_allteams_results(teams, results_df, date):
    '''
    This function calculates all the teams results dataframes to predict the
    rest of the season and not have to calculate them each loop of the prediction

    Inputs:
    teams - list of teams
    results_df - dataframe of all the results

    Outputs:
    team_results_dict - dictionary containing all the cleaned results
                        dataframes of each team in the league
    '''

    team_results_dict = {}
    for team in teams:
        team_results_dict[team] = clean_results(results_df, team, date)

    return team_results_dict


def calc_playoffs(final_standings_df):
    '''
    This function determines which teams make the playoffs given a standings
    dataframe
    '''
    divisions = ['Pacific', 'Atlantic', 'Central', 'Metropolitan']

    division_berths = []
    wild_card_berths = []

    for division in divisions:
        division_standings = final_standings_df[final_standings_df.division == division].sort_values(by='points', ascending=False).reset_index(drop=True)
        division_berths.append(division_standings.loc[0, 'team'])
        division_berths.append(division_standings.loc[1, 'team'])
        division_berths.append(division_standings.loc[2, 'team'])
        if division == 'Pacific':
            central_div = final_standings_df[final_standings_df.division == 'Central'].sort_values(by='points', ascending=False).reset_index(drop=True)

            if division_standings.loc[3, 'points'] > central_div.loc[3, 'points']:
                wild_card_berths.append(division_standings.loc[3, 'team'])

                if division_standings.loc[4, 'points'] > central_div.loc[3, 'points']:
                    wild_card_berths.append(division_standings.loc[4, 'team'])

                elif division_standings.loc[4, 'points'] == central_div.loc[3, 'points']:
                    if np.random.binomial(1, .5, 1) == 1:
                        wild_card_berths.append(division_standings.loc[4, 'team'])
                    else:
                        wild_card_berths.append(central_div.loc[4, 'team'])

                else:
                    wild_card_berths.append(central_div.loc[3, 'team'])

            elif division_standings.loc[3, 'points'] < central_div.loc[3, 'points']:
                wild_card_berths.append(central_div.loc[3, 'team'])

                if central_div.loc[4, 'points'] > division_standings.loc[3, 'points']:
                    wild_card_berths.append(central_div.loc[4, 'team'])

                elif central_div.loc[4, 'points'] == central_div.loc[3, 'points']:
                    if np.random.binomial(1, .5, 1) == 1:
                        wild_card_berths.append(central_div.loc[4, 'team'])
                    else:
                        wild_card_berths.append(division_standings.loc[3, 'team'])
                else:
                    wild_card_berths.append(division_standings.loc[3, 'team'])
            elif division_standings.loc[3, 'points'] == central_div.loc[3, 'points']:
                    wild_card_berths.append(division_standings.loc[3, 'team'])
                    wild_card_berths.append(central_div.loc[3, 'team'])
        if division == 'Metropolitan':
            central_div = final_standings_df[final_standings_df.division == 'Atlantic'].sort_values(by='points', ascending=False).reset_index(drop=True)

            if division_standings.loc[3, 'points'] > central_div.loc[3, 'points']:
                wild_card_berths.append(division_standings.loc[3, 'team'])

                if division_standings.loc[4, 'points'] > central_div.loc[3, 'points']:
                    wild_card_berths.append(division_standings.loc[4, 'team'])

                elif division_standings.loc[4, 'points'] == central_div.loc[3, 'points']:
                    if np.random.binomial(1, .5, 1) == 1:
                        wild_card_berths.append(division_standings.loc[4, 'team'])
                    else:
                        wild_card_berths.append(central_div.loc[4, 'team'])

                else:
                    wild_card_berths.append(central_div.loc[3, 'team'])

            elif division_standings.loc[3, 'points'] < central_div.loc[3, 'points']:
                wild_card_berths.append(central_div.loc[3, 'team'])

                if central_div.loc[4, 'points'] > division_standings.loc[3, 'points']:
                    wild_card_berths.append(central_div.loc[4, 'team'])

                elif central_div.loc[4, 'points'] == central_div.loc[3, 'points']:
                    if np.random.binomial(1, .5, 1) == 1:
                        wild_card_berths.append(central_div.loc[4, 'team'])
                    else:
                        wild_card_berths.append(division_standings.loc[3, 'team'])
                else:
                    wild_card_berths.append(division_standings.loc[3, 'team'])
            elif division_standings.loc[3, 'points'] == central_div.loc[3, 'points']:
                    wild_card_berths.append(division_standings.loc[3, 'team'])
                    wild_card_berths.append(central_div.loc[3, 'team'])

    final_playoff_teams = division_berths + wild_card_berths
    final_standings_df['playoffs'] = np.where(final_standings_df.team.isin(final_playoff_teams), 1, 0)

    final_standings_df = final_standings_df.sort_values(by=['conference', 'division', 'points'], ascending=False)

    return final_standings_df

def predict_seasons(standings_df, sched_df, team_results_dict, date):
    '''
    this function predicts one season of the nhl using the standings
    of the date given and returns a dictionary with the points and whether
    they made the playoffs for each team

    Inputs:
    standings_df - NHL standings up to the date
    sched_df - remaining schedule of the NHL season
    team_results_dict - dictionary of each teams results for each season simmed
    date - date the script is run

    Outputs:
    team_results_dict - results with the current simmed season added
    '''

#create dictionary to hold all points totals and whether they made the playoffs
#or not to calculate playoff probabilities
    np.random.seed()
    total_results_dict = {}
    for team in team_results_dict.keys():
        total_results_dict[team] ={}
        total_results_dict[team]['points'] = []
        total_results_dict[team]['in_playoffs'] = []

    final_standings = predict_rest_season(standings_df, sched_df,
                                          team_results_dict, date)

    final_standings = calc_playoffs(final_standings)

    for index, row in final_standings.iterrows():
        total_results_dict[row.team]['points'].append(row.points)
        total_results_dict[row.team]['in_playoffs'].append(row.playoffs)

    return total_results_dict

def main():

#setup logging
    '''
    logging.basicConfig(filename='season_predict.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.DEBUG)
    '''

    date = datetime.now().strftime('%Y-%m-%d')
    #date = '2018-10-07'

    remaining_sched_df = get_remaining_sched(date)
    results_df = get_results(date)
    standings_df = get_standings(date)

#get the results dataframes for all teams all at once to speed up the process
    team_results_dict = clean_allteams_results(list(results_df.home_team.unique()),
                                               results_df, date)

    start = timer()

    iterations = 1000
    pool =  mp.Pool(os.cpu_count())
    pool_list = [pool.apply_async(predict_seasons,
                                  args=(standings_df, remaining_sched_df,
                                        team_results_dict, date
                                        )) for _ in range(iterations)]

#getting the results for each process after they are finished
    results = [f.get() for f in pool_list]
    pool.close()
    end = timer()

    print(f'Time to run sims: {(end - start)/60}')

    start = timer()
    final_results = results.pop(0)
    for x in range(len(results)):
            for key, value in results[x].items():
                final_results[key]['points'].append(results[x][key]['points'][0])
                final_results[key]['in_playoffs'].append(results[x][key]['in_playoffs'][0])
    end = timer()
    print(f'Time to merge sims: {(end - start)/60}')

    with open('results_file.txt', 'w') as f:
        json.dump(final_results, f)

    results_list = []
    for key, value in final_results.items():
        row = [key,
               sum(final_results[key]["in_playoffs"])/len(final_results[key]["in_playoffs"]),
               sum(final_results[key]["points"])/len(final_results[key]["points"])]
        results_list.append(row)

    probs_df = pd.DataFrame(results_list,
                            columns=['team', 'playoff_probs', 'avg_points'])
    print(probs_df)
    probs_df['date'] = date

    engine = create_engine(os.environ.get('DEV_DB_CONNECT'))
    probs_df.to_sql('season_predictions', schema='nhl_tables', con=engine,
              if_exists='append', index=False)


if __name__ == '__main__':
    main()



