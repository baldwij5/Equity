#

#################
import requests
import pandas as pd
import json
import csv

# csv_name = 'data.csv'
# state = '06'
# county = '037'
# tract = '*'
ed_weight = 0.175
non_white_weight = 0.175
non_english_weight = 0.175
poverty_weight = 0.175
med_income_weight = 0.3


# vars = 'NAME,' \
#        'B02001_001E,' \
#        'B17001_001E,' \
#        'B17001_002E,' \
#        'B01001H_001E,' \
#        'B01001H_006E,' \
#        'B01001H_007E,' \
#        'B01001H_008E,' \
#        'B01001H_009E,' \
#        'B01001H_010E,' \
#        'B01001H_011E,' \
#        'B01001H_012E,' \
#        'B01001H_013E,' \
#        'B01001H_014E,' \
#        'B01001H_015E,' \
#        'B01001H_016E,' \
#        'B01001H_021E,' \
#        'B01001H_022E,' \
#        'B01001H_023E,' \
#        'B01001H_024E,' \
#        'B01001H_025E,' \
#        'B01001H_026E,' \
#        'B01001H_027E,' \
#        'B01001H_028E,' \
#        'B01001H_029E,' \
#        'B01001H_030E,' \
#        'B01001H_031E,' \
#        'B06011_001E,' \
#        'B15002_001E,' \
#        'B15002_003E,' \
#        'B15002_004E,' \
#        'B15002_005E,' \
#        'B15002_006E,' \
#        'B15002_007E,' \
#        'B15002_008E,' \
#        'B15002_009E,' \
#        'B15002_010E,' \
#        'B15002_020E,' \
#        'B15002_021E,' \
#        'B15002_022E,' \
#        'B15002_023E,' \
#        'B15002_024E,' \
#        'B15002_025E,' \
#        'B15002_026E,' \
#        'B15002_027E,' \
#        'B06007_001E,' \
#        'B06007_008E'
#
#
# ###
# # B02001_001E: Population
#
# # B17001_001E: Income in the past 12 months below poverty line UNIVERSE
# # B17001_002E: Income in the past 12 months below poverty line
#
# # B01001H_001E: Sex By Age White Alone UNIVERSE
# # B01001H_002E: Sex By Age White Alone Male
# # B01001H_017E: Sex By Age White Alone Female
#
# # B06011_001E: Median Income
#
# # B15002_001E: Educational Attainment UNIVERSE
# # B15002_003E: Male, no schooling
# # B15002_004E: Male, nursery to 4th
# # B15002_005E: Male, 5th and 6th
# # B15002_006E: Male,7th and 8th
# # B15002_007E: Male, 9th
# # B15002_008E: Male,10th
# # B15002_009E: Male, 11th
# # B15002_010E: Male, 12th no diploma
#
# # B15002_020E: Female, no schooling
# # B15002_021E: Female, nursery to 4th
# # B15002_022E: Female, 5th and 6th
# # B15002_023E: Female, 7th and 8th
# # B15002_024E: Female, 9th
# # B15002_025E: Female, 10th
# # B15002_026E: Female, 11th
# # B15002_027E: Female, 12th no diploma
#
#
# # B06007_001E: Total Speak Other Language but speak English less than "very well" UNIVERSE
# # B06007_008E: Total Speak Other Language but speak English less than "very well"
#
# ###
# vars_list = [ 'Name',
#              'Population', # B02001_001E
#              'Income in the past 12 months below poverty line UNIVERSE', # B17001_001E
#              'Income in the past 12 months below poverty line', # B17001_002E
#              'Sex By Age White Alone UNIVERSE', # B01001H_001E
#              'Male, White Alone, 15 to 17', # B01001H_006E
#              'Male, White Alone, 18 and 19', # B01001H_007E
#              'Male, White Alone, 20 to 24', # B01001H_008E
#              'Male, White Alone, 25 to 29', # B01001H_009E
#              'Male, White Alone, 30 to 34', # B01001H_010E
#              'Male, White Alone, 35 to 44', # B01001H_011E
#              'Male, White Alone, 45 to 54', # B01001H_012E
#              'Male, White Alone, 55 to 64', # B01001H_013E
#              'Male, White Alone, 65 to 74', # B01001H_014E
#              'Male, White Alone, 75 to 84', # B01001H_015E
#              'Male, White Alone, 85 and over', # B01001H_016E
#              'Female, White Alone, 15 to 17',  # B01001H_021E
#              'Female, White Alone, 18 and 19',  # B01001H_022E
#              'Female, White Alone, 20 to 24',  # B01001H_023E
#              'Female, White Alone, 25 to 29',  # B01001H_024E
#              'Female, White Alone, 30 to 34',  # B01001H_025E
#              'Female, White Alone, 35 to 44',  # B01001H_026E
#              'Female, White Alone, 45 to 54',  # B01001H_027E
#              'Female, White Alone, 55 to 64',  # B01001H_028E
#              'Female, White Alone, 65 to 74',  # B01001H_029E
#              'Female, White Alone, 75 to 84',  # B01001H_030E
#              'Female, White Alone, 85 and over',  # B01001H_031E
#              'Median Income', # B06011_001E
#              'Educational Attainment UNIVERSE', # B15002_001E
#              'Male, no schooling', # B15002_003E
#              'Male, nursery to 4th', # B15002_004E
#              'Male, 5th and 6th', # B15002_005E
#              'Male, 7th and 8th', # B15002_006E
#              'Male, 9th', # B15002_007E
#              'Male, 10th', # B15002_008E
#              'Male, 11th', # B15002_009E
#              'Male, 12th no diploma', # B15002_010E
#              'Female, no schooling', # B15002_020E
#              'Female, nursery to 4th', # B15002_021E
#              'Female, 5th and 6th', # B15002_022E
#              'Female, 7th and 8th', # B15002_023E
#              'Female, 9th', # B15002_024E
#              'Female, 10th', # B15002_025E
#              'Female, 11th', # B15002_026E
#              'Female, 12th no diploma', # B15002_027E
#              'Total Speak Other Language but speak English less than "very well" UNIVERSE', # B06007_0081
#              'Total Speak Other Language but speak English less than "very well"', # B06007_008E
#              'State',
#              'County',
#              'Tract']
#
# url = 'https://api.census.gov/data/2018/acs/acs5?get='
#
# r = requests.get(url + vars + '&for=tract:' + tract + '&in=county:' + county + '&in=state:' + state)
# print(r)
# request_text = r.text
# print(r.text)
# data = json.loads(r.text)
# df = pd.DataFrame(data, columns=vars_list)
# print(data)
#
# df.to_csv(r'LACounty_CensusNEW.csv')

df_2 = pd.read_csv(r'LACounty_CensusNEW.csv')

print(df_2)

df_2[['Male, no schooling', 'Male, nursery to 4th', 'Male, 5th and 6th' , 'Male, 7th and 8th', 'Male, 9th', 'Male, 10th', 'Male, 11th', 'Male, 12th no diploma', 'Female, no schooling', 'Female, nursery to 4th', 'Female, 5th and 6th' , 'Female, 7th and 8th', 'Female, 9th', 'Female, 10th', 'Female, 11th', 'Female, 12th no diploma', 'Educational Attainment UNIVERSE', 'Sex By Age White Alone UNIVERSE', 'Total Speak Other Language but speak English less than "very well"', 'Total Speak Other Language but speak English less than "very well" UNIVERSE', 'Income in the past 12 months below poverty line', 'Income in the past 12 months below poverty line UNIVERSE', 'Percent Median Income']] =df_2[['Male, no schooling', 'Male, nursery to 4th', 'Male, 5th and 6th' , 'Male, 7th and 8th', 'Male, 9th', 'Male, 10th', 'Male, 11th', 'Male, 12th no diploma', 'Female, no schooling', 'Female, nursery to 4th', 'Female, 5th and 6th' , 'Female, 7th and 8th', 'Female, 9th', 'Female, 10th', 'Female, 11th', 'Female, 12th no diploma', 'Educational Attainment UNIVERSE', 'Sex By Age White Alone UNIVERSE', 'Total Speak Other Language but speak English less than "very well"', 'Total Speak Other Language but speak English less than "very well" UNIVERSE', 'Income in the past 12 months below poverty line', 'Income in the past 12 months below poverty line UNIVERSE', 'Median Income']].apply(pd.to_numeric)

df_2[['Male, White Alone, 15 to 17', 'Male, White Alone, 18 and 19', 'Male, White Alone, 20 to 24', 'Male, White Alone, 25 to 29', 'Male, White Alone, 30 to 34', 'Male, White Alone, 35 to 44', 'Male, White Alone, 45 to 54', 'Male, White Alone, 55 to 64', 'Male, White Alone, 65 to 74', 'Male, White Alone, 65 to 74', 'Male, White Alone, 85 and over', 'Female, White Alone, 15 to 17', 'Female, White Alone, 18 and 19', 'Female, White Alone, 20 to 24', 'Female, White Alone, 25 to 29', 'Female, White Alone, 30 to 34', 'Female, White Alone, 35 to 44', 'Female, White Alone, 45 to 54', 'Female, White Alone, 55 to 64', 'Female, White Alone, 65 to 74', 'Female, White Alone, 65 to 74', 'Female, White Alone, 85 and over']] = df_2[['Male, White Alone, 15 to 17', 'Male, White Alone, 18 and 19', 'Male, White Alone, 20 to 24', 'Male, White Alone, 25 to 29', 'Male, White Alone, 30 to 34', 'Male, White Alone, 35 to 44', 'Male, White Alone, 45 to 54', 'Male, White Alone, 55 to 64', 'Male, White Alone, 65 to 74', 'Male, White Alone, 65 to 74', 'Male, White Alone, 85 and over', 'Female, White Alone, 15 to 17', 'Female, White Alone, 18 and 19', 'Female, White Alone, 20 to 24', 'Female, White Alone, 25 to 29', 'Female, White Alone, 30 to 34', 'Female, White Alone, 35 to 44', 'Female, White Alone, 45 to 54', 'Female, White Alone, 55 to 64', 'Female, White Alone, 65 to 74', 'Female, White Alone, 65 to 74', 'Female, White Alone, 85 and over']].apply(pd.to_numeric)

# df_2[['State', 'County', 'Tract']] = df_2[['State', 'County', 'Tract']].apply(pd.as_type(str))

df_2['Percent Educational Attainment Less Than HS'] = ((df_2['Male, no schooling'] +
                                               df_2['Male, nursery to 4th'] +
                                               df_2['Male, 5th and 6th'] +
                                               df_2['Male, 7th and 8th'] +
                                               df_2['Male, 9th'] +
                                               df_2['Male, 10th'] +
                                               df_2['Male, 11th'] +
                                               df_2['Male, 12th no diploma'] +
                                               df_2['Female, no schooling'] +
                                               df_2['Female, nursery to 4th'] +
                                               df_2['Female, 5th and 6th'] +
                                               df_2['Female, 7th and 8th'] +
                                               df_2['Female, 9th'] +
                                               df_2['Female, 10th'] +
                                               df_2['Female, 11th'] +
                                               df_2['Female, 12th no diploma']) / df_2['Educational Attainment UNIVERSE']) * 100

df_2['Total Educational Attainment Less Than HS'] = (df_2['Male, no schooling'] +
                                               df_2['Male, nursery to 4th'] +
                                               df_2['Male, 5th and 6th'] +
                                               df_2['Male, 7th and 8th'] +
                                               df_2['Male, 9th'] +
                                               df_2['Male, 10th'] +
                                               df_2['Male, 11th'] +
                                               df_2['Male, 12th no diploma'] +
                                               df_2['Female, no schooling'] +
                                               df_2['Female, nursery to 4th'] +
                                               df_2['Female, 5th and 6th'] +
                                               df_2['Female, 7th and 8th'] +
                                               df_2['Female, 9th'] +
                                               df_2['Female, 10th'] +
                                               df_2['Female, 11th'] +
                                               df_2['Female, 12th no diploma'])

df_2['Percent Non-White Population'] = 100 - (((
    df_2['Male, White Alone, 15 to 17'] +
    df_2['Male, White Alone, 18 and 19'] +
    df_2['Male, White Alone, 20 to 24'] +
    df_2['Male, White Alone, 25 to 29'] +
    df_2['Male, White Alone, 30 to 34'] +
    df_2['Male, White Alone, 35 to 44'] +
    df_2['Male, White Alone, 45 to 54'] +
    df_2['Male, White Alone, 55 to 64'] +
    df_2['Male, White Alone, 65 to 74'] +
    df_2['Male, White Alone, 75 to 84'] +
    df_2['Male, White Alone, 85 and over'] +
    df_2['Female, White Alone, 15 to 17'] +
    df_2['Female, White Alone, 18 and 19'] +
    df_2['Female, White Alone, 20 to 24'] +
    df_2['Female, White Alone, 25 to 29'] +
    df_2['Female, White Alone, 30 to 34'] +
    df_2['Female, White Alone, 35 to 44'] +
    df_2['Female, White Alone, 45 to 54'] +
    df_2['Female, White Alone, 55 to 64'] +
    df_2['Female, White Alone, 65 to 74'] +
    df_2['Female, White Alone, 75 to 84'] +
    df_2['Female, White Alone, 85 and over']
                                        ) / df_2['Sex By Age White Alone UNIVERSE']) *100)

df_2['Total Non-White Population'] = (
    df_2['Male, White Alone, 15 to 17'] +
    df_2['Male, White Alone, 18 and 19'] +
    df_2['Male, White Alone, 20 to 24'] +
    df_2['Male, White Alone, 25 to 29'] +
    df_2['Male, White Alone, 30 to 34'] +
    df_2['Male, White Alone, 35 to 44'] +
    df_2['Male, White Alone, 45 to 54'] +
    df_2['Male, White Alone, 55 to 64'] +
    df_2['Male, White Alone, 65 to 74'] +
    df_2['Male, White Alone, 75 to 84'] +
    df_2['Male, White Alone, 85 and over'] +
    df_2['Female, White Alone, 15 to 17'] +
    df_2['Female, White Alone, 18 and 19'] +
    df_2['Female, White Alone, 20 to 24'] +
    df_2['Female, White Alone, 25 to 29'] +
    df_2['Female, White Alone, 30 to 34'] +
    df_2['Female, White Alone, 35 to 44'] +
    df_2['Female, White Alone, 45 to 54'] +
    df_2['Female, White Alone, 55 to 64'] +
    df_2['Female, White Alone, 65 to 74'] +
    df_2['Female, White Alone, 75 to 84'] +
    df_2['Female, White Alone, 85 and over']
                                        )

df_2['Percent Total Speak Other Language but speak English less than "very well'] = (df_2['Total Speak Other Language but speak English less than "very well"'] / df_2['Total Speak Other Language but speak English less than "very well" UNIVERSE']) * 100

df_2['Percent Below Poverty Line'] = (df_2['Income in the past 12 months below poverty line'] / df_2['Income in the past 12 months below poverty line UNIVERSE']) * 100

df_2['State'] = "0" + df_2['State'].astype(str)
df_2['County'] = "0" + df_2['County'].astype(str)
df_2['Tract'] = df_2['Tract'].astype(str)
df_2['FIPS_Interim'] = df_2['State'] + df_2['County'] + df_2['Tract']

for i in df_2['Median Income']:
    if i < 0:
        i = df_2['Median Income'].median()


df_3 = df_2[['Percent Educational Attainment Less Than HS', 'Total Educational Attainment Less Than HS', 'Total Non-White Population', 'Percent Non-White Population', 'Percent Total Speak Other Language but speak English less than "very well' , 'Total Speak Other Language but speak English less than "very well"', 'Percent Below Poverty Line', 'Income in the past 12 months below poverty line', 'Income in the past 12 months below poverty line', 'Median Income', 'State', 'County', 'Tract', 'Name', 'FIPS_Interim', 'Population']]

df_3.to_csv(r'LACounty_CensusNEW2.csv')

pd.read_csv(r'LACounty_CensusNEW2.csv')

df_3['Normalized Educational Attainment'] = (df_3['Percent Educational Attainment Less Than HS']/100)
df_3['Normalized Non-White'] = df_3['Percent Non-White Population']/100
df_3['Normalized Non-English Speaking'] = df_3['Percent Total Speak Other Language but speak English less than "very well']/100
df_3['Normalized Poverty Score'] = df_3['Percent Below Poverty Line']/100
df_3['Normalized Median Income'] = 1 - (df_3['Median Income']/df_3['Median Income'].max())

df_3['Equity Index'] = ((df_3['Normalized Educational Attainment'] * ed_weight) +
                             (df_3['Normalized Non-White'] * non_white_weight) +
                             (df_3['Normalized Non-English Speaking'] * non_english_weight) +
                             (df_3['Normalized Poverty Score'] * poverty_weight) +
                             (df_3['Normalized Median Income'] * med_income_weight))


df_4 = df_3[['State', 'County', 'Tract', 'Name', 'FIPS_Interim', 'Population', 'Normalized Educational Attainment',  'Normalized Non-White', 'Normalized Non-English Speaking', 'Normalized Poverty Score', 'Normalized Median Income', 'Percent Educational Attainment Less Than HS', 'Total Educational Attainment Less Than HS', 'Total Non-White Population', 'Percent Non-White Population', 'Percent Total Speak Other Language but speak English less than "very well' , 'Total Speak Other Language but speak English less than "very well"', 'Percent Below Poverty Line', 'Income in the past 12 months below poverty line', 'Income in the past 12 months below poverty line', 'Median Income']]
df_3.to_csv(r'LACounty_Census_MedIncWeight.csv')