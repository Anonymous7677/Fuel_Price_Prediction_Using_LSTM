import csv
import requests
import requests
import yfinance
from bs4 import BeautifulSoup
from datetime import datetime


def get_oil_dataset(oil_url, newfile):
    response = requests.get(oil_url)
    open(newfile, 'wb').write(response.content)

    return None


def get_forex_dataset(newfile):
    df = yfinance.download('MYRUSD=X', period='max')
    df_resampled = df['Close'].resample('W', label='left').mean().round(9)
    df_resampled.rename('MYR/USD', inplace=True)
    df_resampled.to_csv(newfile)

    return None


# Start scraping fuel price from data source
def get_request_content(fuel_url):
    list_of_rows = []

    page = requests.get(fuel_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table')
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        list_of_rows.append([ele for ele in cols if ele])

    return list_of_rows


# Check whether the datetime of the month match defined format: full or abbreviated
def validate_full_month(date_text):
    try:
        datetime.strptime(date_text, '%d %B %Y')
        return True
    except ValueError:
        return False
    

def validate_abbr_month(date_text):
    try:
        datetime.strptime(date_text, '%d %b %Y')
        return True
    except ValueError:
        return False


def get_filtered_data(fuel_url):
    list_of_rows = get_request_content(fuel_url)

    for i in range(1, len(list_of_rows), +1):
        # Eliminate the price comparison between new fuel and the previous one
        for j in range(1, len(list_of_rows[i])):
            original_item = list_of_rows[i][j]
            list_of_rows[i][j] = original_item[2:6]

        # Take only the end of the date within the date range given in the first element of each row
        for j in range(0, 1):
            # Example: 8 September - 14 September 2022
            # Find the index of the symbol '-' and take only the strings behind it
            item = list_of_rows[i][j]
            res = item.find('-')
            item = item[res+2:]

            # Eliminate the space or \xa0 in the first index
            if ((item[0] == ' ') or (item[0] == '\xa0')):
                item = item[1:]

            if (item[-1].isdigit() == False):
                item += ' 2022'

            # Change datetime format to %Y-%m-%d
            if (validate_full_month(item) == True):
                item = datetime.strptime(item, '%d %B %Y').date()
            elif (validate_abbr_month(item) == True):
                item = datetime.strptime(item, '%d %b %Y').date()
            else:
                item = datetime.strptime(item.replace(
                    'Sept', 'Sep'), '%d %b %Y').date()

            list_of_rows[i][j] = item

    return list_of_rows


def get_fuel_dataset(fuel_url, fuel_file):
    list_of_rows = get_filtered_data(fuel_url)
    list_of_rows[1:] = list(reversed(list_of_rows[1:]))

    with open(fuel_file, 'w') as f_output:
        writer = csv.writer(f_output)
        for item in list_of_rows:
            writer.writerow(
                [str(data).replace('Weekly Petrol Price Update', 'Date') for data in item])

    return None
