#!/usr/bin/env python3
'''Scrape user location from GitHub'''

import sys
import requests
import time
from datetime import datetime


def get_user_location(api_url):
    '''Return user's location'''
    try:
        response = requests.get(api_url, timeout=10)  # Add timeout
        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')
            if location:
                print(location)
            else:
                print('Location not specified')
        elif response.status_code == 404:
            print('Not found')
        elif response.status_code == 403:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            current_time = datetime.utcnow().timestamp()
            reset_in = (reset_time - current_time) / 60
            print('Reset in {} min'.format(int(reset_in)))
        else:
            print('Error: {}'.format(response.status_code))
    except requests.exceptions.RequestException as e:
        print('Request failed: {}'.format(e))
    except ValueError as e:
        print('Invalid API URL or response: {}'.format(e))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ./2-user_location.py <API_URL>')
    else:
        api_url = sys.argv[1]
        get_user_location(api_url)
