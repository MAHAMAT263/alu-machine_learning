#!/usr/bin/env python3
'''Script to fetch GitHub user location from API URL.'''

import sys
import requests
import time


def get_user_location(api_url):
    '''Fetch and print the user's location from GitHub API.'''
    try:
        response = requests.get(api_url)
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
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                reset_in = (int(reset_time) - int(time.time())) / 60
                print('Reset in {} min'.format(int(reset_in)))
            else:
                print('Reset time not available')
        else:
            print(f'Error: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ./2-user_location.py <API_URL>')
        sys.exit(1)
    api_url = sys.argv[1]
    get_user_location(api_url)
