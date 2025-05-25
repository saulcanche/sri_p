import requests
import pandas as pd


class API_CF:
    BASE_URL = 'https://codeforces.com/api'
    DEFAUL_RATING = 1000 #for unrated users

    def get_problems(self):
        response = requests.get(f"{self.BASE_URL}/problemset.problems")
        response.raise_for_status()
        data = response.json()['result']
        problems = pd.DataFrame(data['problems'])
        statistics = pd.DataFrame(data['problemStatistics'])
        df = problems.merge(statistics[['contestId', 'index', 'solvedCount']],
                         on=['contestId', 'index'], how='left')
        return df

