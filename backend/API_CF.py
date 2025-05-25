import requests
import pandas as pd


class API_CF:
    BASE_URL = 'https://codeforces.com/api'
    DEFAULT_RATING = 1000 #for unrated users

    def get_problems(self) -> pd.DataFrame:
        response = requests.get(f"{self.BASE_URL}/problemset.problems")
        response.raise_for_status()
        data = response.json()['result']
        problems = pd.DataFrame(data['problems'])
        statistics = pd.DataFrame(data['problemStatistics'])
        df = problems.merge(statistics[['contestId', 'index', 'solvedCount']],
                         on=['contestId', 'index'], how='left')
        return df
    def get_user_submissions(self, handle: str, max_count: int = 10000) -> pd.DataFrame:
        """get user's submissions, include solved & unsolved attempts, and compute struggle score."""
        resp = requests.get(
            f"{self.BASE_URL}/user.status?handle={handle}&from=1&count={max_count}"
        )
        resp.raise_for_status()
        submissions = pd.DataFrame(resp.json()['result'])

        # Add problem_id for grouping
        submissions['problem_id'] = submissions.problem.apply(lambda p: f"{p.get('contestId', 0)}{p['index']}")
        
        # Sort by creation time to analyze progress
        submissions = submissions.sort_values(by='creationTimeSeconds')

        problem_stats = []

        for pid, group in submissions.groupby('problem_id'):
            total_attempts = len(group)
            ok_subs = group[group.verdict == 'OK']
            first_sub = group.iloc[0]

            if ok_subs.empty:
                # Unsolved: max struggle
                problem_stats.append({
                    'problem_id': pid,
                    'name': first_sub['problem']['name'],
                    'contestId': first_sub.get('contestId', None),
                    'index': first_sub['problem']['index'],
                    'attempts': total_attempts,
                    'solved': False,
                    'time_to_solve_seconds': None,
                    'relative_time_in_contest': None,
                    'struggle_percent': 100.0,
                    'verdict': group.iloc[-1]['verdict'],
                    'language': group.iloc[-1]['programmingLanguage'],
                    'timeConsumedMillis': None,
                    'memoryConsumedBytes': None,
                })
            else:
                # Solved case
                first_ok = ok_subs.iloc[0]
                time_to_solve = first_ok['creationTimeSeconds'] - first_sub['creationTimeSeconds']
                struggle_score = min(1.0, total_attempts / 10) * 100  # cap at 100%
                problem_stats.append({
                    'problem_id': pid,
                    'name': first_ok['problem']['name'],
                    'contestId': first_ok.get('contestId', None),
                    'index': first_ok['problem']['index'],
                    'attempts': total_attempts,
                    'solved': True,
                    'time_to_solve_seconds': time_to_solve,
                    'relative_time_in_contest': first_ok.get('relativeTimeSeconds', None),
                    'struggle_percent': struggle_score,
                    'verdict': first_ok['verdict'],
                    'language': first_ok['programmingLanguage'],
                    'timeConsumedMillis': first_ok['timeConsumedMillis'],
                    'memoryConsumedBytes': first_ok['memoryConsumedBytes'],
                })

        return pd.DataFrame(problem_stats)

    def get_user_info(self, handle: str) -> dict:
        """Get comprehensive user information"""
        resp = requests.get(f"{self.BASE_URL}/user.info?handles={handle}")
        resp.raise_for_status()
        info = resp.json()['result'][0]
        return {
            'handle': info.get('handle'),
            'rating': info.get('rating', self.DEFAULT_RATING),
            'maxRating': info.get('maxRating', self.DEFAULT_RATING),
            'rank': info.get('rank', 'unrated'),
            'maxRank': info.get('maxRank', 'unrated'),
            'contribution': info.get('contribution', 0),
            'friendOfCount': info.get('friendOfCount', 0)
        }
    
    def get_user_rating_history(self, handle: str) -> pd.DataFrame:
        """Get a user's complete rating history"""
        resp = requests.get(f"{self.BASE_URL}/user.rating?handle={handle}")
        resp.raise_for_status()
        return pd.DataFrame(resp.json()['result'])

    def get_contest_list(self, gym: bool = False) -> pd.DataFrame:
        """Get list of all contests (including Gym if specified)"""
        resp = requests.get(f"{self.BASE_URL}/contest.list?gym={'true' if gym else 'false'}")
        resp.raise_for_status()
        return pd.DataFrame(resp.json()['result'])
    def get_rated_list(self, active_only: bool = True) -> pd.DataFrame:
        """Get list of all rated users (optionally active only)"""
        resp = requests.get(f"{self.BASE_URL}/user.ratedList?activeOnly={'true' if active_only else 'false'}")
        resp.raise_for_status()
        return pd.DataFrame(resp.json()['result'])
