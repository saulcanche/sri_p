import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import math

class CodeforcesRecommendationEngine:
    """
    Comprehensive Codeforces Problem Recommendation Engine
    
    Features:
    - Cold Start handling
    - Hybrid recommendation model (Content-based + Collaborative + Knowledge-based)
    - Ranking and Boosting
    - Organic and explainable recommendations
    """
    
    def __init__(self, api_cf):
        self.api = api_cf
        self.problems_df = None
        self.users_df = None
        self.rating_ranges = {
            'newbie': (0, 1199),
            'pupil': (1200, 1399),
            'specialist': (1400, 1599),
            'expert': (1600, 1899),
            'candidate_master': (1900, 2099),
            'master': (2100, 2299),
            'international_master': (2300, 2399),
            'grandmaster': (2400, float('inf'))
        }
        self.difficulty_tags = ['implementation', 'math', 'greedy', 'dp', 'data structures', 
                               'brute force', 'constructive algorithms', 'graphs', 'sortings',
                               'binary search', 'dfs and similar', 'trees', 'strings', 'number theory']
        
        # Initialize components
        self.content_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.problem_features = None
        self.user_clusters = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_data(self):
        """Load and preprocess all necessary data"""
        self.logger.info("Loading problems data...")
        self.problems_df = self.api.get_problems()
        
        # Clean and enrich problems data
        self._preprocess_problems()
        
        self.logger.info("Loading users data...")
        self.users_df = self.api.get_rated_list(active_only=True)
        
        # Build feature matrices
        self._build_content_features()
        self._cluster_users()
        
        self.logger.info("Recommendation engine initialized successfully!")
    
    def _preprocess_problems(self):
        """Clean and enrich problems dataset"""
        # Handle missing values
        self.problems_df = self.problems_df.dropna(subset=['name', 'contestId', 'index'])
        
        # Create difficulty categories
        self.problems_df['difficulty_category'] = self.problems_df['rating'].apply(
            lambda x: self._get_difficulty_category(x) if pd.notna(x) else 'unrated'
        )
        
        # Process tags
        self.problems_df['tags_str'] = self.problems_df['tags'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        # Create problem ID
        self.problems_df['problem_id'] = (
            self.problems_df['contestId'].astype(str) + 
            self.problems_df['index'].astype(str)
        )
        
        # Calculate popularity score
        self.problems_df['popularity_score'] = np.log1p(
            self.problems_df['solvedCount'].fillna(0)
        )
        
        # Calculate recency score (newer problems get slight boost)
        current_time = datetime.now().timestamp()
        self.problems_df['recency_score'] = self.problems_df['contestId'].apply(
            lambda x: 1.0 + 0.1 * math.exp(-(current_time - x * 100000) / 1000000) if pd.notna(x) else 1.0
        )
    
    def _get_difficulty_category(self, rating):
        """Map rating to difficulty category"""
        if pd.isna(rating):
            return 'unrated'
        for category, (min_r, max_r) in self.rating_ranges.items():
            if min_r <= rating <= max_r:
                return category
        return 'grandmaster'
    
    def _build_content_features(self):
        """Build content-based feature matrix"""
        # Combine problem name and tags for text features
        text_features = (
            self.problems_df['name'].fillna('') + ' ' + 
            self.problems_df['tags_str'].fillna('')
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.content_vectorizer.fit_transform(text_features)
        
        # Numerical features
        numerical_features = []
        for _, row in self.problems_df.iterrows():
            features = [
                row.get('rating', 1000),  # Problem difficulty
                row.get('solvedCount', 0),  # Popularity
                len(row.get('tags', [])),  # Number of tags
                row.get('popularity_score', 0),
                row.get('recency_score', 1.0)
            ]
            numerical_features.append(features)
        
        numerical_features = np.array(numerical_features)
        numerical_features_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine text and numerical features
        self.problem_features = np.hstack([
            tfidf_matrix.toarray(),
            numerical_features_scaled
        ])
    
    def _cluster_users(self):
        """Cluster users for collaborative filtering"""
        if len(self.users_df) > 100:  # Only cluster if we have enough users
            user_features = []
            for _, user in self.users_df.iterrows():
                features = [
                    user.get('rating', 1000),
                    user.get('maxRating', 1000),
                    user.get('contribution', 0),
                    user.get('friendOfCount', 0)
                ]
                user_features.append(features)
            
            user_features = np.array(user_features)
            user_features_scaled = self.scaler.fit_transform(user_features)
            
            # Determine optimal number of clusters
            n_clusters = min(10, len(self.users_df) // 20)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self.user_clusters = kmeans.fit_predict(user_features_scaled)
                self.users_df['cluster'] = self.user_clusters

    def cold_start_recommendations(self, user_rating: int = None, preferred_tags: List[str] = None, 
                                   n_recommendations: int = 10) -> List[Dict]:
        """
        Handle cold start problem - recommend problems for new users
        
        Strategy:
        1. Use rating-based filtering if available
        2. Recommend popular problems in appropriate difficulty range
        3. Include diverse problem types
        """
        self.logger.info("Generating cold start recommendations...")
        
        if user_rating is None:
            user_rating = 1000  # Default rating for new users
        
        # Define appropriate difficulty range (slightly challenging)
        min_rating = max(800, user_rating - 200)
        max_rating = user_rating + 300
        
        # Filter problems by difficulty
        suitable_problems = self.problems_df[
            (self.problems_df['rating'].between(min_rating, max_rating)) |
            (self.problems_df['rating'].isna())
        ].copy()
        
        # Boost problems with preferred tags
        if preferred_tags:
            suitable_problems['tag_match_score'] = suitable_problems['tags'].apply(
                lambda tags: len(set(tags) & set(preferred_tags)) if isinstance(tags, list) else 0
            )
        else:
            suitable_problems['tag_match_score'] = 0
        
        # Calculate cold start score
        suitable_problems['cold_start_score'] = (
            suitable_problems['popularity_score'] * 0.4 +
            suitable_problems['tag_match_score'] * 0.3 +
            suitable_problems['recency_score'] * 0.2 +
            np.random.random(len(suitable_problems)) * 0.1  # Add randomness for diversity
        )
        
        # Get top recommendations
        recommendations = suitable_problems.nlargest(n_recommendations, 'cold_start_score')
        
        return self._format_recommendations(recommendations, "Cold Start")
    
    def _enhanced_cold_start_for_active_user(self, user_info: Dict, user_submissions: pd.DataFrame, n_recommendations: int) -> List[Dict]:
        """
        Enhanced cold start for users with limited solved problems but known rating
        """
        user_rating = user_info.get('rating', 1000)
        
        # For active competitive programmers, use a more aggressive difficulty range
        if user_rating >= 1200:  # Specialist and above
            min_rating = user_rating - 100  # Slightly easier for confidence
            max_rating = user_rating + 400  # More challenging problems
        else:
            min_rating = max(800, user_rating - 200)
            max_rating = user_rating + 300
        
        self.logger.info(f"Enhanced cold start for rating {user_rating}: difficulty range {min_rating}-{max_rating}")
        
        # Analyze attempted problems to understand preferences
        attempted_tags = set()
        if not user_submissions.empty:
            for _, submission in user_submissions.iterrows():
                problem_row = self.problems_df[self.problems_df['problem_id'] == submission['problem_id']]
                if not problem_row.empty:
                    tags = problem_row.iloc[0].get('tags', [])
                    if isinstance(tags, list):
                        attempted_tags.update(tags)
        
        preferred_tags = list(attempted_tags)[:5] if attempted_tags else ['implementation', 'math', 'greedy']
        
        return self.cold_start_recommendations(
            user_rating=user_rating,
            preferred_tags=preferred_tags,
            n_recommendations=n_recommendations
        )
    
    def _get_rating_appropriate_recommendations(self, user_info: Dict, user_submissions: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Get recommendations strictly based on user's rating level
        """
        user_rating = user_info.get('rating', 1000)
        
        # Define appropriate difficulty range
        min_rating = user_rating - 100
        max_rating = user_rating + 300
        
        # Filter problems by rating
        suitable_problems = self.problems_df[
            (self.problems_df['rating'].between(min_rating, max_rating)) &
            (self.problems_df['rating'].notna())  # Only rated problems
        ].copy()
        
        # Remove already solved problems
        solved_problem_ids = set(user_submissions[user_submissions['solved'] == True]['problem_id'])
        suitable_problems = suitable_problems[
            ~suitable_problems['problem_id'].isin(solved_problem_ids)
        ]
        
        # Score based on appropriateness for user level
        recommendations = []
        for _, problem in suitable_problems.iterrows():
            rating_diff = abs(problem['rating'] - user_rating)
            
            # Prefer problems slightly above user's rating
            if problem['rating'] > user_rating:
                difficulty_score = 1.0 - (rating_diff / 400)  # Penalty for being too hard
            else:
                difficulty_score = 0.8 - (rating_diff / 300)  # Penalty for being too easy
            
            popularity_score = problem.get('popularity_score', 0) * 0.3
            final_score = max(0, difficulty_score + popularity_score)
            
            recommendations.append((problem['problem_id'], final_score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:30]
    
    def _filter_by_user_level(self, recommendations: List[Dict], user_rating: int) -> List[Dict]:
        """
        Filter recommendations to ensure they're appropriate for user's level
        """
        if not recommendations:
            return recommendations
        
        # Define acceptable rating range based on user level
        if user_rating >= 1600:  # Expert+
            min_acceptable = user_rating - 200
            max_acceptable = user_rating + 500
        elif user_rating >= 1200:  # Specialist+
            min_acceptable = user_rating - 150
            max_acceptable = user_rating + 400
        else:  # Below specialist
            min_acceptable = max(800, user_rating - 200)
            max_acceptable = user_rating + 300
        
        filtered_recs = []
        for rec in recommendations:
            problem_rating = rec.get('rating')
            
            # Include unrated problems (they can be educational)
            if problem_rating is None or pd.isna(problem_rating):
                filtered_recs.append(rec)
            elif min_acceptable <= problem_rating <= max_acceptable:
                filtered_recs.append(rec)
            # Skip problems that are clearly too easy or too hard
        
        self.logger.info(f"Filtered {len(recommendations)} recommendations to {len(filtered_recs)} appropriate for rating {user_rating}")
        return filtered_recs
    def hybrid_recommendations(self, handle: str, n_recommendations: int = 10) -> List[Dict]:
        """Generate hybrid recommendations combining content-based, collaborative, and knowledge-based approaches"""
        logging.info(f"Generating hybrid recommendations for user: {handle}")
        
        # Get user info and submissions
        try:
            user_info = self.api.get_user_info(handle)
            if not user_info:
                logging.warning(f"User '{handle}' not found. Falling back to cold start recommendations.")
                return self.cold_start_recommendations(n_recommendations=n_recommendations)

            user_rating = user_info.get('rating', self.api.DEFAULT_RATING)
            logging.info(f"User {handle} rating: {user_rating}")

            user_submissions_raw = self.api.get_user_submissions(handle)

            if user_submissions_raw.empty:
                logging.info(f"User '{handle}' has no submissions. Falling back to cold start recommendations.")
                return self.cold_start_recommendations(user_rating=user_rating, n_recommendations=n_recommendations)

        except Exception as e:
            logging.error(f"Error fetching user data: {e}")
            return self.cold_start_recommendations(n_recommendations=n_recommendations)

        try:
            # Ensure problem_id is consistently formatted
            user_submissions_raw['problem_id'] = user_submissions_raw.apply(
                lambda row: f"{row.get('contestId', '')}{row.get('index', '')}" if pd.notna(row.get('contestId')) else row.get('index', ''),
                axis=1
            )

            solved_problems_df = user_submissions_raw[user_submissions_raw['solved'] == True]
            unsolved_problems_df = user_submissions_raw[~user_submissions_raw['solved']]

            logging.info(f"User has {len(solved_problems_df)} solved problems out of {len(user_submissions_raw)} total submissions")

            # Get skill gaps and struggle problems
            skill_gaps = self.get_skill_gaps(solved_problems_df, self.problems_df)
            struggle_problems = self.get_struggle_problems(unsolved_problems_df, self.problems_df)
            collaborative_recs = self._collaborative_recommendations(user_info, user_submissions_raw)

            # Convert collaborative recommendations to DataFrame if needed
            if collaborative_recs:
                collaborative_recs = pd.DataFrame(collaborative_recs, columns=['problem_id', 'collaborative_score'])
            else:
                collaborative_recs = pd.DataFrame(columns=['problem_id', 'collaborative_score'])

            # Filter out solved problems
            solved_problem_ids = solved_problems_df['problem_id'].unique()
            problems_to_consider = self.problems_df[~self.problems_df['problem_id'].isin(solved_problem_ids)].copy()

            if problems_to_consider.empty:
                logging.info("No suitable problems found for recommendations.")
                return []

            # Ensure rating column is numeric
            problems_to_consider['rating'] = pd.to_numeric(problems_to_consider['rating'], errors='coerce')
            problems_to_consider = problems_to_consider.dropna(subset=['rating'])

            # Initialize recommendation scores
            problems_to_consider['recommendation_score'] = 0.0

            # Apply skill gap scores
            if not skill_gaps.empty:
                # Group by problem_id and take the maximum skill gap score for each problem
                skill_gaps_by_problem = skill_gaps.groupby('problem_id')['skill_gap_score'].max().reset_index()
                temp_df = problems_to_consider.merge(
                    skill_gaps_by_problem,
                    on='problem_id',
                    how='left'
                )
                problems_to_consider['recommendation_score'] += temp_df['skill_gap_score'].fillna(0) * 0.4

            # Apply struggle scores
            if not struggle_problems.empty:
                temp_df = problems_to_consider.merge(
                    struggle_problems[['problem_id', 'struggle_score']],
                    on='problem_id',
                    how='left'
                )
                problems_to_consider['recommendation_score'] += temp_df['struggle_score'].fillna(0) * 0.3

            # Apply collaborative scores
            if not collaborative_recs.empty:
                temp_df = problems_to_consider.merge(
                    collaborative_recs[['problem_id', 'collaborative_score']],
                    on='problem_id',
                    how='left'
                )
                problems_to_consider['recommendation_score'] += temp_df['collaborative_score'].fillna(0) * 0.3

            # Final ranking and selection
            recommended_problems = problems_to_consider.sort_values(
                by=['recommendation_score', 'rating'],
                ascending=[False, True]
            ).head(n_recommendations)

            # Format results
            results = []
            for _, row in recommended_problems.iterrows():
                explanation = self._generate_explanation(
                    problem_info=row.to_dict(),
                    user_rating=user_rating,
                    recommendation_type="Hybrid"
                )
                results.append({
                    'contestId': row.get('contestId'),
                    'index': row.get('index'),
                    'name': row['name'],
                    'rating': row['rating'],
                    'tags': row.get('tags', []),
                    'solvedCount': row.get('solvedCount'),
                    'problem_id': row['problem_id'],
                    'recommendation_score': row['recommendation_score'],
                    'recommendation_type': "Hybrid",
                    'explanation': explanation
                })

            return results

        except Exception as e:
            logging.error(f"Error generating hybrid recommendations: {str(e)}")
            return self.cold_start_recommendations(user_rating=user_rating, n_recommendations=n_recommendations)
    
    def get_skill_gaps(self, solved_problems_df: pd.DataFrame, problems_df: pd.DataFrame) -> pd.DataFrame:
        """Identify skill gaps based on solved problems"""
        if solved_problems_df.empty:
            return pd.DataFrame(columns=['problem_id', 'tag', 'skill_gap_score'])

        # Merge solved problems with full problem details
        merged = solved_problems_df.merge(
            problems_df[['problem_id', 'tags', 'rating']],
            on='problem_id',
            how='left')
        
        # Count attempts and successes per tag
        all_tags = set()
        tag_performance = {}
        for _, row in merged.iterrows():
            tags = row.get('tags', [])
            if not isinstance(tags, list):
                continue
            for tag in tags:
                all_tags.add(tag)
                if tag not in tag_performance:
                    tag_performance[tag] = {'attempts': 0, 'solved': 0}
                tag_performance[tag]['attempts'] += 1
                if row['solved']:
                    tag_performance[tag]['solved'] += 1
        
        # Calculate skill gaps
        skill_gaps = []
        for tag in all_tags:
            stats = tag_performance.get(tag, {'attempts': 0, 'solved': 0})
            success_rate = stats['solved'] / stats['attempts'] if stats['attempts'] > 0 else 0
            skill_gap_score = 1 - success_rate  # Higher score = bigger gap
            
            # Find problems with this tag for recommendations
            problems_with_tag = problems_df[
                problems_df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)
            ]
            
            for _, problem in problems_with_tag.iterrows():
                skill_gaps.append({
                    'problem_id': problem['problem_id'],
                    'tag': tag,
                    'skill_gap_score': skill_gap_score
                })
        
        return pd.DataFrame(skill_gaps)
    
    def _content_based_recommendations(self, user_submissions: pd.DataFrame, 
                                       user_info: Dict) -> List[Tuple[str, float]]:
        """Content-based filtering using solved problems similarity"""
        solved_problems = user_submissions[user_submissions['solved'] == True]
        
        if solved_problems.empty:
            return []
        
        # Find solved problem indices in our dataset
        solved_indices = []
        for _, problem in solved_problems.iterrows():
            matching_problems = self.problems_df[
                self.problems_df['problem_id'] == problem['problem_id']
            ]
            if not matching_problems.empty:
                solved_indices.append(matching_problems.index[0])
        
        if not solved_indices:
            return []
        
        # Calculate average feature vector of solved problems
        solved_features = self.problem_features[solved_indices]
        user_profile = np.mean(solved_features, axis=0)
        
        # Calculate similarity with all problems
        sim_scores = cosine_similarity([user_profile], self.problem_features)[0]
        
        # Filter out already solved problems
        solved_problem_ids = set(solved_problems['problem_id'])
        recommendations = []
        
        for idx, similarity in enumerate(sim_scores):
            problem_id = self.problems_df.iloc[idx]['problem_id']
            if problem_id not in solved_problem_ids:
                recommendations.append((problem_id, similarity))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:50]
    
    def _collaborative_recommendations(self, user_info: Dict, 
                                       user_submissions: pd.DataFrame) -> List[Tuple[str, float]]:
        """Collaborative filtering based on similar users"""
        if self.user_clusters is None:
            return []
        
        user_rating = user_info.get('rating', 1000)
        
        # Find similar users (same rating range)
        similar_users = self.users_df[
            self.users_df['rating'].between(user_rating - 100, user_rating + 100)
        ]
        
        if len(similar_users) < 2:
            return []
        
        # For simplicity, recommend popular problems in similar rating range
        # In a real system, you'd analyze what similar users solved
        suitable_problems = self.problems_df[
            self.problems_df['rating'].between(user_rating - 100, user_rating + 200)
        ]
        
        solved_problem_ids = set(user_submissions['problem_id'])
        recommendations = []
        
        for _, problem in suitable_problems.iterrows():
            if problem['problem_id'] not in solved_problem_ids:
                collab_score = problem.get('popularity_score', 0) * 0.8
                recommendations.append((problem['problem_id'], collab_score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:50]
    
    def _knowledge_based_recommendations(self, user_submissions: pd.DataFrame, 
                                         user_info: Dict) -> List[Tuple[str, float]]:
        """Knowledge-based recommendations to fill skill gaps"""
        user_rating = user_info.get('rating', 1000)
        solved_problems = user_submissions[user_submissions['solved'] == True]
        
        # Analyze solved problem tags to identify gaps
        solved_tags = set()
        if not solved_problems.empty:
            for _, problem in solved_problems.iterrows():
                problem_row = self.problems_df[
                    self.problems_df['problem_id'] == problem['problem_id']
                ]
                if not problem_row.empty:
                    tags = problem_row.iloc[0].get('tags', [])
                    if isinstance(tags, list):
                        solved_tags.update(tags)
        
        # Find underrepresented tags
        missing_tags = set(self.difficulty_tags) - solved_tags
        
        recommendations = []
        solved_problem_ids = set(user_submissions['problem_id'])
        
        # Recommend problems with missing tags
        for missing_tag in missing_tags:
            tag_problems = self.problems_df[
                (self.problems_df['tags'].apply(
                    lambda x: missing_tag in x if isinstance(x, list) else False
                )) &
                (self.problems_df['rating'].between(user_rating - 100, user_rating + 200))
            ]
            
            for _, problem in tag_problems.head(5).iterrows():
                if problem['problem_id'] not in solved_problem_ids:
                    knowledge_score = 1.0 + problem.get('popularity_score', 0) * 0.3
                    recommendations.append((problem['problem_id'], knowledge_score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:30]
    
    def _combine_recommendations(self, content_recs: List[Tuple[str, float]], 
                                 collab_recs: List[Tuple[str, float]], 
                                 knowledge_recs: List[Tuple[str, float]],
                                 user_submissions: pd.DataFrame) -> List[Dict]:
        """Combine and rank recommendations from all approaches with boosting"""
        
        # Create a dictionary to accumulate scores
        combined_scores = {}
        
        # Weight different recommendation types
        weights = {'content': 0.4, 'collaborative': 0.3, 'knowledge': 0.3}
        
        # Add content-based recommendations
        for problem_id, score in content_recs:
            combined_scores[problem_id] = combined_scores.get(problem_id, 0) + \
                                         score * weights['content']
        
        # Add collaborative recommendations
        for problem_id, score in collab_recs:
            combined_scores[problem_id] = combined_scores.get(problem_id, 0) + \
                                         score * weights['collaborative']
        
        # Add knowledge-based recommendations
        for problem_id, score in knowledge_recs:
            combined_scores[problem_id] = combined_scores.get(problem_id, 0) + \
                                         score * weights['knowledge']
        
        # Apply boosting based on user performance patterns
        boosted_scores = self._apply_boosting(combined_scores, user_submissions)
        
        # Sort and format final recommendations
        sorted_recommendations = sorted(boosted_scores.items(), 
                                        key=lambda x: x[1], reverse=True)
        
        return self._format_recommendations_from_scores(sorted_recommendations)
    
    def _apply_boosting(self, scores: Dict[str, float], 
                         user_submissions: pd.DataFrame) -> Dict[str, float]:
        """Apply boosting based on user patterns and problem characteristics"""
        
        # Analyze user's struggle patterns
        avg_struggle = user_submissions['struggle_percent'].mean() if not user_submissions.empty else 50
        recent_performance = self._get_recent_performance(user_submissions)
        
        boosted_scores = {}
        
        for problem_id, base_score in scores.items():
            boost_factor = 1.0
            
            # Get problem details
            problem_row = self.problems_df[
                self.problems_df['problem_id'] == problem_id
            ]
            
            if not problem_row.empty:
                problem = problem_row.iloc[0]
                
                # Boost 1: Difficulty adjustment based on recent performance
                if recent_performance > 0.7:  # User doing well
                    if problem.get('rating', 1000) > 1200:
                        boost_factor *= 1.2  # Slightly harder problems
                elif recent_performance < 0.3:  # User struggling
                    if problem.get('rating', 1000) < 1400:
                        boost_factor *= 1.3  # Easier problems
                
                # Boost 2: Diversity boost for different problem types
                problem_tags = problem.get('tags', [])
                if isinstance(problem_tags, list) and len(problem_tags) > 2:
                    boost_factor *= 1.1  # Multi-concept problems
                
                # Boost 3: Recency and popularity balance
                popularity = problem.get('popularity_score', 0)
                recency = problem.get('recency_score', 1.0)
                boost_factor *= (0.7 * popularity + 0.3 * recency)
            
            boosted_scores[problem_id] = base_score * boost_factor
        
        return boosted_scores
    
    def _get_recent_performance(self, user_submissions: pd.DataFrame) -> float:
        """Calculate user's recent performance (success rate in last 20 problems)"""
        if user_submissions.empty:
            return 0.5
        
        recent_submissions = user_submissions.tail(20)
        if recent_submissions.empty:
            return 0.5
        
        success_rate = len(recent_submissions[recent_submissions['solved'] == True]) / len(recent_submissions)
        return success_rate
    
    def _format_recommendations(self, problems_df: pd.DataFrame, recommendation_type: str, user_rating: float = None) -> List[Dict]:
        """
        Formats a DataFrame of recommended problems into a list of dictionaries.
        Includes an explanation and URL for each recommendation.
        """
        formatted_recs = []
        for _, problem in problems_df.iterrows():
            # Ensure problem['tags'] is a list, even if it's missing or None
            tags = problem.get('tags', [])
            if not isinstance(tags, list):
                tags = []
            
            # Generate URL for the problem
            contest_id = problem.get('contestId')
            index = problem.get('index')
            url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}" if contest_id and index else None
            
            # Pass user_rating to _generate_explanation if available
            explanation = self._generate_explanation(problem.to_dict(), user_rating, recommendation_type)
            
            rec_dict = {
                'contestId': contest_id,
                'index': index,
                'name': problem['name'],
                'rating': problem['rating'],
                'tags': tags,
                'solvedCount': problem.get('solvedCount'),
                'problem_id': problem.get('problem_id'),
                'recommendation_type': recommendation_type,
                'explanation': explanation,
                'url': url
            }
            # Add recommendation_score only if it exists (for hybrid, organic)
            if 'recommendation_score' in problem:
                rec_dict['recommendation_score'] = problem['recommendation_score']
            formatted_recs.append(rec_dict)
        return formatted_recs
    
    def _format_recommendations_from_scores(self, scored_recommendations: List[Tuple[str, float]]) -> List[Dict]:
        """Format scored recommendations for output"""
        recommendations = []
        
        for problem_id, score in scored_recommendations:
            problem_row = self.problems_df[
                self.problems_df['problem_id'] == problem_id
            ]
            
            if not problem_row.empty:
                problem = problem_row.iloc[0]
                contest_id = problem.get('contestId')
                index = problem.get('index')
                
                rec = {
                    'problem_id': problem_id,
                    'name': problem['name'],
                    'contestId': contest_id,
                    'index': index,
                    'rating': problem.get('rating'),
                    'tags': problem.get('tags', []),
                    'solvedCount': problem.get('solvedCount', 0),
                    'recommendation_score': round(score, 3),
                    'url': f"https://codeforces.com/problemset/problem/{contest_id}/{index}" if contest_id and index else None,
                    'recommendation_type': 'Hybrid',
                    'explanation': self._generate_explanation(problem, None, 'Hybrid')
                }
                recommendations.append(rec)
        
        return recommendations
    def _generate_explanation(self, problem_info: dict, user_rating: float, recommendation_type: str, focus_area: str = None) -> str:
        """Generate explanation for a recommended problem in Spanish."""
        if recommendation_type == "Hybrid" and 'recommendation_score' in problem_info:
            score = problem_info['recommendation_score']
            return f"Altamente recomendado según tu perfil (Puntuación: {score:.2f}); Rating: {problem_info['rating']}; Se enfoca en {', '.join(problem_info.get('tags', []))}. Este problema te ayudará a mejorar en áreas con oportunidad y se alinea con las preferencias de usuarios similares."
        
        elif recommendation_type == "Cold Start":
            nivel = "principiante"
            if problem_info['rating'] >= 1200:
                nivel = "aprendiz"
            elif problem_info['rating'] >= 1500:
                nivel = "especialista"
            
            tags_str = ', '.join(problem_info.get('tags', []))
            if len(problem_info.get('tags', [])) > 1:
                tags_str = f"combina los conceptos de **{tags_str}**"
            else:
                tags_str = f"se enfoca en **{tags_str}**"
            
            return f"Recomendado para principiantes; popular entre los usuarios; {tags_str}; este es un problema de nivel **{nivel}** (rating **{problem_info['rating']}**)"
        
        elif recommendation_type == "Organic":
            focus_str = f" con énfasis en {focus_area}" if focus_area else ""
            return f"Este problema está recomendado para mejorar tus habilidades{focus_str} (Rating: {problem_info['rating']}); Tags: {', '.join(problem_info.get('tags', []))}. Se alinea con tus intentos anteriores y ayuda a diversificar tus capacidades de resolución de problemas."
        
        return "Un buen problema para intentar." # Default fallback en español
    
    def get_organic_recommendations(self, handle: str, focus_area: str = None, 
                                    n_recommendations: int = 5) -> List[Dict]:
        """
        Generate organic, explainable recommendations focusing on skill development
        
        This provides transparent reasoning and focuses on educational value
        """
        self.logger.info(f"Generating organic recommendations for {handle}")
        
        try:
            user_info = self.api.get_user_info(handle)
            user_rating = user_info.get('rating', 1000)
            user_submissions = self.api.get_user_submissions(handle)
            
            if user_submissions.empty:
                return self.cold_start_recommendations(user_rating, n_recommendations=n_recommendations)
            
            # Analyze user's skill gaps and strengths
            skill_analysis = self._analyze_user_skills(user_submissions)
            
            # Generate targeted recommendations
            organic_recs = []
            
            # Focus on weak areas (educational approach)
            weak_areas = skill_analysis['weak_areas'][:3] # Top 3 weak areas
            
            for area in weak_areas:
                area_problems = self._get_problems_by_skill_area(
                    area, user_rating, user_submissions
                )
                
                if area_problems:
                    best_problem = area_problems[0]
                    organic_recs.append({
                        'problem_id': best_problem['problem_id'],
                        'name': best_problem['name'],
                        'contestId': best_problem.get('contestId'),
                        'index': best_problem.get('index'),
                        'rating': best_problem.get('rating'),
                        'tags': best_problem.get('tags', []),
                        'solvedCount': best_problem.get('solvedCount', 0),
                        'recommendation_type': 'Organic',
                        'explanation': self._generate_explanation(
                            problem_info=best_problem,
                            user_rating=user_rating,
                            recommendation_type='Organic',
                            focus_area=area
                        )
                    })
            
            # If not enough organic recommendations, supplement with general knowledge-based
            if len(organic_recs) < n_recommendations:
                supplemental_recs = self._knowledge_based_recommendations(user_submissions, user_info)
                current_problem_ids = {rec['problem_id'] for rec in organic_recs}
                
                for problem_id, score in supplemental_recs:
                    if problem_id not in current_problem_ids and len(organic_recs) < n_recommendations:
                        problem_row = self.problems_df[self.problems_df['problem_id'] == problem_id].iloc[0]
                        problem_dict = problem_row.to_dict()
                        organic_recs.append({
                            'problem_id': problem_id,
                            'name': problem_row['name'],
                            'contestId': problem_row.get('contestId'),
                            'index': problem_row.get('index'),
                            'rating': problem_row.get('rating'),
                            'tags': problem_row.get('tags', []),
                            'solvedCount': problem_row.get('solvedCount', 0),
                            'recommendation_type': 'Organic (Supplemental)',
                            'explanation': self._generate_explanation(
                                problem_info=problem_dict,
                                user_rating=user_rating,
                                recommendation_type='Organic'
                            )
                        })
                        current_problem_ids.add(problem_id)
            
            return organic_recs[:n_recommendations]

        except Exception as e:
            self.logger.error(f"Error generating organic recommendations: {e}")
            return self.cold_start_recommendations(user_info.get('rating', 1000) if 'user_info' in locals() else 1000, n_recommendations=n_recommendations)

    def _analyze_user_skills(self, user_submissions: pd.DataFrame) -> Dict:
        """
        Analyze user's performance across different tags to identify strengths and weaknesses.
        Returns a dictionary with 'strong_areas' and 'weak_areas'.
        """
        solved_problems = user_submissions[user_submissions['solved'] == True]
        
        tag_performance = {tag: {'solved': 0, 'attempted': 0} for tag in self.difficulty_tags}
        
        for _, submission in user_submissions.iterrows():
            problem_row = self.problems_df[self.problems_df['problem_id'] == submission['problem_id']]
            if not problem_row.empty:
                tags = problem_row.iloc[0].get('tags', [])
                for tag in tags:
                    if tag in tag_performance:
                        tag_performance[tag]['attempted'] += 1
                        if submission['solved']:
                            tag_performance[tag]['solved'] += 1
        
        # Calculate success rate for each tag
        tag_success_rates = {}
        for tag, data in tag_performance.items():
            if data['attempted'] > 0:
                tag_success_rates[tag] = data['solved'] / data['attempted']
            else:
                tag_success_rates[tag] = 0.5 # Neutral for tags not attempted
        
        # Identify strong and weak areas
        sorted_tags = sorted(tag_success_rates.items(), key=lambda item: item[1])
        
        weak_areas = [tag for tag, rate in sorted_tags if rate < 0.6]
        strong_areas = [tag for tag, rate in sorted_tags if rate >= 0.8]
        
        # If no weak areas found, consider least successful tags as weak
        if not weak_areas and sorted_tags:
            # Take bottom 3 tags as weak areas
            weak_areas = [tag for tag, _ in sorted_tags[:3]]
        
        # If still no weak areas, default to some common challenging tags
        if not weak_areas:
            weak_areas = ['dp', 'graphs', 'math'][:3]
        
        return {'strong_areas': strong_areas, 'weak_areas': weak_areas}

    def _get_problems_by_skill_area(self, tag: str, user_rating: int, user_submissions: pd.DataFrame) -> List[Dict]:
        """
        Get problems for a specific skill area (tag) appropriate for the user's rating.
        """
        solved_problem_ids = set(user_submissions['problem_id'])
        
        # Define a slightly broader rating range for targeted skill development
        min_rating = max(800, user_rating - 150)
        max_rating = user_rating + 250
        
        suitable_problems = self.problems_df[
            (self.problems_df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)) &
            (self.problems_df['rating'].between(min_rating, max_rating)) &
            (~self.problems_df['problem_id'].isin(solved_problem_ids))
        ].copy()
        
        # Rank by popularity and recency for diverse options
        suitable_problems['score'] = suitable_problems['popularity_score'] * 0.7 + \
                                      suitable_problems['recency_score'] * 0.3
        
        return suitable_problems.sort_values(by='score', ascending=False).to_dict('records')
    
    def get_struggle_problems(self, unsolved_problems_df: pd.DataFrame, problems_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify problems the user struggled with (many attempts, not solved).
        Returns a DataFrame with 'problem_id' and 'struggle_score'.
        """
        if unsolved_problems_df.empty:
            return pd.DataFrame(columns=['problem_id', 'struggle_score'])

        # Count attempts per problem
        attempts = unsolved_problems_df.groupby('problem_id').size().reset_index(name='attempts')
        # Merge with problem details
        merged = attempts.merge(problems_df, on='problem_id', how='left')
        # Struggle score: more attempts = higher struggle
        merged['struggle_score'] = merged['attempts'] / merged['attempts'].max()
        return merged[['problem_id', 'struggle_score']]