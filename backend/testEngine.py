import pandas as pd
from recommenderEngine import CodeforcesRecommendationEngine # Import your class
from API_CF import API_CF # Import your existing API class
import logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def main():
    # 1. Initialize the actual Codeforces API
    cf_api = API_CF()

    # 2. Initialize the Recommendation Engine with the actual API
    engine = CodeforcesRecommendationEngine(api_cf=cf_api)
    
    print("Initializing recommendation engine and fetching data from Codeforces. This might take a moment...")
    engine.initialize_data() # This will now call methods from your API_CF class

    print("\n--- Testing Cold Start Recommendations ---")
    # Scenario 1: Brand new user with no history, default rating
    print(f"\nRecommendations for a new user (cold start, default rating):")
    cold_start_recs = engine.cold_start_recommendations(n_recommendations=5)
    for i, rec in enumerate(cold_start_recs):
        print(f"{i+1}. {rec['name']} (Rating: {rec['rating']}, Tags: {', '.join(rec['tags'])}, Type: {rec['recommendation_type']}) - {rec['explanation']}")

    # Scenario 2: New user but with a specified initial rating and preferred tags
    print(f"\nRecommendations for a new user (cold start, rating 1200, preferred tags: math, greedy):")
    cold_start_recs_rated = engine.cold_start_recommendations(user_rating=1200, preferred_tags=['math', 'greedy'], n_recommendations=5)
    for i, rec in enumerate(cold_start_recs_rated):
        print(f"{i+1}. {rec['name']} (Rating: {rec['rating']}, Tags: {', '.join(rec['tags'])}, Type: {rec['recommendation_type']}) - {rec['explanation']}")

    # ---
    print("\n--- Testing Hybrid Recommendations (requires a valid Codeforces handle) ---")
    # IMPORTANT: Replace 'YourCodeforcesHandle' with an actual Codeforces handle for these tests to work!
    user_handle_to_test = 'Liuxito2040' # Using the handle you provided

    # Scenario 3: Existing user with some solved problems
    print(f"\nRecommendations for '{user_handle_to_test}' (hybrid):")
    hybrid_recs = engine.hybrid_recommendations(handle=user_handle_to_test, n_recommendations=5)
    
    if hybrid_recs:
        for i, rec in enumerate(hybrid_recs):
            # Check if 'recommendation_score' exists before trying to print it
            score_str = f"Score: {rec['recommendation_score']:.2f}" if 'recommendation_score' in rec else "N/A"
            print(f"{i+1}. {rec['name']} (Rating: {rec['rating']}, {score_str}, Tags: {', '.join(rec.get('tags', []))}, Type: {rec.get('recommendation_type', 'Unknown')}) - {rec['explanation']}")
    else:
        print(f"Could not get hybrid recommendations for '{user_handle_to_test}'. Make sure the handle is valid and has submissions, or the recommendation logic had an issue.")


    # ---
    print("\n--- Testing Organic Recommendations (requires a valid Codeforces handle) ---")
    # Scenario 4: Organic recommendations for an existing user (focus on skill gaps)
    print(f"\nOrganic recommendations for '{user_handle_to_test}':")
    organic_recs = engine.get_organic_recommendations(handle=user_handle_to_test, n_recommendations=3)
    if organic_recs:
        for i, rec in enumerate(organic_recs):
            print(f"{i+1}. {rec['name']} (Rating: {rec['rating']}, Tags: {', '.join(rec.get('tags', []))}, Type: {rec.get('recommendation_type', 'Unknown')}) - {rec['explanation']}")
    else:
        print(f"Could not get organic recommendations for '{user_handle_to_test}'. Make sure the handle is valid and has submissions.")


if __name__ == "__main__":
    main()