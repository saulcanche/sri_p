from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from recommenderEngine import CodeforcesRecommendationEngine
from API_CF import API_CF
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the recommendation engine
logger.info("Initializing recommendation engine...")
cf_api = API_CF()
engine = CodeforcesRecommendationEngine(api_cf=cf_api)
engine.initialize_data()
logger.info("Recommendation engine initialized successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/api/cold-start', methods=['POST'])
def get_cold_start_recommendations():
    """Get recommendations for new users"""
    try:
        data = request.get_json()
        user_rating = data.get('rating')
        preferred_tags = data.get('preferred_tags', [])
        n_recommendations = data.get('n_recommendations', 5)
        recommendations = engine.cold_start_recommendations(
            user_rating=user_rating,
            preferred_tags=preferred_tags,
            n_recommendations=n_recommendations
        )

        return jsonify({
            "success": True,
            "recommendations": recommendations
        }), 200

    except Exception as e:
        logger.error(f"Error in cold start recommendations: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get recommendations",
            "message": str(e)
        }), 500

@app.route('/api/hybrid/<handle>', methods=['GET'])
def get_hybrid_recommendations(handle):
    """Get hybrid recommendations for a specific user"""
    try:
        n_recommendations = request.args.get('n', default=5, type=int)
        recommendations = engine.hybrid_recommendations(
            handle=handle,
            n_recommendations=n_recommendations
        )

        return jsonify({
            "success": True,
            "recommendations": recommendations
        }), 200

    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get recommendations",
            "message": str(e)
        }), 500

@app.route('/api/organic/<handle>', methods=['GET'])
def get_organic_recommendations(handle):
    """Get organic recommendations for a specific user"""
    try:
        n_recommendations = request.args.get('n', default=3, type=int)
        focus_area = request.args.get('focus_area', default=None)
        
        recommendations = engine.get_organic_recommendations(
            handle=handle,
            focus_area=focus_area,
            n_recommendations=n_recommendations
        )

        return jsonify({
            "success": True,
            "recommendations": recommendations
        }), 200

    except Exception as e:
        logger.error(f"Error in organic recommendations: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get recommendations",
            "message": str(e)
        }), 500

@app.route('/api/user-info/<handle>', methods=['GET'])
def get_user_info(handle):
    """Get user information"""
    try:
        user_info = cf_api.get_user_info(handle)
        return jsonify({
            "success": True,
            "user_info": user_info
        }), 200

    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get user information",
            "message": str(e)
        }), 500

@app.route('/api/user-submissions/<handle>', methods=['GET'])
def get_user_submissions(handle):
    """Get user submissions"""
    try:
        max_count = request.args.get('max_count', default=10000, type=int)
        submissions = cf_api.get_user_submissions(handle, max_count=max_count)
        
        # Convert DataFrame to dictionary format for JSON response
        submissions_list = submissions.to_dict('records')
        
        return jsonify({
            "success": True,
            "submissions": submissions_list
        }), 200

    except Exception as e:
        logger.error(f"Error getting user submissions: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get user submissions",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)