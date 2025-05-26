import React, { useState } from 'react';
import axios from 'axios';
import Form from './Form';
import ProblemCard from './ProblemCard';

const API_BASE_URL = 'http://localhost:5000/api';

export default function Main() {
    const [recommendations, setRecommendations] = useState({
        hybrid: [],
        organic: [],
        byRating: [],
        byTopic: [],
        practice: []
    });
    const [userInfo, setUserInfo] = useState(null);
    const [submissions, setSubmissions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const analyzeUserData = (submissions, userInfo) => {
        // Find topics where user has low success rate
        const weakTopics = submissions?.reduce((acc, sub) => {
            const topic = sub.problem.tags?.[0];
            if (topic && sub.verdict !== 'OK') {
                acc[topic] = (acc[topic] || 0) + 1;
            }
            return acc;
        }, {});

        // Find rating range where user struggles
        const ratingStruggle = submissions?.reduce((acc, sub) => {
            if (sub.problem.rating && sub.verdict !== 'OK') {
                acc[sub.problem.rating] = (acc[sub.problem.rating] || 0) + 1;
            }
            return acc;
        }, {});

        return {
            weakTopics: Object.entries(weakTopics || {}).sort((a, b) => b[1] - a[1])[0]?.[0],
            ratingStruggle: Object.entries(ratingStruggle || {}).sort((a, b) => b[1] - a[1])[0]?.[0]
        };
    };

    const fetchRecommendations = async (handle) => {
        try {
            setLoading(true);
            setError(null);

            // Fetch all necessary data
            const [userInfoResponse, submissionsResponse, hybridResponse, organicResponse] = await Promise.all([
                axios.get(`${API_BASE_URL}/user-info/${handle}`),
                axios.get(`${API_BASE_URL}/user-submissions/${handle}`),
                axios.get(`${API_BASE_URL}/hybrid/${handle}?n=3`),
                axios.get(`${API_BASE_URL}/organic/${handle}?n=3`)
            ]);

            const userData = {
                userInfo: userInfoResponse.data.user_info,
                submissions: submissionsResponse.data.submissions
            };

            setUserInfo(userData.userInfo);
            setSubmissions(userData.submissions);

            const analysis = analyzeUserData(userData.submissions, userData.userInfo);

            // Organize recommendations into different categories
            const allRecs = {
                hybrid: hybridResponse.data.recommendations,
                organic: organicResponse.data.recommendations,
                byRating: organicResponse.data.recommendations.filter(p => p.rating === analysis.ratingStruggle),
                byTopic: organicResponse.data.recommendations.filter(p => p.tags?.includes(analysis.weakTopics)),
                practice: hybridResponse.data.recommendations.filter(p => p.rating <= userData.userInfo.rating - 100)
            };

            setRecommendations(allRecs);

        } catch (err) {
            setError(err.message || 'Failed to fetch recommendations');
        } finally {
            setLoading(false);
        }
    };

    const getRatingColor = (rating) => {
        if (rating < 1200) return '#808080';
        if (rating < 1400) return '#008000';
        if (rating < 1600) return '#03a89e';
        if (rating < 1900) return '#0000ff';
        if (rating < 2100) return '#aa00aa';
        if (rating < 2400) return '#ff8c00';
        return '#ff0000';
    };

    return (
        <main className="recommendations-main">
            <Form onSubmit={fetchRecommendations} />
            
            {loading && <div className="loading">Cargando recomendaciones...</div>}
            {error && <div className="error">{error}</div>}
            
            {userInfo && (
                <div className="user-info">
                    <h2>{userInfo.handle}</h2>
                    <p style={{ color: getRatingColor(userInfo.rating) }}>
                        Rating: {userInfo.rating}
                    </p>
                </div>
            )}

            <div className="recommendations-grid">
                {recommendations.hybrid.length > 0 && (
                    <section className="recommendation-section">
                        <h3>Recomendado para Ti</h3>
                        <p className="recommendation-explanation">
                            Basado en tus envíos exitosos recientes y tipos de problemas preferidos
                        </p>
                        <div className="problems-container">
                            {recommendations.hybrid.slice(0, 3).map((problem) => (
                                <ProblemCard 
                                    key={`${problem.contestId}${problem.index}`}
                                    problem={problem}
                                    problemStatistics={problem}
                                />
                            ))}
                        </div>
                    </section>
                )}

                {recommendations.byTopic.length > 0 && (
                    <section className="recommendation-section">
                        <h3>Para Mejorar tus Habilidades</h3>
                        <p className="recommendation-explanation">
                            Problemas enfocados en {recommendations.byTopic[0]?.tags?.[0]} - 
                            un área donde podrías necesitar más práctica
                        </p>
                        <div className="problems-container">
                            {recommendations.byTopic.slice(0, 2).map((problem) => (
                                <ProblemCard 
                                    key={`${problem.contestId}${problem.index}`}
                                    problem={problem}
                                    problemStatistics={problem}
                                />
                            ))}
                        </div>
                    </section>
                )}

                {recommendations.byRating.length > 0 && (
                    <section className="recommendation-section">
                        <h3>Problemas de Nivel Siguiente</h3>
                        <p className="recommendation-explanation">
                            Problemas de rating {recommendations.byRating[0]?.rating} - 
                            resolverlos te ayudará a superar tu nivel actual
                        </p>
                        <div className="problems-container">
                            {recommendations.byRating.slice(0, 2).map((problem) => (
                                <ProblemCard 
                                    key={`${problem.contestId}${problem.index}`}
                                    problem={problem}
                                    problemStatistics={problem}
                                />
                            ))}
                        </div>
                    </section>
                )}

                {recommendations.practice.length > 0 && (
                    <section className="recommendation-section">
                        <h3>La Práctica Hace al Maestro</h3>
                        <p className="recommendation-explanation">
                            Problemas un poco más sencillos para reforzar tus fundamentos y construir confianza
                        </p>
                        <div className="problems-container">
                            {recommendations.practice.slice(0, 2).map((problem) => (
                                <ProblemCard 
                                    key={`${problem.contestId}${problem.index}`}
                                    problem={problem}
                                    problemStatistics={problem}
                                />
                            ))}
                        </div>
                    </section>
                )}

                {recommendations.organic.length > 0 && (
                    <section className="recommendation-section">
                        <h3>Desafíate a Ti Mismo</h3>
                        <p className="recommendation-explanation">
                            Problemas ligeramente por encima de tu rating actual - perfectos para superarte
                        </p>
                        <div className="problems-container">
                            {recommendations.organic.slice(0, 2).map((problem) => (
                                <ProblemCard 
                                    key={`${problem.contestId}${problem.index}`}
                                    problem={problem}
                                    problemStatistics={problem}
                                />
                            ))}
                        </div>
                    </section>
                )}
            </div>
        </main>
    );
}