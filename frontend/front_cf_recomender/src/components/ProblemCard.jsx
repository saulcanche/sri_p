import React from 'react';
import './ProblemCard.css';
import Usermg from '../assets/user-icon.svg'

const ProblemCard = ({ problem, problemStatistics }) => {
  const baseUrl = 'https://codeforces.com/problemset/problem';

  const handleUrlRedirect = () => {
    const redirectUrl = `${baseUrl}/${problem.contestId}/${problem.index}`;
    window.open(redirectUrl, '_blank');
  };

  return (
    <div className="problem-card" onClick={handleUrlRedirect}>
      <div className="cell flex-1">{`${problemStatistics.contestId}${problemStatistics.index}`}</div>
      <div className="cell flex-2">{problem.name}</div>
      <div className="cell flex-1">{problem.rating === undefined ? 0 : problem.rating}</div>
      <div className="cell flex-1">
        <img src={Usermg} alt="" />
        <span>{`x${problemStatistics.solvedCount}`}</span>
      </div>
    </div>
  );
};

export default ProblemCard;
