import React from "react";

const StartScreen = ({ onUserClick, onGuestClick }) => {
  return (
    <div className="container">
      <div className="card">
        <h1 className="logo">ESSENCE</h1>
        <p className="subtitle">AROMAS</p>

        <button className="btn" onClick={onUserClick}>
          <span className="icon">👤</span> USER
        </button>

        <button className="btn" onClick={onGuestClick}>
          <span className="icon">👥</span> GUEST
        </button>
      </div>
    </div>
  );
};

export default StartScreen;