import React from "react";

const StartScreen = ({ onUserClick, onGuestClick }) => {
  return (
    <div className="video-background-container">
      {/* Video de fondo */}
      <video 
        autoPlay 
        muted 
        loop 
        playsInline
        className="video-background"
      >
        <source src="/videos/vid1.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      
      {/* Overlay casi transparente */}
      <div className="video-overlay"></div>
      
      {/* Contenido */}
      <div className="video-content">
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
    </div>
  );
};

export default StartScreen;