// src/components/UserScreen.js
import React from "react";
import { Heart, BookOpen, FlaskConical, Copy, Star, Search, Home } from "lucide-react";

const UserScreen = ({ user, onLogout, onSearchClick, onClonesClick, onCelebrityClick, onPerfumesClick }) => {
  const containerStyle = {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    overflow: 'hidden'
  };

  const videoStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    zIndex: 0
  };

  const overlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'rgba(0, 0, 0, 0.4)',
    zIndex: 1
  };

  const contentStyle = {
    position: 'relative',
    zIndex: 2,
    flex: 1,
    display: 'flex',
    flexDirection: 'column'
  };

  const menuItems = [
    { icon: <Heart />, label: "WISHLIST" },
    { icon: <BookOpen />, label: "MI COLECCIÓN" },
    { 
      icon: <FlaskConical />, 
      label: "PERFUMES",
      action: onPerfumesClick
    },
    { 
      icon: <Copy />, 
      label: "CLONES/INSPIRACIONES",
      action: onClonesClick
    },
    { 
      icon: <Star />, 
      label: "CELEBRITIES",
      action: onCelebrityClick
    },
    { 
      icon: <Search />, 
      label: "MULTIBUSCADOR",
      action: onSearchClick
    },
  ];

  return (
    <div style={containerStyle}>
      <video autoPlay muted loop playsInline style={videoStyle}>
        <source src="/videos/vid.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      
      <div style={overlayStyle}></div>

      <div style={contentStyle}>
        <header className="header" style={{ position: 'relative', zIndex: 3 }}>
          <div className="header-left">
            <h1 className="logo">ESSENCE</h1>
            <span className="user-mode-label">
              [USER: {user?.name || "Usuario"} {user?.lastname || ""}]
            </span>
          </div>
          <button className="exit-btn" onClick={onLogout}>
            <Home size={16} /> Salir
          </button>
        </header>

        <main className="menu-grid" style={{ position: 'relative', zIndex: 2, padding: '2rem' }}>
          {menuItems.map((item, i) => (
            <button 
              key={i} 
              className="menu-btn" 
              onClick={item.action || (() => console.log("Función en desarrollo"))}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </main>
      </div>
    </div>
  );
};

export default UserScreen;