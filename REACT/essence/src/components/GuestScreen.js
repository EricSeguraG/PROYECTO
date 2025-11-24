// src/components/GuestScreen.js
import React, { useState } from "react";
import { Heart, BookOpen, FlaskConical, Copy, Star, Search, Home } from "lucide-react";

const GuestScreen = ({ onExit, onSearchClick, onClonesClick, onCelebrityClick, onPerfumesClick }) => {
  const [toast, setToast] = useState("");

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

  const showToast = (text) => {
    setToast(text);
    setTimeout(() => setToast(""), 2500);
  };

  const menuItems = [
    { 
      icon: <Heart />, 
      label: "WISHLIST",
      disabled: true
    },
    { 
      icon: <BookOpen />, 
      label: "MI COLECCIÓN",
      disabled: true
    },
    { 
      icon: <FlaskConical />, 
      label: "PERFUMES",
      disabled: false,
      action: onPerfumesClick
    },
    { 
      icon: <Copy />, 
      label: "CLONES/INSPIRACIONES",
      disabled: false,
      action: onClonesClick
    },
    { 
      icon: <Star />, 
      label: "CELEBRITIES",
      disabled: false,
      action: onCelebrityClick
    },
    { 
      icon: <Search />, 
      label: "MULTIBUSCADOR",
      disabled: false,
      action: onSearchClick
    },
  ];

  return (
    <div style={containerStyle}>
      <video autoPlay muted loop playsInline style={videoStyle}>
        <source src="/videos/vid2.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      
      <div style={overlayStyle}></div>

      <div style={contentStyle}>
        <header className="header" style={{ position: 'relative', zIndex: 3 }}>
          <div className="header-left">
            <h1 className="logo">ESSENCE</h1>
            <span className="user-mode-label">[GUEST MODE]</span>
          </div>
          <button className="exit-btn" onClick={onExit}>
            <Home size={16} /> Salir
          </button>
        </header>

        <main className="menu-grid" style={{ position: 'relative', zIndex: 2, padding: '2rem' }}>
          {menuItems.map((item, i) => (
            <button
              key={i}
              className={`menu-btn ${item.disabled ? "disabled" : ""}`}
              onClick={() => {
                if (item.disabled) {
                  showToast("🔒 INICIA SESION PARA ACCEDER.");
                } else if (item.action) {
                  item.action();
                }
              }}
            >
              {item.icon}
              {item.label}
              {item.disabled && <span className="disabled-cross"> ❌</span>}
            </button>
          ))}
        </main>

        {toast && <div className="toast" style={{ position: 'fixed', zIndex: 4 }}>{toast}</div>}
      </div>
    </div>
  );
};

export default GuestScreen;