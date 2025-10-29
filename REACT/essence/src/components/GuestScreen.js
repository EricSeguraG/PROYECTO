import React, { useState } from "react";
import { Heart, BookOpen, FlaskConical, Copy, Star, Search, Home } from "lucide-react";

const GuestScreen = ({ onExit }) => {
  const [toast, setToast] = useState("");

  const menuItems = [
    { icon: <Heart />, label: "WISHLIST" },
    { icon: <BookOpen />, label: "MI COLECCIÓN" },
    { icon: <FlaskConical />, label: "PERFUMES" },
    { icon: <Copy />, label: "CLONES/INSPIRACIONES" },
    { icon: <Star />, label: "CELEBRITIES" },
    { icon: <Search />, label: "MULTIBUSCADOR" },
  ];

  const disabledLabels = ["WISHLIST", "MI COLECCIÓN"];

  const showToast = (text) => {
    setToast(text);
    setTimeout(() => setToast(""), 2500);
  };

  return (
    <div className="guest-mode">
      <header className="header">
        <div className="header-left">
          <h1 className="logo">ESSENCE</h1>
          <span className="user-mode-label">[GUEST MODE]</span>
        </div>
        <button className="exit-btn" onClick={onExit}>
          <Home size={16} /> Salir
        </button>
      </header>

      <main className="menu-grid">
        {menuItems.map((item, i) => {
          const isDisabled = disabledLabels.includes(item.label);
          return (
            <button
              key={i}
              className={`menu-btn ${isDisabled ? "disabled" : ""}`}
              onClick={() =>
                isDisabled && showToast("🔒 Inicia sesión para acceder.")
              }
            >
              {item.icon}
              {item.label}
              {isDisabled && <span className="disabled-cross"> ❌</span>}
            </button>
          );
        })}
      </main>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
};

export default GuestScreen;