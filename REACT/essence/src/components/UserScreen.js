import React from "react";
import { Heart, BookOpen, FlaskConical, Copy, Star, Search, Home } from "lucide-react";

const UserScreen = ({ user, onLogout }) => {
  const menuItems = [
    { icon: <Heart />, label: "WISHLIST" },
    { icon: <BookOpen />, label: "MI COLECCIÓN" },
    { icon: <FlaskConical />, label: "PERFUMES" },
    { icon: <Copy />, label: "CLONES/INSPIRACIONES" },
    { icon: <Star />, label: "CELEBRITIES" },
    { icon: <Search />, label: "MULTIBUSCADOR" },
  ];

  return (
    <div className="user-mode">
      <header className="header">
        <div className="header-left">
          <h1 className="logo">ESSENCE</h1>
          <span className="user-mode-label">[USER: {user?.username}]</span>
        </div>
        <button className="exit-btn" onClick={onLogout}>
          <Home size={16} /> Salir
        </button>
      </header>

      <main className="menu-grid">
        {menuItems.map((item, i) => (
          <button key={i} className="menu-btn">
            {item.icon}
            {item.label}
          </button>
        ))}
      </main>
    </div>
  );
};

export default UserScreen;