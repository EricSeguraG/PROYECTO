import React, { useState } from "react";
import {
  Heart,
  BookOpen,
  FlaskConical,
  Copy,
  Star,
  Search,
  Home,
} from "lucide-react";
import "./App.css";

function App() {
  const [mode, setMode] = useState("start"); // start | login | user | guest
  const [user, setUser] = useState(null);
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [toast, setToast] = useState(""); // 👈 mensaje temporal para invitados

  const menuItems = [
    { icon: <Heart />, label: "WISHLIST" },
    { icon: <BookOpen />, label: "MI COLECCIÓN" },
    { icon: <FlaskConical />, label: "PERFUMES" },
    { icon: <Copy />, label: "CLONES/INSPIRACIONES" },
    { icon: <Star />, label: "CELEBRITIES" },
    { icon: <Search />, label: "MULTIBUSCADOR" },
  ];

  // 🔐 Autenticación local simulada
  const handleAuth = () => {
    if (!username || !password) {
      setMessage("Por favor completa ambos campos.");
      return;
    }

    const users = JSON.parse(localStorage.getItem("users") || "[]");
    const existingUser = users.find((u) => u.username === username);

    if (isRegister) {
      if (existingUser) {
        setMessage("El usuario ya existe.");
      } else {
        users.push({ username, password });
        localStorage.setItem("users", JSON.stringify(users));
        setMessage("Registro exitoso ✅ Ahora puedes iniciar sesión.");
        setIsRegister(false);
      }
    } else {
      if (!existingUser || existingUser.password !== password) {
        setMessage("Credenciales incorrectas.");
      } else {
        localStorage.setItem("sessionUser", JSON.stringify(existingUser));
        setUser(existingUser);
        setMode("user");
        setMessage("");
      }
    }
  };

  // 🚪 Cerrar sesión
  const handleLogout = () => {
    localStorage.removeItem("sessionUser");
    setUser(null);
    setMode("start");
  };

  // 🔔 Mostrar mensaje temporal (toast)
  const showToast = (text) => {
    setToast(text);
    setTimeout(() => setToast(""), 2500);
  };

  // --- Pantalla 1: Inicio ---
  if (mode === "start") {
    return (
      <div className="container">
        <div className="card">
          <h1 className="logo">ESSENCE</h1>
          <p className="subtitle">AROMAS</p>

          <button className="btn" onClick={() => setMode("login")}>
            <span className="icon">👤</span> USER
          </button>

          <button className="btn" onClick={() => setMode("guest")}>
            <span className="icon">👥</span> GUEST
          </button>
        </div>
      </div>
    );
  }

  // --- Pantalla 2: Login / Registro ---
  if (mode === "login") {
    return (
      <div className="container">
        <div className="card">
          <h1 className="logo">ESSENCE</h1>
          <p className="subtitle">{isRegister ? "Registro" : "Iniciar Sesión"}</p>

          <input
            type="text"
            placeholder="Usuario"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="input"
          />
          <input
            type="password"
            placeholder="Contraseña"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="input"
          />
          <button className="btn" onClick={handleAuth}>
            {isRegister ? "Registrarse" : "Entrar"}
          </button>

          <p className="message">{message}</p>

          <button
            className="link"
            onClick={() => {
              setIsRegister(!isRegister);
              setMessage("");
            }}
          >
            {isRegister
              ? "¿Ya tienes cuenta? Inicia sesión"
              : "¿No tienes cuenta? Regístrate"}
          </button>

          <button className="link" onClick={() => setMode("start")}>
            ← Volver al inicio
          </button>
        </div>
      </div>
    );
  }

  // --- Pantalla 3: Modo Usuario ---
  if (mode === "user") {
    return (
      <div className="user-mode">
        <header className="header">
          <div className="header-left">
            <h1 className="logo">ESSENCE</h1>
            <span className="user-mode-label">[USER: {user?.username}]</span>
          </div>
          <button className="exit-btn" onClick={handleLogout}>
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
  }

  // --- Pantalla 4: Modo Invitado ---
  if (mode === "guest") {
    const disabledLabels = ["WISHLIST", "MI COLECCIÓN"];

    return (
      <div className="guest-mode">
        <header className="header">
          <div className="header-left">
            <h1 className="logo">ESSENCE</h1>
            <span className="user-mode-label">[GUEST MODE]</span>
          </div>
          <button className="exit-btn" onClick={() => setMode("start")}>
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
  }
}

export default App;
