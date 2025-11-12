
import React from "react";
import { useAuth } from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";

export const HomeScreen = () => {
  const { user, handleLogout } = useAuth();
  const navigate = useNavigate();

  const salir = () => {
    handleLogout();
    navigate("/");
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="logo">BIENVENIDO</h1>

        <p className="subtitle">
          Hola <strong>{user?.name} {user?.lastname}</strong> 👋
        </p>

        <p style={{ marginTop: "1rem" }}>
          Estás dentro de la aplicación.
        </p>

        <button className="btn" onClick={salir}>
          Cerrar sesión
        </button>
      </div>
    </div>
  );
};
