import React from "react";

const LoginScreen = ({ 
  isRegister, 
  setIsRegister, 
  username, 
  setUsername, 
  password, 
  setPassword, 
  message, 
  handleAuth, 
  resetForm,
  onBack 
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    handleAuth();
  };

  const toggleMode = () => {
    setIsRegister(!isRegister);
    resetForm();
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="logo">ESSENCE</h1>
        <p className="subtitle">{isRegister ? "Registro" : "Iniciar Sesión"}</p>

        <form onSubmit={handleSubmit}>
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
          <button type="submit" className="btn">
            {isRegister ? "Registrarse" : "ENTRAR"}
          </button>
        </form>

        <p className="message">{message}</p>

        <button className="link" onClick={toggleMode}>
          {isRegister
            ? "¿Ya tienes cuenta? Inicia sesión"
            : "¿No tienes cuenta? Regístrate"}
        </button>

        <button className="link" onClick={onBack}>
          ← Volver al inicio
        </button>
      </div>
    </div>
  );
};

export default LoginScreen;