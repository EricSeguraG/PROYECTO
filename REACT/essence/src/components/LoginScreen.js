import React from "react";

const LoginScreen = ({ 
  isRegister, 
  setIsRegister, 
  username, 
  setUsername, 
  password, 
  setPassword,
  name,
  setName,
  lastname,
  setLastname,
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
          {isRegister && (
            <>
              <input
                type="text"
                placeholder="Nombre"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="input"
              />
              <input
                type="text"
                placeholder="Apellidos"
                value={lastname}
                onChange={(e) => setLastname(e.target.value)}
                className="input"
              />
            </>
          )}
          <input
            type="text"
            placeholder="Nombre de usuario"
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
            {isRegister ? "REGISTRAR" : "ENTRAR"}
          </button>
        </form>

        {message && (
          <p className="message" style={{ whiteSpace: "pre-line" }}>
            {message}
          </p>
        )}

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