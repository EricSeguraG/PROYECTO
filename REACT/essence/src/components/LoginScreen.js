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
  const containerStyle = {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden'
  };

  const videoStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    zIndex: 0
  };

  const overlayStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'rgba(0, 0, 0, 0.05)', // Overlay casi transparente
    zIndex: 1
  };

  const contentStyle = {
    position: 'relative',
    zIndex: 2,
    textAlign: 'center',
    width: '100%'
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleAuth();
  };

  const toggleMode = () => {
    setIsRegister(!isRegister);
    resetForm();
  };

  return (
    <div style={containerStyle}>
      {/* Video de fondo - mismo que StartScreen */}
      <video 
        autoPlay 
        muted 
        loop 
        playsInline
        style={videoStyle}
      >
        <source src="/videos/vid1.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      
      {/* Overlay para mejor contraste */}
      <div style={overlayStyle}></div>
      
      {/* Contenido */}
      <div style={contentStyle}>
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
    </div>
  );
};

export default LoginScreen;