import { useState, useEffect } from "react";
import { perfumeAPI } from "../services/api";

export const useAuth = () => {
  const [user, setUser] = useState(null);
  const [isRegister, setIsRegister] = useState(false);
  
  // Campos del formulario
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [lastname, setLastname] = useState("");
  
  const [message, setMessage] = useState("");

  // Mantener sesión al recargar
  useEffect(() => {
    const storedUser = localStorage.getItem("sessionUser");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  // Función principal
  const handleAuth = async (e) => {
    // 1. IMPORTANTE: Prevenir que el formulario recargue la página
    if (e && e.preventDefault) {
      e.preventDefault();
    }

    setMessage(""); // Limpiar mensajes viejos

    // 2. Validaciones estrictas (Frontend)
    if (!username.trim() || !password.trim()) {
      setMessage("❌ Faltan usuario o contraseña.");
      return;
    }

    if (isRegister && (!name.trim() || !lastname.trim())) {
      setMessage("❌ Por favor completa nombre y apellido.");
      return;
    }

    try {
      if (isRegister) {
        // --- REGISTRO ---
        console.log("Intentando registrar:", username);
        await perfumeAPI.register({ username, password, name, lastname });
        
        setMessage("✅ ¡Registro exitoso! Ahora inicia sesión.");
        setIsRegister(false); // Cambiar a pantalla de login
        setPassword(""); // Limpiar pass por seguridad
        // NO seteamos user aquí, obligamos a que haga login
        
      } else {
        // --- LOGIN ---
        console.log("Intentando login:", username);
        const userData = await perfumeAPI.login({ username, password });
        
        console.log("Datos recibidos del servidor:", userData);
        
        // Guardamos usuario y sesión
        setUser(userData); 
        localStorage.setItem("sessionUser", JSON.stringify(userData));
        setMessage(""); 
      }
    } catch (error) {
      console.error("Error Auth:", error);
      setMessage(`❌ ${error.message}`);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("sessionUser");
    setUser(null);
    setUsername("");
    setPassword("");
    setName("");
    setLastname("");
    setMessage("");
  };

  const resetForm = () => {
    setMessage("");
  };

  return {
    user,
    setUser,
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
    handleAuth, // Esta es la función clave
    handleLogout,
    resetForm
  };
};