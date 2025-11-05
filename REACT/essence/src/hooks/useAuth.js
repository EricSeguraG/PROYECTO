import { useState, useEffect } from "react";

export const useAuth = () => {
  const [user, setUser] = useState(null);
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [lastname, setLastname] = useState("");
  const [message, setMessage] = useState("");

  // Cargar usuario desde localStorage al iniciar
  useEffect(() => {
    const storedUser = localStorage.getItem("sessionUser");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const handleAuth = () => {
    if (!username || !password) {
      setMessage("Por favor completa los campos obligatorios.");
      return false;
    }

    if (isRegister && (!name || !lastname)) {
      setMessage("Por favor completa todos los campos.");
      return false;
    }

    const users = JSON.parse(localStorage.getItem("users") || "[]");
    const existingUser = users.find((u) => u.username === username);

    if (isRegister) {
      if (existingUser) {
        setMessage("El nombre de usuario ya existe.");
        return false;
      } else {
        const newUser = { username, password, name, lastname };
        users.push(newUser);
        localStorage.setItem("users", JSON.stringify(users));
        setMessage("✅ REGISTRADO CORRECTAMENTE \n  Ahora puedes iniciar sesión.");
        // Cambiar a modo login automáticamente después del registro exitoso
        setIsRegister(false);
        // Limpiar solo los campos de nombre y apellidos, mantener usuario y contraseña
        setName("");
        setLastname("");
        return false; // No acceder al modo user
      }
    } else {
      // Modo login
      if (!existingUser) {
        setMessage("Usuario no encontrado.");
        return false;
      } else if (existingUser.password !== password) {
        setMessage("Contraseña incorrecta.");
        return false;
      } else {
        // Login exitoso
        localStorage.setItem("sessionUser", JSON.stringify(existingUser));
        setUser(existingUser);
        setMessage("");
        return true; // Acceder al modo user
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("sessionUser");
    setUser(null);
    resetForm();
  };

  const resetForm = () => {
    setUsername("");
    setPassword("");
    setName("");
    setLastname("");
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
    setMessage,
    handleAuth,
    handleLogout,
    resetForm
  };
};