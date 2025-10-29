import { useState, useEffect } from "react";

export const useAuth = () => {
  const [user, setUser] = useState(null);
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
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
      setMessage("Por favor completa ambos campos.");
      return false;
    }

    const users = JSON.parse(localStorage.getItem("users") || "[]");
    const existingUser = users.find((u) => u.username === username);

    if (isRegister) {
      if (existingUser) {
        setMessage("El usuario ya existe.");
        return false;
      } else {
        users.push({ username, password });
        localStorage.setItem("users", JSON.stringify(users));
        setMessage("Registro exitoso ✅ Ahora puedes iniciar sesión.");
        setIsRegister(false);
        return true;
      }
    } else {
      if (!existingUser || existingUser.password !== password) {
        setMessage("Credenciales incorrectas.");
        return false;
      } else {
        localStorage.setItem("sessionUser", JSON.stringify(existingUser));
        setUser(existingUser);
        setMessage("");
        return true;
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("sessionUser");
    setUser(null);
    setUsername("");
    setPassword("");
    setMessage("");
  };

  const resetForm = () => {
    setUsername("");
    setPassword("");
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
    message,
    setMessage,
    handleAuth,
    handleLogout,
    resetForm
  };
};