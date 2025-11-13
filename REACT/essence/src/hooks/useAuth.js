import { useState, useEffect } from "react";

export const useAuth = () => {
  const [user, setUser] = useState(null);
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [lastname, setLastname] = useState("");
  const [message, setMessage] = useState("");

  // Cargar usuario desde localStorage al iniciar (solo para tener los datos)
  useEffect(() => {
    const storedUser = localStorage.getItem("sessionUser");
    if (storedUser) {
      try {
        const userData = JSON.parse(storedUser);
        setUser(userData);
      } catch (error) {
        console.error("Error parsing stored user:", error);
        localStorage.removeItem("sessionUser");
      }
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
        setIsRegister(false);
        setName("");
        setLastname("");
        return false;
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
        return true;
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