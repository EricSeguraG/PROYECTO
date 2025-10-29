import React, { useState } from "react";
import { useAuth } from "./hooks/useAuth";
import StartScreen from "./components/StartScreen";
import LoginScreen from "./components/LoginScreen";
import UserScreen from "./components/UserScreen";
import GuestScreen from "./components/GuestScreen";
import "./App.css";

function App() {
  const [mode, setMode] = useState("start"); // start | login | user | guest
  const auth = useAuth();

  const handleAuthSuccess = () => {
    if (auth.handleAuth()) {
      setMode("user");
    }
  };

  const handleLogout = () => {
    auth.handleLogout();
    setMode("start");
  };

  // Renderizar pantallas
  if (mode === "start") {
    return (
      <StartScreen
        onUserClick={() => setMode("login")}
        onGuestClick={() => setMode("guest")}
      />
    );
  }

  if (mode === "login") {
    return (
      <LoginScreen
        {...auth}
        handleAuth={handleAuthSuccess}
        onBack={() => {
          setMode("start");
          auth.resetForm();
        }}
      />
    );
  }

  if (mode === "user") {
    return <UserScreen user={auth.user} onLogout={handleLogout} />;
  }

  if (mode === "guest") {
    return <GuestScreen onExit={() => setMode("start")} />;
  }
}

export default App;