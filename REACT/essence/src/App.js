import React, { useState, useEffect } from "react";
import { useAuth } from "./hooks/useAuth";
import StartScreen from "./components/StartScreen";
import LoginScreen from "./components/LoginScreen";
import UserScreen from "./components/UserScreen";
import GuestScreen from "./components/GuestScreen";
import SearchScreen from "./components/SearchScreen";
import "./App.css";

function App() {
  const [mode, setMode] = useState("start");
  const [searchMode, setSearchMode] = useState("user");
  const auth = useAuth();

  useEffect(() => {
    const storedUser = localStorage.getItem("sessionUser");
    if (storedUser) {
      try {
        const userData = JSON.parse(storedUser);
        auth.setUser(userData);
      } catch (error) {
        console.error("Error parsing stored user:", error);
        localStorage.removeItem("sessionUser");
      }
    }
  }, [auth]);

  const handleAuthSuccess = () => {
    if (auth.handleAuth()) {
      setMode("user");
    }
  };

  const handleLogout = () => {
    auth.handleLogout();
    setMode("start");
  };

  const handleSearchClick = (fromMode) => {
    setSearchMode(fromMode);
    setMode("search");
  };

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
    return (
      <UserScreen 
        user={auth.user} 
        onLogout={handleLogout}
        onSearchClick={() => handleSearchClick("user")}
      />
    );
  }

  if (mode === "guest") {
    return (
      <GuestScreen 
        onExit={() => setMode("start")}
        onSearchClick={() => handleSearchClick("guest")}
      />
    );
  }

  if (mode === "search") {
    return (
      <SearchScreen 
        onBack={() => setMode(searchMode)}
        searchMode={searchMode}
      />
    );
  }

  return null;
}

export default App;