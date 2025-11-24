// src/App.js
import React, { useState, useEffect } from "react";
import { useAuth } from "./hooks/useAuth";
import StartScreen from "./components/StartScreen";
import LoginScreen from "./components/LoginScreen";
import UserScreen from "./components/UserScreen";
import GuestScreen from "./components/GuestScreen";
import SearchScreen from "./components/SearchScreen";
import ClonesScreen from "./components/ClonesScreen";
import CelebrityScreen from "./components/CelebrityScreen";
import BrandsScreen from "./components/BrandsScreen";
import PerfumesByBrandScreen from "./components/PerfumesByBrandScreen";
import "./App.css";

function App() {
  const [mode, setMode] = useState("start"); // start | login | user | guest | search | clones | celebrity | brands | perfumes-by-brand
  const [searchMode, setSearchMode] = useState("user");
  const [selectedBrand, setSelectedBrand] = useState("");
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
  }, []); 

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

  const handleClonesClick = (fromMode) => {
    setSearchMode(fromMode);
    setMode("clones");
  };

  const handleCelebrityClick = (fromMode) => {
    setSearchMode(fromMode);
    setMode("celebrity");
  };

  const handleBrandsClick = (fromMode) => {
    setSearchMode(fromMode);
    setMode("brands");
  };

  const handleBrandSelect = (brandName) => {
    setSelectedBrand(brandName);
    setMode("perfumes-by-brand");
  };

  const handleBackFromPerfumes = () => {
    setMode("brands");
  };

  // --- RENDERIZADO DE PANTALLAS ---

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
        onClonesClick={() => handleClonesClick("user")}
        onCelebrityClick={() => handleCelebrityClick("user")}
        onPerfumesClick={() => handleBrandsClick("user")}
      />
    );
  }

  if (mode === "guest") {
    return (
      <GuestScreen 
        onExit={() => setMode("start")}
        onSearchClick={() => handleSearchClick("guest")}
        onClonesClick={() => handleClonesClick("guest")}
        onCelebrityClick={() => handleCelebrityClick("guest")}
        onPerfumesClick={() => handleBrandsClick("guest")}
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

  if (mode === "clones") {
    return (
      <ClonesScreen 
        onBack={() => setMode(searchMode)}
        searchMode={searchMode}
      />
    );
  }

  if (mode === "celebrity") {
    return (
      <CelebrityScreen 
        onBack={() => setMode(searchMode)}
        searchMode={searchMode}
      />
    );
  }

  if (mode === "brands") {
    return (
      <BrandsScreen 
        onBack={() => setMode(searchMode)}
        onBrandSelect={handleBrandSelect}
        searchMode={searchMode}
      />
    );
  }

  if (mode === "perfumes-by-brand") {
    return (
      <PerfumesByBrandScreen 
        onBack={handleBackFromPerfumes}
        brandName={selectedBrand}
        searchMode={searchMode}
      />
    );
  }

  return null;
}

export default App;