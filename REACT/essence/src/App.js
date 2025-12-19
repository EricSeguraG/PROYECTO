// src/App.js
import React, { useState, useEffect } from "react";
import { useAuth } from "./hooks/useAuth";

// Componentes
import StartScreen from "./components/StartScreen";
import LoginScreen from "./components/LoginScreen";
import UserScreen from "./components/UserScreen";
import GuestScreen from "./components/GuestScreen";
import SearchScreen from "./components/SearchScreen";
import ClonesScreen from "./components/ClonesScreen";
import CelebrityScreen from "./components/CelebrityScreen";
import BrandsScreen from "./components/BrandsScreen";
import PerfumesByBrandScreen from "./components/PerfumesByBrandScreen";
import TopRatedScreen from "./components/TopRatedScreen"; // <-- Nuevo componente
import "./App.css";

function App() {
  // Estados de navegación
  const [mode, setMode] = useState("start"); // start | login | user | guest | search | clones | celebrity | brands | perfumes-by-brand | top-rated
  const [searchMode, setSearchMode] = useState("user");
  const [selectedBrand, setSelectedBrand] = useState("");
  
  // Hook de autenticación
  const auth = useAuth();

  useEffect(() => {
    if (auth.user) {
      setMode("user");
    }
  }, [auth.user]);

  const handleLogout = () => {
    auth.handleLogout();
    setMode("start");
  };

  // Funciones de navegación existentes
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

  // NUEVA FUNCIÓN para "Los más votados"
  const handleTopRatedClick = (fromMode) => {
    setSearchMode(fromMode);
    setMode("top-rated");
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
        onUserClick={() => {
          if (auth.user) {
            setMode("user");
          } else {
            setMode("login");
          }
        }}
        onGuestClick={() => setMode("guest")}
      />
    );
  }

  if (mode === "login") {
    return (
      <LoginScreen
        {...auth}
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
        onTopRatedClick={() => handleTopRatedClick("user")} // <-- Nuevo prop
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
        onTopRatedClick={() => handleTopRatedClick("guest")} // <-- Nuevo prop
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

  // NUEVA PANTALLA: Los más votados
  if (mode === "top-rated") {
    return (
      <TopRatedScreen 
        onBack={() => setMode(searchMode)}
        searchMode={searchMode}
      />
    );
  }

  return null;
}

export default App;