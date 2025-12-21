import React, { useState } from 'react';
import { Star, ArrowLeft, Search } from 'lucide-react';
import { useSearch } from '../hooks/useSearch';
import { useAuth } from '../hooks/useAuth';

const CelebrityScreen = ({ onBack, searchMode }) => {
  const [query, setQuery] = useState('');
  const { searchResults, loading, loadingImages, error, searchByCelebrity } = useSearch();
  const auth = useAuth();

  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) {
      searchByCelebrity(query);
    }
  };

  const handleClear = () => {
    setQuery('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && query.trim()) {
      handleSearch(e);
    }
  };

  // --- ESTILOS ---
  const containerStyle = {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    overflow: 'hidden'
  };

  const videoStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    zIndex: 0
  };

  const overlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'rgba(0, 0, 0, 0.4)',
    zIndex: 1
  };

  const contentStyle = {
    position: 'relative',
    zIndex: 2,
    flex: 1,
    display: 'flex',
    flexDirection: 'column'
  };

  const getUserDisplayName = () => {
    if (searchMode === 'user' && auth.user) {
      return `${auth.user.name || 'Usuario'} ${auth.user.lastname || ''}`.trim();
    }
    return 'Usuario';
  };

  return (
    <div style={containerStyle}>
      <video autoPlay muted loop playsInline style={videoStyle}>
        <source src="/videos/vid.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      
      <div style={overlayStyle}></div>

      <div style={contentStyle}>
        {/* Header */}
        <header className="header">
          <div className="header-left">
            <h1 className="logo">ESSENCE</h1>
            <span className="user-mode-label">
              {searchMode === 'user' 
                ? `${getUserDisplayName()}` 
                : '[GUEST MODE]'
              }
            </span>
          </div>
          
          <button className="exit-btn" onClick={onBack}>
            <ArrowLeft size={16} /> Volver
          </button>
        </header>

        {/* Contenido principal */}
        <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
          <div className="card" style={{ maxWidth: '900px', margin: '0 auto', width: '95%' }}>
            
            {/* Título */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
              <div className="logo" style={{
                fontSize: '1.8rem',
                letterSpacing: '2px',
                fontWeight: '700',
                fontFamily: '"Orbitron", sans-serif',
                background: 'rgb(243,229,171)',
                padding: '12px 25px',
                borderRadius: '12px',
                border: '2px solid #713600',
                boxShadow: '0 0 20px rgba(243,229,171), inset 0 0 15px rgba(255, 215, 0, 0.2)',
                color: '#713600',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <Star size={24} style={{ color: '#713600' }} />
                CELEBRITY MATCH
              </div>
            </div>

            <p className="subtitle" style={{ 
              fontSize: '1rem', 
              marginBottom: '1.5rem',
              textAlign: 'center',
              color: '#F3E5AB'
            }}>
              Descubre qué perfume usan tus ídolos
            </p>

            {/* Buscador */}
            <div style={{ 
              display: 'flex', 
              gap: '0.8rem', 
              marginBottom: '2rem',
              alignItems: 'center'
            }}>
              <div style={{ position: 'relative', flex: 1 }}>
                <input
                  type="text"
                  placeholder="Ej: Rihanna, Brad Pitt, Messi..."
                  className="input"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  style={{ 
                    paddingLeft: '2.5rem',
                    width: '100%',
                    fontSize: '0.95rem'
                  }}
                  autoFocus
                />
                <Search 
                  size={20} 
                  style={{ 
                    position: 'absolute', 
                    left: '0.8rem', 
                    top: '50%', 
                    transform: 'translateY(-50%)',
                    color: '#F3E5AB'
                  }} 
                />
              </div>
              
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button 
                  type="button" 
                  className="btn" 
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  style={{ 
                    width: '120px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}
                >
                  {loading ? 'Buscando...' : 'Buscar'}
                </button>
                
                <button 
                  type="button" 
                  className="btn" 
                  onClick={handleClear}
                  disabled={loading}
                  style={{ 
                    width: '120px',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  Limpiar
                </button>
              </div>
            </div>

            {/* Mensaje de Error */}
            {error && (
              <div style={{ 
                textAlign: 'center', 
                padding: '1rem', 
                background: 'rgba(255, 68, 68, 0.1)', 
                borderRadius: '0.8rem', 
                marginBottom: '1.5rem',
                border: '1px solid #ff4444'
              }}>
                <p style={{ color: '#ff4444', margin: 0, fontSize: '0.9rem' }}>
                  ❌ {error}
                </p>
              </div>
            )}

            {/* Resultados */}
            <div style={{ marginTop: '1.5rem' }}>
              {searchResults.length > 0 ? (
                searchResults.map((item, index) => (
                  <div 
                    key={index} 
                    style={{ 
                      marginBottom: '2rem', 
                      background: 'rgba(113, 54, 0, 0.8)', 
                      padding: '1.5rem', 
                      borderRadius: '0.8rem',
                      border: '2px solid #F3E5AB'
                    }}
                  >
                    <h3 style={{ 
                      color: '#F3E5AB', 
                      fontSize: '1.4rem', 
                      marginBottom: '1rem', 
                      borderBottom: '2px solid #F3E5AB', 
                      paddingBottom: '0.5rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem'
                    }}>
                      <Star size={20} style={{ color: 'gold' }} />
                      {item.celebrity}
                    </h3>
                    
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', 
                      gap: '1.5rem' 
                    }}>
                      {item.perfumes.map((perfume, perfumeIndex) => (
                        <div 
                          key={perfumeIndex} 
                          style={{ 
                            textAlign: 'center',
                            background: 'rgba(243, 229, 171, 0.1)',
                            padding: '1rem',
                            borderRadius: '0.8rem',
                            border: '1px solid #F3E5AB'
                          }}
                        >
                          <div style={{ 
                            height: '140px', 
                            background: 'rgba(243, 229, 171, 0.9)', 
                            marginBottom: '0.8rem', 
                            borderRadius: '0.5rem', 
                            overflow: 'hidden',
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'center',
                            padding: '10px'
                          }}>
                            {loadingImages ? (
                              <div style={{ 
                                display: 'flex', 
                                flexDirection: 'column',
                                alignItems: 'center', 
                                justifyContent: 'center',
                                width: '100%',
                                height: '100%',
                                color: '#713600',
                                fontSize: '0.8rem',
                                textAlign: 'center'
                              }}>
                                <div style={{ 
                                  width: '30px', 
                                  height: '30px', 
                                  border: '3px solid rgba(113, 54, 0, 0.3)',
                                  borderTop: '3px solid #713600',
                                  borderRadius: '50%',
                                  animation: 'spin 1s linear infinite',
                                  marginBottom: '0.5rem'
                                }}></div>
                                Cargando...
                              </div>
                            ) : perfume.img ? (
                              <img 
                                src={perfume.img} 
                                alt={perfume.name} 
                                style={{ 
                                  maxHeight: '100%', 
                                  maxWidth: '100%', 
                                  objectFit: 'contain' 
                                }} 
                                onError={(e) => {
                                  e.target.onerror = null;
                                  e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%23713600'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='Arial' font-size='12' fill='%23F3E5AB'%3ENo Image%3C/text%3E%3C/svg%3E";
                                }}
                              />
                            ) : (
                              <div style={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                justifyContent: 'center',
                                width: '100%',
                                height: '100%',
                                color: '#713600',
                                fontSize: '0.8rem',
                                textAlign: 'center',
                                padding: '1rem'
                              }}>
                                Imagen no disponible
                              </div>
                            )}
                          </div>
                          <h4 style={{ 
                            color: '#F3E5AB', 
                            fontSize: '0.95rem', 
                            margin: '0.3rem 0',
                            fontWeight: '600'
                          }}>
                            {perfume.name}
                          </h4>
                          <p style={{ 
                            color: '#F3E5AB', 
                            fontSize: '0.8rem', 
                            margin: 0,
                            opacity: 0.9
                          }}>
                            {perfume.brand}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                !loading && !loadingImages && !error && (
                  <div style={{ 
                    textAlign: 'center', 
                    padding: '2rem',
                    color: '#F3E5AB'
                  }}>
                    <p style={{ fontSize: '1rem', margin: 0 }}>
                      🔍 Ingresa el nombre de una celebridad para descubrir sus perfumes
                    </p>
                  </div>
                )
              )}
            </div>

            {/* Mensaje de carga */}
            {loading && (
              <div style={{ 
                textAlign: 'center', 
                padding: '2rem',
                color: '#F3E5AB'
              }}>
                <p style={{ fontSize: '1rem', margin: 0 }}>
                  🔍 Buscando información de la celebridad...
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Estilos para la animación de spinner */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default CelebrityScreen;