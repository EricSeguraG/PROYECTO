// src/components/ClonesScreen.js
import React, { useState, useEffect } from 'react';
import { Sparkles, ArrowLeft, Search, X } from 'lucide-react';
import { useClones } from '../hooks/useClones';
import { useAuth } from '../hooks/useAuth';
import PerfumeDetailModal from './PerfumeDetailModal'; // <--- 1. Importamos el Modal

const ClonesScreen = ({ onBack, searchMode }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [similarityThreshold, setSimilarityThreshold] = useState(70);
  
  // 2. Estado para el Modal
  const [selectedPerfume, setSelectedPerfume] = useState(null);
  
  const { 
    clonesResults, 
    loading, 
    error, 
    selectedPerfume: originalPerfume,
    searchClones, 
    clearResults 
  } = useClones();

  const auth = useAuth();

  // Calcular el porcentaje para el gradiente
  const calculateProgress = () => {
    const min = 50;
    const max = 95;
    return ((similarityThreshold - min) / (max - min)) * 100;
  };

  const progress = calculateProgress();

  // Aplicar estilos al input range dinámicamente
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .custom-range {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        border-radius: 4px;
        outline: none;
        background: linear-gradient(to right, #ff9900ff ${progress}%, #ffffff ${progress}%);
      }
      .custom-range::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #ff9900ff;
        border: 2px solid #000000ff;
        cursor: pointer;
        box-shadow: 0 0 5px rgba(0,0,0,0.3);
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, [progress]);

  const handleSearch = (e) => {
    e.preventDefault();
    searchClones(searchQuery, similarityThreshold);
  };

  const handleClear = () => {
    setSearchQuery('');
    clearResults();
    setSelectedPerfume(null); // 3. Limpiar perfume seleccionado
  };

  // 4. Modificamos esta función para abrir el Modal
  const handleViewDetails = (perfume) => {
    setSelectedPerfume(perfume);
  };

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
          <div className="card" style={{ maxWidth: '800px', margin: '0 auto', width: '95%' }}>
            
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
                display: 'inline-block',
                border: '2px solid #713600',
                boxShadow: '0 0 20px rgba(243,229,171), inset 0 0 15px rgba(255, 215, 0, 0.2)',
                color: '#713600'
              }}>
                CLONES/INSPIRACIONES
              </div>
            </div>

            {/* Barra de búsqueda */}
            <div style={{ position: 'relative', marginBottom: '1.5rem' }}>
              <input
                type="text"
                placeholder="Ingresa el nombre del perfume a comparar..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input"
                style={{ paddingLeft: '2.5rem', width: '100%' }}
              />
              <Search size={20} style={{ 
                position: 'absolute', 
                left: '0.8rem', 
                top: '50%', 
                transform: 'translateY(-50%)',
                color: '#ff9900ff'
              }} />
              {searchQuery && (
                <button
                  type="button"
                  onClick={handleClear}
                  style={{
                    position: 'absolute',
                    right: '0.8rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: '#F3E5AB',
                    cursor: 'pointer'
                  }}
                >
                  <X size={20} />
                </button>
              )}
            </div>

            {/* Filtro de similitud */}
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(113, 54, 0, 0.8)', borderRadius: '0.8rem', border: '1px solid #F3E5AB' }}>
              <label style={{ color: '#ffffffff', fontSize: '0.9rem', marginBottom: '0.5rem', display: 'block' }}>
                Porcentaje mínimo de similitud: <strong>{similarityThreshold}</strong>
              </label>
              <input
                type="range"
                min="50"
                max="95"
                step="5"
                value={similarityThreshold}
                onChange={(e) => setSimilarityThreshold(parseInt(e.target.value))}
                className="custom-range"
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#F3E5AB', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                <span>50</span>
                <span>70</span>
                <span>95</span>
              </div> 
            </div>

            {/* Botón Buscar */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
              <button 
                className="btn" 
                onClick={handleSearch}
                disabled={loading || !searchQuery.trim()}
                style={{ width: '200px' }}
              >
                {loading ? 'Buscando...' : 'Buscar Clones'}
              </button>
            </div>

            {/* Mensajes de estado */}
            {loading && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginBottom: '1rem' }}>
                🔍 Buscando clones e inspiraciones...
              </p>
            )}
            {error && (
              <p style={{ color: '#ff4444', textAlign: 'center', marginBottom: '1rem' }}>
                ❌ FAIL
              </p>
            )}
            {originalPerfume && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginBottom: '1rem' }}>
                Mostrando clones e inspiraciones de: <strong>{originalPerfume}</strong>
              </p>
            )}
          </div>

          {/* Resultados */}
          <div style={{ maxWidth: '800px', margin: '1rem auto 0', width: '95%' }}>
            {clonesResults.length > 0 && (
              <div className="card">
                <h3 style={{ color: '#F3E5AB', marginBottom: '1rem' }}>
                  {clonesResults.length} clones/inspiraciones encontrados
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  {clonesResults.map((perfume, index) => (
                    <div 
                      key={index}
                      style={{
                        padding: '1rem',
                        background: 'rgba(113, 54, 0, 0.9)',
                        borderRadius: '0.8rem',
                        border: '2px solid #F3E5AB',
                        color: '#F3E5AB',
                        position: 'relative'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                        <h4 style={{ margin: 0, color: '#F3E5AB', fontSize: '1.1rem' }}>
                          {perfume.perfume || 'Nombre no disponible'}
                        </h4>
                        <span style={{ 
                          background: 'rgba(243, 229, 171, 0.9)', 
                          color: '#713600',
                          padding: '0.2rem 0.6rem',
                          borderRadius: '1rem',
                          fontSize: '0.8rem',
                          fontWeight: 'bold'
                        }}>
                          {perfume.similitud || '0'} de similitud
                        </span>
                      </div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.9rem' }}>
                        <div>
                          <strong>Marca:</strong> {perfume.marca || 'N/A'}
                        </div>
                        <div>
                          <strong>Género:</strong> {perfume.genero || 'N/A'}
                        </div>
                        <div>
                          <strong>Perfumista:</strong> {perfume.perfumista || 'N/A'}
                        </div>
                        <div>
                          <strong>Año:</strong> {perfume.año || 'N/A'}
                        </div>
                      </div>

                      {/* Notas */}
                      {(perfume.salida || perfume.corazon || perfume.base) && (
                        <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                          <strong>Notas:</strong>
                          {perfume.salida && <div>• Salida: {perfume.salida}</div>}
                          {perfume.corazon && <div>• Corazón: {perfume.corazon}</div>}
                          {perfume.base && <div>• Base: {perfume.base}</div>}
                        </div>
                      )}

                      {/* Acordes principales */}
                      {perfume.main_accords && perfume.main_accords.length > 0 && (
                        <div style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                          <strong>Acordes:</strong> {Array.isArray(perfume.main_accords) ? perfume.main_accords.join(', ') : perfume.main_accords}
                        </div>
                      )}

                      {/* Botones de acción */}
                      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.8rem', justifyContent: 'flex-end' }}>
                        <button 
                          className="btn"
                          style={{ 
                            padding: '0.3rem 0.6rem', 
                            fontSize: '0.8rem',
                            background: 'rgba(243, 229, 171, 0.9)',
                            color: '#713600'
                          }}
                          onClick={() => handleViewDetails(perfume)}
                        >
                          Ver Detalles y Comentar
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && clonesResults.length === 0 && originalPerfume && (
              <div className="card" style={{ textAlign: 'center' }}>
                <p style={{ color: '#F3E5AB' }}>
                  No se encontraron clones o inspiraciones con {similarityThreshold} de similitud
                </p>
                <button className="link" onClick={handleClear}>
                  Nueva búsqueda
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 5. AÑADIDO: El Modal de Comentarios */}
      {selectedPerfume && (
        <PerfumeDetailModal 
          perfume={selectedPerfume}
          user={auth.user} 
          onClose={() => setSelectedPerfume(null)}
        />
      )}
    </div>
  );
};

export default ClonesScreen;