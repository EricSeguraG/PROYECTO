// src/components/SearchScreen.js
import React, { useState, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useSearch } from '../hooks/useSearch';
import { useAuth } from '../hooks/useAuth';
import PerfumeDetailModal from './PerfumeDetailModal';

const SearchScreen = ({ onBack, searchMode }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({
    perfume: '',
    marca: '',
    genero: '',
    nota: '',
    acorde: '',
    perfumista: ''
  });
  
  const [selectedPerfume, setSelectedPerfume] = useState(null);
  
  const { 
    searchResults, 
    loading, 
    error, 
    searchPerfumes, 
    clearResults 
  } = useSearch();

  const auth = useAuth();
  const gridRef = useRef(null);

  // Función para obtener la imagen del perfume
  const getPerfumeImage = (url) => {
    if (!url) return null;

    const match = url.match(/(\d+)\.html$/);
    if (!match) return null;

    const id = match[1];
    return `https://fimgs.net/mdimg/perfume/375x500.${id}.jpg`;
  };

  const handleSearch = (e) => {
    e?.preventDefault();
    
    console.log('🎯 Iniciando búsqueda...');
    console.log('📝 Query:', searchQuery);
    console.log('🔧 Filtros:', filters);
    
    const searchFilters = { ...filters };
    
    if (searchQuery.trim()) {
      searchFilters.perfume = searchQuery;
    }
    
    searchPerfumes(searchQuery, searchFilters);
  };

  const handleClear = () => {
    setSearchQuery('');
    setFilters({
      perfume: '',
      marca: '',
      genero: '',
      nota: '',
      acorde: '',
      perfumista: ''
    });
    clearResults();
    setSelectedPerfume(null);
  };

  const handleViewDetails = (perfume) => {
    setSelectedPerfume(perfume);
  };

  // --- ESTILOS VISUALES ---
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

  // Grid de 2 columnas con scroll - IDÉNTICO a PerfumesByBrandScreen
  const gridContainerStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1.2rem',
    marginTop: '1rem',
    maxHeight: 'calc(280px * 5 + 1.2rem * 4)',
    overflowY: 'auto',
    paddingRight: '0.5rem',
    paddingBottom: '1rem'
  };

  // Estilos CSS para la scrollbar - IDÉNTICO
  const scrollbarStyles = `
    .search-results-grid::-webkit-scrollbar {
      width: 8px;
    }
    
    .search-results-grid::-webkit-scrollbar-track {
      background: rgba(243, 229, 171, 0.1);
      border-radius: 4px;
    }
    
    .search-results-grid::-webkit-scrollbar-thumb {
      background: rgba(243, 229, 171, 0.6);
      border-radius: 4px;
    }
    
    .search-results-grid::-webkit-scrollbar-thumb:hover {
      background: rgba(243, 229, 171, 0.8);
    }

    /* Responsive - IDÉNTICO */
    @media (max-width: 768px) {
      .search-results-grid {
        grid-template-columns: 1fr !important;
      }
    }
  `;

  const getUserDisplayName = () => {
    if (searchMode === 'user' && auth.user) {
      return `${auth.user.name || 'Usuario'} ${auth.user.lastname || ''}`.trim();
    }
    return 'Usuario';
  };

  return (
    <div style={containerStyle}>
      <style>{scrollbarStyles}</style>
      
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
          
          <button className="exit-btn" onClick={onBack} style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <ArrowLeft size={16} /> Volver
          </button>
        </header>

        {/* Contenido principal */}
        <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
          <div className="card" style={{ maxWidth: '1100px', margin: '0 auto', width: '95%' }}>
            
            {/* Recuadro MULTIBUSCADOR arriba del buscador */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'center', 
              marginBottom: '1.5rem' 
            }}>
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
                boxShadow: 
                  '0 0 20px rgba(243,229,171), inset 0 0 15px rgba(255, 215, 0, 0.2)',
                color: '#713600',
                animation: 'none',
                WebkitFontSmoothing: 'antialiased',
                MozOsxFontSmoothing: 'grayscale',
                textRendering: 'optimizeLegibility',
                transition: 'all 0.3s ease'
              }}>
                MULTIBUSCADOR
              </div>
            </div>

            {/* Filtros siempre visibles - 3 arriba y 3 abajo */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)', 
              gap: '0.8rem',
              marginBottom: '1.5rem',
              padding: '1rem',
              background: 'rgba(113, 54, 0, 0.8)',
              borderRadius: '0.8rem',
              border: '1px solid #F3E5AB'
            }}>
              {/* Fila 1 - 3 filtros */}
              <div>
                <label style={{ color: '#F3E5AB', fontSize: '0.85rem', marginBottom: '0.3rem', display: 'block' }}>
                  Nombre
                </label>
                <input
                  type="text"
                  placeholder="Ej: Sauvage"
                  value={filters.perfume}
                  onChange={(e) => setFilters({...filters, perfume: e.target.value})}
                  className="input"
                  style={{ fontSize: '0.85rem', padding: '0.4rem 0.6rem', width: '100%' }}
                />
              </div>

              <div>
                <label style={{ color: '#F3E5AB', fontSize: '0.85rem', marginBottom: '0.3rem', display: 'block' }}>
                  Marca
                </label>
                <input
                  type="text"
                  placeholder="Ej: Dior"
                  value={filters.marca}
                  onChange={(e) => setFilters({...filters, marca: e.target.value})}
                  className="input"
                  style={{ fontSize: '0.85rem', padding: '0.4rem 0.6rem', width: '100%' }}
                />
              </div>

              <div>
                <label style={{ color: '#F3E5AB', fontSize: '0.85rem', marginBottom: '0.3rem', display: 'block' }}>
                  Género
                </label>
                <select 
                  value={filters.genero}
                  onChange={(e) => setFilters({...filters, genero: e.target.value})}
                  className="input"
                  style={{ 
                    fontSize: '0.85rem', 
                    padding: '0.4rem 0.6rem', 
                    width: '100%',
                    color: '#000000',
                    backgroundColor: '#F3E5AB',
                    border: '1px solid #713600',
                    borderRadius: '4px'
                  }}
                >
                  <option value="">Todos</option>
                  <option value="Masculino">Masculino</option>
                  <option value="Femenino">Femenino</option>
                  <option value="Unisex">Unisex</option>
                </select>
              </div>

              {/* Fila 2 - 3 filtros */}
              <div>
                <label style={{ color: '#F3E5AB', fontSize: '0.85rem', marginBottom: '0.3rem', display: 'block' }}>
                  Notas
                </label>
                <input
                  type="text"
                  placeholder="Ej: vainilla, ámbar"
                  value={filters.nota}
                  onChange={(e) => setFilters({...filters, nota: e.target.value})}
                  className="input"
                  style={{ fontSize: '0.85rem', padding: '0.4rem 0.6rem', width: '100%' }}
                />
              </div>

              <div>
                <label style={{ color: '#F3E5AB', fontSize: '0.85rem', marginBottom: '0.3rem', display: 'block' }}>
                  Acordes
                </label>
                <input
                  type="text"
                  placeholder="Ej: amaderado, especiado"
                  value={filters.acorde}
                  onChange={(e) => setFilters({...filters, acorde: e.target.value})}
                  className="input"
                  style={{ fontSize: '0.85rem', padding: '0.4rem 0.6rem', width: '100%' }}
                />
              </div>

              <div>
                <label style={{ color: '#F3E5AB', fontSize: '0.85rem', marginBottom: '0.3rem', display: 'block' }}>
                  Perfumista
                </label>
                <input
                  type="text"
                  placeholder="Ej: F. Demachy"
                  value={filters.perfumista}
                  onChange={(e) => setFilters({...filters, perfumista: e.target.value})}
                  className="input"
                  style={{ fontSize: '0.85rem', padding: '0.4rem 0.6rem', width: '100%' }}
                />
              </div>
            </div>

            {/* Botones Buscar y Limpiar */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'center', 
              gap: '1rem',
              marginBottom: '1rem' 
            }}>
              <button 
                type="button" 
                className="btn" 
                onClick={handleSearch}
                disabled={loading}
                style={{ 
                  width: '140px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
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
                  width: '140px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}
              >
                Limpiar
              </button>
            </div>

            {/* Información sobre los filtros activos */}
            {(filters.perfume || filters.marca || filters.genero || filters.nota || filters.acorde || filters.perfumista) && (
              <div style={{ 
                marginBottom: '1rem',
                padding: '0.6rem',
                background: 'rgba(243, 229, 171, 0.1)',
                borderRadius: '0.5rem',
                border: '1px solid #F3E5AB'
              }}>
                <p style={{ color: '#F3E5AB', fontSize: '0.8rem', margin: 0 }}>
                  <strong>Filtros activos:</strong> 
                  {filters.perfume && ` Perfume: "${filters.perfume}"`}
                  {filters.marca && ` Marca: "${filters.marca}"`}
                  {filters.genero && ` Género: "${filters.genero}"`}
                  {filters.nota && ` Notas: "${filters.nota}"`}
                  {filters.acorde && ` Acordes: "${filters.acorde}"`}
                  {filters.perfumista && ` Perfumista: "${filters.perfumista}"`}
                </p>
              </div>
            )}

            {/* Mensajes de estado */}
            {loading && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginBottom: '1rem', fontSize: '0.9rem' }}>
                🔍 Buscando en la base de datos de perfumes...
              </p>
            )}
            {error && (
              <p style={{ color: '#ff4444', textAlign: 'center', marginBottom: '1rem', fontSize: '0.9rem' }}>
                ❌ Error: {error}
              </p>
            )}
          </div>

          {/* Resultados en Grid de 2 columnas - CON IMÁGENES */}
          {searchResults.length > 0 && (
            <div className="card" style={{ 
              maxWidth: '1100px', 
              margin: '1rem auto 0', 
              width: '95%',
              padding: '1.5rem'
            }}>
              <h3 style={{ 
                color: '#F3E5AB', 
                marginBottom: '1.5rem', 
                fontSize: '1.2rem',
                textAlign: 'center'
              }}>
                 {searchResults.length} perfumes encontrados
              </h3>
              
              <div 
                className="search-results-grid"
                style={gridContainerStyle}
                ref={gridRef}
              >
                {searchResults.map((perfume, index) => (
                  <div 
                    key={`${perfume.id || index}`}
                    style={{
                      padding: '1.2rem',
                      background: 'rgba(0, 0, 0, 0.9)',
                      borderRadius: '0.8rem',
                      border: '2px solid #F3E5AB',
                      color: '#F3E5AB',
                      display: 'flex',
                      flexDirection: 'column',
                      minHeight: '280px'
                    }}
                  >
                    {/* Nombre del perfume */}
                    <h4 style={{ 
                      margin: '0 0 1rem 0', 
                      color: '#F3E5AB', 
                      fontSize: '1.1rem',
                      textAlign: 'center',
                      lineHeight: '1.3'
                    }}>
                      {perfume.perfume || 'Nombre no disponible'}
                    </h4>
                    
                    {/* Imagen del perfume - AÑADIDO */}
                    {getPerfumeImage(perfume.url) && (
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'center', 
                        marginBottom: '0.8rem' 
                      }}>
                        <img
                          src={getPerfumeImage(perfume.url)}
                          alt={perfume.perfume || 'Perfume'}
                          loading="lazy"
                          style={{
                            width: '90px',
                            height: '120px',
                            objectFit: 'contain',
                            borderRadius: '0.4rem',
                            background: 'rgba(243, 229, 171, 0.15)',
                            padding: '0.3rem'
                          }}
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                    )}
                    
                    {/* Información básica en grid */}
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: '1fr 1fr', 
                      gap: '0.8rem', 
                      fontSize: '0.9rem',
                      marginBottom: '1rem',
                      flex: 1
                    }}>
                      <div><strong>Marca:</strong> {perfume.marca || 'N/A'}</div>
                      <div><strong>Género:</strong> {perfume.genero || 'N/A'}</div>
                      <div><strong>Año:</strong> {perfume.año || 'N/A'}</div>
                      <div><strong>Perfumista:</strong> {perfume.perfumista || 'N/A'}</div>
                    </div>

                    {/* Notas */}
                    {(perfume.salida || perfume.corazon || perfume.base) && (
                      <div style={{ 
                        marginBottom: '1rem', 
                        fontSize: '0.85rem',
                        background: 'rgba(243, 229, 171, 0.1)',
                        padding: '0.8rem',
                        borderRadius: '0.4rem'
                      }}>
                        <strong style={{ display: 'block', marginBottom: '0.3rem' }}>Notas:</strong>
                        {perfume.salida && <div style={{ fontSize: '0.8rem' }}>• Salida: {perfume.salida}</div>}
                        {perfume.corazon && <div style={{ fontSize: '0.8rem' }}>• Corazón: {perfume.corazon}</div>}
                        {perfume.base && <div style={{ fontSize: '0.8rem' }}>• Base: {perfume.base}</div>}
                      </div>
                    )}

                    {/* Acordes principales */}
                    {perfume.main_accords && perfume.main_accords.length > 0 && (
                      <div style={{ 
                        marginBottom: '1rem', 
                        fontSize: '0.85rem',
                        background: 'rgba(243, 229, 171, 0.1)',
                        padding: '0.8rem',
                        borderRadius: '0.4rem'
                      }}>
                        <strong style={{ display: 'block', marginBottom: '0.3rem' }}>Acordes:</strong>
                        <div style={{ fontSize: '0.8rem' }}>
                          {Array.isArray(perfume.main_accords) 
                            ? perfume.main_accords.join(', ')
                            : perfume.main_accords}
                        </div>
                      </div>
                    )}

                    {/* Botón de acción */}
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'flex-end', 
                      marginTop: 'auto'
                    }}>
                      <button 
                        className="btn"
                        style={{ 
                          padding: '0.5rem 1rem', 
                          fontSize: '0.85rem',
                          background: 'rgba(243, 229, 171, 0.9)',
                          color: '#713600',
                          border: '1px solid #F3E5AB',
                          fontWeight: 'bold',
                          cursor: 'pointer'
                        }}
                        onClick={() => handleViewDetails(perfume)}
                      >
                        Ver Detalles y Comentar
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Indicador de scroll si hay muchos resultados */}
              {searchResults.length > 10 && (
                <div style={{ 
                  textAlign: 'center', 
                  marginTop: '1rem',
                  color: 'rgba(243, 229, 171, 0.7)',
                  fontSize: '0.85rem'
                }}>
                  <span style={{ display: 'inline-block', animation: 'bounce 2s infinite' }}>
                    ⬇️ Desplázate para ver más perfumes
                  </span>
                  <style>{`
                    @keyframes bounce {
                      0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
                      40% {transform: translateY(-5px);}
                      60% {transform: translateY(-3px);}
                    }
                  `}</style>
                </div>
              )}
            </div>
          )}

          {/* Mensaje cuando no hay resultados */}
          {!loading && searchResults.length === 0 && (filters.perfume || filters.marca || filters.genero || filters.nota || filters.acorde || filters.perfumista) && (
            <div className="card" style={{ 
              maxWidth: '1100px', 
              margin: '1rem auto 0', 
              width: '95%',
              textAlign: 'center', 
              padding: '3rem' 
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔍</div>
              <p style={{ color: '#F3E5AB', marginBottom: '1.5rem', fontSize: '1.1rem' }}>
                No se encontraron perfumes que coincidan con tu búsqueda
              </p>
              <button 
                className="btn" 
                onClick={handleClear}
                style={{ 
                  padding: '0.7rem 2rem',
                  background: 'rgba(243, 229, 171, 0.9)',
                  color: '#713600'
                }}
              >
                Limpiar búsqueda
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Modal de Comentarios */}
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

export default SearchScreen;