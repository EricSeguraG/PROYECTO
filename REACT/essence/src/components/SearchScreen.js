import React, { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useSearch } from '../hooks/useSearch';
import { useAuth } from '../hooks/useAuth';

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
  
  const { 
    searchResults, 
    loading, 
    error, 
    searchPerfumes, 
    clearResults 
  } = useSearch();

  const auth = useAuth();

  const handleSearch = (e) => {
    e?.preventDefault();
    
    console.log('🎯 Iniciando búsqueda...');
    console.log('📝 Query:', searchQuery);
    console.log('🔧 Filtros:', filters);
    
    // Crear objeto de filtros combinando búsqueda principal y filtros
    const searchFilters = { ...filters };
    
    // Si hay búsqueda en el input principal, usarla como filtro de perfume
    if (searchQuery.trim()) {
      searchFilters.perfume = searchQuery;
    }
    
    // Llamar a la función de búsqueda
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
  };

  const handleViewDetails = (perfume) => {
    alert(`Detalles de ${perfume.perfume}\nMarca: ${perfume.marca}\nGénero: ${perfume.genero}\nPerfumista: ${perfume.perfumista || 'N/A'}`);
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

  // Obtener el nombre del usuario si está en modo user
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
        {/* Header con las clases CSS definidas en App.css */}
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
          <div className="card" style={{ maxWidth: '800px', margin: '0 auto', width: '95%' }}>
            
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

          {/* Resultados */}
          <div style={{ maxWidth: '800px', margin: '1rem auto 0', width: '95%' }}>
            {searchResults.length > 0 && (
              <div className="card">
                <h3 style={{ color: '#F3E5AB', marginBottom: '1rem', fontSize: '1.1rem' }}>
                  {searchResults.length} perfumes encontrados
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  {searchResults.map((perfume, index) => (
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
                      <h4 style={{ 
                        margin: '0 0 0.5rem 0', 
                        color: '#F3E5AB',
                        fontSize: '1.1rem'
                      }}>
                        {perfume.perfume || 'Nombre no disponible'}
                      </h4>
                      
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

                      {/* Botón de acción - Solo Ver Detalles */}
                      <div style={{ 
                        display: 'flex', 
                        gap: '0.5rem', 
                        marginTop: '0.8rem',
                        justifyContent: 'flex-end'
                      }}>
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
                          Ver Detalles
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && searchResults.length === 0 && (filters.perfume || filters.marca || filters.genero || filters.nota || filters.acorde || filters.perfumista) && (
              <div className="card" style={{ textAlign: 'center', padding: '1.5rem' }}>
                <p style={{ color: '#F3E5AB', marginBottom: '1rem' }}>
                  No se encontraron perfumes que coincidan con tu búsqueda
                </p>
                <button className="link" onClick={handleClear}>
                  Limpiar búsqueda
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchScreen;