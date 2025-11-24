// src/components/BrandsScreen.js
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Search } from 'lucide-react';
import { perfumeAPI } from '../services/api';
import { useAuth } from '../hooks/useAuth';

const BrandsScreen = ({ onBack, onBrandSelect, searchMode }) => {
  const [brands, setBrands] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  const auth = useAuth();

  useEffect(() => {
    loadBrands();
  }, []);

  const loadBrands = async () => {
    try {
      setLoading(true);
      setError(null);
      const brandsData = await perfumeAPI.getBrands();
      
      // Filtrar y validar marcas, luego convertir a mayúsculas
      const brandsWithDisplay = brandsData
        .filter(brand => brand && typeof brand === 'string') // Filtrar valores undefined/null
        .map(brand => ({
          original: brand,
          display: brand.toUpperCase()
        }));
      
      setBrands(brandsWithDisplay);
    } catch (err) {
      setError('Error cargando marcas');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Filtrar marcas con validación para evitar errores
  const filteredBrands = brands.filter(brand => {
    if (!brand || !brand.original) return false;
    return brand.original.toLowerCase().includes(searchQuery.toLowerCase());
  });

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

  // Estilos para el contenedor del grid con scroll
  const gridContainerStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
    gap: '1rem',
    marginTop: '1rem',
    maxHeight: 'calc(80px * 5 + 1rem * 4)', // 5 filas de 80px + gaps
    overflowY: 'auto',
    paddingRight: '0.5rem'
  };

  // Estilos personalizados para el scrollbar
  const scrollbarStyles = `
    .brands-grid::-webkit-scrollbar {
      width: 8px;
    }
    
    .brands-grid::-webkit-scrollbar-track {
      background: rgba(243, 229, 171, 0.1);
      border-radius: 4px;
    }
    
    .brands-grid::-webkit-scrollbar-thumb {
      background: rgba(243, 229, 171, 0.6);
      border-radius: 4px;
    }
    
    .brands-grid::-webkit-scrollbar-thumb:hover {
      background: rgba(243, 229, 171, 0.8);
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
      {/* Estilos del scrollbar */}
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
                display: 'inline-block',
                border: '2px solid #713600',
                boxShadow: '0 0 20px rgba(243,229,171), inset 0 0 15px rgba(255, 215, 0, 0.2)',
                color: '#713600',
                textTransform: 'uppercase'
              }}>
                MARCAS DE PERFUMES
              </div>
            </div>

            {/* Buscador */}
            <div style={{ position: 'relative', marginBottom: '1.5rem' }}>
              <input
                type="text"
                placeholder="Buscar marca..."
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
            </div>

            {/* Mensajes de estado */}
            {loading && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginBottom: '1rem' }}>
                🔍 Cargando marcas...
              </p>
            )}
            {error && (
              <p style={{ color: '#ff4444', textAlign: 'center', marginBottom: '1rem' }}>
                ❌ {error}
              </p>
            )}

            {/* Grid de marcas con scroll limitado a 5 filas */}
            <div 
              className="brands-grid"
              style={gridContainerStyle}
            >
              {filteredBrands.map((brand, index) => (
                <button
                  key={index}
                  className="menu-btn"
                  onClick={() => onBrandSelect(brand.original)}
                  style={{
                    padding: '1rem',
                    fontSize: '0.9rem',
                    height: '80px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textTransform: 'uppercase'
                  }}
                >
                  {brand.display}
                </button>
              ))}
            </div>

            {!loading && filteredBrands.length === 0 && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginTop: '2rem' }}>
                {searchQuery ? 'No se encontraron marcas' : 'No hay marcas disponibles'}
              </p>
            )}

          </div>
        </div>
      </div>
    </div>
  );
};

export default BrandsScreen;