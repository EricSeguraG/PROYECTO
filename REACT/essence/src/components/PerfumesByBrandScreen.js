// src/components/PerfumesByBrandScreen.js
import React, { useState, useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';
import { perfumeAPI } from '../services/api';
import { useAuth } from '../hooks/useAuth';

const PerfumesByBrandScreen = ({ onBack, brandName, searchMode }) => {
  const [perfumes, setPerfumes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const auth = useAuth();

  useEffect(() => {
    loadPerfumes();
  }, [brandName]);

  const loadPerfumes = async () => {
    try {
      setLoading(true);
      setError(null);
      const perfumesData = await perfumeAPI.getPerfumesByBrand(brandName);
      setPerfumes(perfumesData);
    } catch (err) {
      setError('Error cargando perfumes');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (perfume) => {
    alert(`Detalles de ${perfume.perfume}\nMarca: ${perfume.marca}\nGénero: ${perfume.genero}\nAño: ${perfume.año || 'N/A'}`);
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

  // Estilos para el grid de perfumes con scroll (2 columnas)
  const gridContainerStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)', // 2 columnas
    gap: '1.2rem',
    marginTop: '1rem',
    maxHeight: 'calc(280px * 5 + 1.2rem * 4)', // 5 filas de 280px + gaps
    overflowY: 'auto',
    paddingRight: '0.5rem'
  };

  // Estilos personalizados para el scrollbar
  const scrollbarStyles = `
    .perfumes-grid::-webkit-scrollbar {
      width: 8px;
    }
    
    .perfumes-grid::-webkit-scrollbar-track {
      background: rgba(243, 229, 171, 0.1);
      border-radius: 4px;
    }
    
    .perfumes-grid::-webkit-scrollbar-thumb {
      background: rgba(243, 229, 171, 0.6);
      border-radius: 4px;
    }
    
    .perfumes-grid::-webkit-scrollbar-thumb:hover {
      background: rgba(243, 229, 171, 0.8);
    }

    @media (max-width: 768px) {
      .perfumes-grid {
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
          <div className="card" style={{ maxWidth: '1100px', margin: '0 auto', width: '95%' }}>
            
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
                textAlign: 'center'
              }}>
                PERFUMES {brandName.toUpperCase()}
              </div>
            </div>

            {/* Mensajes de estado */}
            {loading && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginBottom: '1rem' }}>
                🔍 Cargando perfumes...
              </p>
            )}
            {error && (
              <p style={{ color: '#ff4444', textAlign: 'center', marginBottom: '1rem' }}>
                ❌ {error}
              </p>
            )}

            {/* Grid de perfumes en 2 columnas con scroll */}
            <div 
              className="perfumes-grid"
              style={gridContainerStyle}
            >
              {perfumes.map((perfume, index) => (
                <div 
                  key={index}
                  style={{
                    padding: '1.2rem',
                    background: 'rgba(113, 54, 0, 0.9)',
                    borderRadius: '0.8rem',
                    border: '2px solid #F3E5AB',
                    color: '#F3E5AB',
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: '280px'
                  }}
                >
                  {/* Nombre del perfume */}
                  <h3 style={{ 
                    margin: '0 0 1rem 0', 
                    color: '#F3E5AB', 
                    fontSize: '1.1rem',
                    textAlign: 'center',
                    lineHeight: '1.3'
                  }}>
                    {perfume.perfume}
                  </h3>
                  
                  {/* Información básica en grid */}
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: '1fr 1fr', 
                    gap: '0.8rem', 
                    fontSize: '0.9rem',
                    marginBottom: '1rem',
                    flex: 1
                  }}>
                    <div><strong>Marca:</strong> {perfume.marca}</div>
                    <div><strong>Género:</strong> {perfume.genero}</div>
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

                  {/* Botón de detalles */}
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 'auto' }}>
                    <button 
                      className="btn"
                      style={{ 
                        padding: '0.5rem 1rem', 
                        fontSize: '0.85rem'
                      }}
                      onClick={() => handleViewDetails(perfume)}
                    >
                      Ver Detalles
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {!loading && perfumes.length === 0 && (
              <p style={{ color: '#F3E5AB', textAlign: 'center', marginTop: '2rem' }}>
                No se encontraron perfumes para {brandName}
              </p>
            )}

            {/* Contador de perfumes */}
            {!loading && perfumes.length > 0 && (
              <p style={{ 
                color: '#F3E5AB', 
                textAlign: 'center', 
                marginTop: '1rem',
                fontSize: '0.9rem'
              }}>
                📊 {perfumes.length} {perfumes.length === 1 ? 'PERFUME ENCONTRADO' : 'PERFUMES ENCONTRADOS'}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerfumesByBrandScreen;