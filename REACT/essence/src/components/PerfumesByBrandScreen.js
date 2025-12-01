// src/components/PerfumesByBrandScreen.js
import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, ChevronLeft, ChevronRight, Loader } from 'lucide-react';
import { perfumeAPI } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import PerfumeDetailModal from './PerfumeDetailModal';

const PerfumesByBrandScreen = ({ onBack, brandName, searchMode }) => {
  const [perfumes, setPerfumes] = useState([]);
  const [allPerfumes, setAllPerfumes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPerfume, setSelectedPerfume] = useState(null);
  
  // Estados de paginación
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoadingMorePages, setIsLoadingMorePages] = useState(false);
  
  const gridRef = useRef(null);
  const auth = useAuth();

  useEffect(() => {
    loadAllPerfumes();
  }, [brandName]);

  // Función para ordenar perfumes alfabéticamente
  const sortPerfumesAlphabetically = (perfumesArray) => {
    return [...perfumesArray].sort((a, b) => {
      const nameA = (a.perfume || a.nombre || '').toLowerCase();
      const nameB = (b.perfume || b.nombre || '').toLowerCase();
      return nameA.localeCompare(nameB);
    });
  };

  // Cargar TODOS los perfumes de la marca
  const loadAllPerfumes = async () => {
    try {
      setLoading(true);
      setError(null);
      setAllPerfumes([]);
      setPerfumes([]);
      
      console.log(`🔍 Cargando perfumes de: ${brandName}`);
      
      // Cargar primera página
      const firstPageData = await perfumeAPI.getPerfumesByBrand(brandName, 1);
      console.log(`📊 Página 1: ${firstPageData.length} perfumes`);
      
      if (firstPageData.length === 0) {
        setLoading(false);
        return;
      }
      
      // Cargar el resto de páginas si es necesario
      await loadRemainingPages(firstPageData);
      
    } catch (err) {
      setError('Error cargando perfumes');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Cargar páginas restantes
  const loadRemainingPages = async (firstPageData) => {
    let allPerfumesData = [...firstPageData];
    
    // Si la primera página tiene 20 perfumes, puede haber más páginas
    if (firstPageData.length >= 20) {
      setIsLoadingMorePages(true);
      
      try {
        // Intentar cargar hasta 10 páginas (máximo 200 perfumes)
        for (let page = 2; page <= 10; page++) {
          console.log(`🔄 Intentando cargar página ${page}...`);
          const pageData = await perfumeAPI.getPerfumesByBrand(brandName, page);
          
          // Si la página está vacía, detenemos
          if (!pageData || pageData.length === 0) {
            console.log(`⏹️ Página ${page} vacía, deteniendo...`);
            break;
          }
          
          allPerfumesData = [...allPerfumesData, ...pageData];
          console.log(`✅ Página ${page}: ${pageData.length} perfumes (Total: ${allPerfumesData.length})`);
          
          // Si la página tiene menos de 20 perfumes, es la última
          if (pageData.length < 20) {
            console.log(`🎯 Última página detectada (${pageData.length} perfumes)`);
            break;
          }
          
          // Pequeña pausa para no sobrecargar
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      } catch (error) {
        console.error('Error cargando páginas adicionales:', error);
      } finally {
        setIsLoadingMorePages(false);
      }
    }
    
    // Ordenar todos los perfumes alfabéticamente
    const sortedPerfumes = sortPerfumesAlphabetically(allPerfumesData);
    setAllPerfumes(sortedPerfumes);
    
    // Calcular páginas totales
    const total = Math.ceil(sortedPerfumes.length / itemsPerPage);
    setTotalPages(total);
    
    // Mostrar primera página
    goToPage(1, sortedPerfumes);
    
    console.log(`🎯 TOTAL cargado: ${sortedPerfumes.length} perfumes en ${total} páginas (ordenados alfabéticamente)`);
  };

  // Cambiar de página
  const goToPage = (pageNumber, perfumesData = allPerfumes) => {
    if (pageNumber < 1 || pageNumber > totalPages) return;
    
    setCurrentPage(pageNumber);
    
    const startIndex = (pageNumber - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const pagePerfumes = perfumesData.slice(startIndex, endIndex);
    
    setPerfumes(pagePerfumes);
    
    // Scroll al top del grid
    if (gridRef.current) {
      gridRef.current.scrollTop = 0;
    }
  };

  // Ir a la página siguiente
  const nextPage = () => {
    if (currentPage < totalPages) {
      goToPage(currentPage + 1);
    }
  };

  // Ir a la página anterior
  const prevPage = () => {
    if (currentPage > 1) {
      goToPage(currentPage - 1);
    }
  };

  // Generar números de página para mostrar
  const getPageNumbers = () => {
    const pages = [];
    const maxVisiblePages = 5;
    
    if (totalPages <= maxVisiblePages) {
      // Mostrar todas las páginas
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Mostrar páginas con elipsis
      if (currentPage <= 3) {
        // Primeras páginas
        for (let i = 1; i <= 4; i++) {
          pages.push(i);
        }
        pages.push('...');
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        // Últimas páginas
        pages.push(1);
        pages.push('...');
        for (let i = totalPages - 3; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        // Páginas intermedias
        pages.push(1);
        pages.push('...');
        pages.push(currentPage - 1);
        pages.push(currentPage);
        pages.push(currentPage + 1);
        pages.push('...');
        pages.push(totalPages);
      }
    }
    
    return pages;
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

  // Calcular índices para mostrar
  const startIndex = (currentPage - 1) * itemsPerPage + 1;
  const endIndex = Math.min(currentPage * itemsPerPage, allPerfumes.length);

  return (
    <div style={containerStyle}>
      <style>{scrollbarStyles}</style>
      
      <video autoPlay muted loop playsInline style={videoStyle}>
        <source src="/videos/vid.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      
      <div style={overlayStyle}></div>

      <div style={contentStyle}>
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

        <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
          <div className="card" style={{ maxWidth: '1100px', margin: '0 auto', width: '95%' }}>
            
            {/* Título con indicador de orden alfabético */}
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
                PERFUMES {brandName.toUpperCase()} (A-Z)
              </div>
            </div>

            {/* SE ELIMINÓ LA SECCIÓN DE INFORMACIÓN DE CARGA Y PAGINACIÓN SUPERIOR */}
            {/* Aquí estaba la información que mostraba "Página X de X" y los controles de paginación */}

            {/* Mensajes de error */}
            {error && (
              <p style={{ color: '#ff4444', textAlign: 'center', marginBottom: '1rem' }}>
                ❌ {error}
              </p>
            )}

            {/* Grid de perfumes */}
            {!loading && (
              <div 
                className="perfumes-grid"
                style={gridContainerStyle}
                ref={gridRef}
              >
                {perfumes.length > 0 ? (
                  perfumes.map((perfume, index) => (
                    <div 
                      key={`${perfume.id || index}-${currentPage}`}
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
                        {perfume.perfume || perfume.nombre}
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
                  ))
                ) : (
                  <div style={{ 
                    gridColumn: '1 / -1', 
                    textAlign: 'center', 
                    padding: '3rem',
                    color: '#F3E5AB'
                  }}>
                    <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>😔</div>
                    <p>No se encontraron perfumes para {brandName}</p>
                  </div>
                )}
              </div>
            )}

            {/* Controles de paginación inferiores */}
            {!loading && totalPages > 1 && (
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '0.5rem',
                flexWrap: 'wrap',
                background: 'rgba(113, 54, 0, 0.2)',
                padding: '1rem',
                borderRadius: '0.5rem',
                marginTop: '1.5rem'
              }}>
                <span style={{ color: '#F3E5AB', fontSize: '0.9rem', marginRight: '1rem' }}>
                  Página {currentPage} de {totalPages}
                </span>
                
                <button
                  onClick={prevPage}
                  disabled={currentPage === 1}
                  style={{
                    padding: '0.5rem 1rem',
                    background: currentPage === 1 
                      ? 'rgba(243, 229, 171, 0.3)' 
                      : 'rgba(243, 229, 171, 0.9)',
                    color: currentPage === 1 ? 'rgba(113, 54, 0, 0.5)' : '#713600',
                    border: '1px solid #F3E5AB',
                    borderRadius: '0.3rem',
                    cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.3rem',
                    fontWeight: 'bold'
                  }}
                >
                  <ChevronLeft size={16} />
                  Anterior
                </button>
                
                {/* Números de página inferiores */}
                <div style={{ display: 'flex', gap: '0.3rem' }}>
                  {getPageNumbers().map((pageNum, index) => (
                    <button
                      key={`bottom-${index}`}
                      onClick={() => typeof pageNum === 'number' && goToPage(pageNum)}
                      style={{
                        minWidth: '2rem',
                        height: '2rem',
                        padding: '0 0.5rem',
                        background: pageNum === currentPage 
                          ? 'rgba(243, 229, 171, 0.9)'
                          : 'rgba(243, 229, 171, 0.2)',
                        color: pageNum === currentPage ? '#713600' : '#F3E5AB',
                        border: `1px solid ${pageNum === currentPage ? '#F3E5AB' : 'rgba(243, 229, 171, 0.3)'}`,
                        borderRadius: '0.3rem',
                        cursor: typeof pageNum === 'number' ? 'pointer' : 'default',
                        fontWeight: pageNum === currentPage ? 'bold' : 'normal'
                      }}
                      disabled={typeof pageNum !== 'number'}
                    >
                      {pageNum}
                    </button>
                  ))}
                </div>
                
                <button
                  onClick={nextPage}
                  disabled={currentPage === totalPages}
                  style={{
                    padding: '0.5rem 1rem',
                    background: currentPage === totalPages 
                      ? 'rgba(243, 229, 171, 0.3)' 
                      : 'rgba(243, 229, 171, 0.9)',
                    color: currentPage === totalPages ? 'rgba(113, 54, 0, 0.5)' : '#713600',
                    border: '1px solid #F3E5AB',
                    borderRadius: '0.3rem',
                    cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.3rem',
                    fontWeight: 'bold'
                  }}
                >
                  Siguiente
                  <ChevronRight size={16} />
                </button>
              </div>
            )}

            {/* Información total */}
            {!loading && allPerfumes.length > 0 && (
              <div style={{ 
                textAlign: 'center', 
                marginTop: '1rem',
                color: 'rgba(243, 229, 171, 0.8)',
                fontSize: '0.9rem',
                padding: '0.8rem',
                background: 'rgba(113, 54, 0, 0.2)',
                borderRadius: '0.5rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                  <span> Total: {allPerfumes.length} perfumes</span>
                 
                  {allPerfumes.length % itemsPerPage !== 0 && (
                    <>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
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

      {/* Animación CSS para el spinner */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default PerfumesByBrandScreen;