// src/components/CollectionScreen.js
import React, { useState, useEffect, useRef } from 'react';
import { 
  Heart, Trash2, ArrowLeft, 
  Calendar, User, AlertCircle,
  Home, ExternalLink, ChevronLeft, ChevronRight,
  CheckCircle, SortAsc, SortDesc
} from 'lucide-react';
import { perfumeAPI } from '../services/api';
import { getCurrentUser, isAuthenticated } from '../services/api';
import PerfumeDetailModal from './PerfumeDetailModal';

const CollectionScreen = ({ onBack, user: propUser }) => {
  const [collection, setCollection] = useState([]);
  const [sortedCollection, setSortedCollection] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [user, setUser] = useState(propUser);
  
  const [selectedPerfume, setSelectedPerfume] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [perfumeImages, setPerfumeImages] = useState({});
  
  // Nuevo estado para controlar el orden
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc' o 'desc'
  
  const gridRef = useRef(null);

  // Función para ordenar la colección
  const sortCollection = (collection, order = 'asc') => {
    if (!collection || collection.length === 0) return [];
    
    const sorted = [...collection].sort((a, b) => {
      const nameA = (a.perfume_name || '').toLowerCase();
      const nameB = (b.perfume_name || '').toLowerCase();
      
      if (order === 'asc') {
        return nameA.localeCompare(nameB);
      } else {
        return nameB.localeCompare(nameA);
      }
    });
    
    return sorted;
  };

  // Función para obtener imágenes
  const getPerfumeImage = async (perfumeName, perfumeId) => {
    try {
      let id = perfumeId;
      
      if (!id && perfumeName) {
        try {
          const searchResults = await perfumeAPI.searchPerfumes(perfumeName, 1);
          if (searchResults && searchResults.length > 0) {
            const perfume = searchResults[0];
            
            if (perfume.url) {
              const match = perfume.url.match(/(\d+)\.html$/);
              if (match) {
                id = match[1];
              }
            }
            
            if (perfume.id) {
              id = perfume.id;
            }
          }
        } catch (err) {
          console.error('Error buscando perfume:', err);
        }
      }
      
      if (id) {
        return `https://fimgs.net/mdimg/perfume/375x500.${id}.jpg`;
      }
      
      return null;
    } catch (error) {
      console.error('Error obteniendo imagen:', error);
      return null;
    }
  };

  useEffect(() => {
    if (!user) {
      const currentUser = getCurrentUser();
      if (!currentUser || !isAuthenticated()) {
        onBack();
        return;
      }
      setUser(currentUser);
    }
    
    if (user?.id) {
      loadCollection(user.id);
    }
  }, [user, onBack]);

  const loadCollection = async (userId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await perfumeAPI.getCollection(userId);
      
      // Ordenar la colección alfabéticamente
      const sortedData = sortCollection(data || [], sortOrder);
      
      setCollection(data || []);
      setSortedCollection(sortedData);
      
      // Cargar imágenes para los perfumes
      loadPerfumeImages(sortedData);
      
    } catch (err) {
      console.error('Error cargando colección:', err);
      setError('No se pudo cargar tu colección. Intenta de nuevo.');
    } finally {
      setLoading(false);
    }
  };

  // Función para cargar imágenes de los perfumes
  const loadPerfumeImages = async (perfumes) => {
    const imagePromises = perfumes.map(async (perfume, index) => {
      try {
        const imageUrl = await getPerfumeImage(perfume.perfume_name, perfume.perfume_id);
        return {
          index,
          perfumeId: perfume.perfume_id || perfume.perfume_name,
          imageUrl
        };
      } catch (error) {
        console.error(`Error cargando imagen para ${perfume.perfume_name}:`, error);
        return {
          index,
          perfumeId: perfume.perfume_id || perfume.perfume_name,
          imageUrl: null
        };
      }
    });

    try {
      const imageResults = await Promise.all(imagePromises);
      const imagesMap = {};
      imageResults.forEach(result => {
        imagesMap[result.perfumeId] = result.imageUrl;
      });
      setPerfumeImages(imagesMap);
    } catch (error) {
      console.error('Error cargando imágenes:', error);
    }
  };

  const loadPerfumeDetails = async (item) => {
    try {
      const identifier = item.perfume_id || encodeURIComponent(item.perfume_name);
      const perfumeDetails = await perfumeAPI.searchPerfumes(identifier, 1);
      
      if (perfumeDetails && perfumeDetails.length > 0) {
        return perfumeDetails[0];
      }
      
      return {
        id: item.perfume_id,
        perfume: item.perfume_name,
        nombre: item.perfume_name,
        marca: item.marca,
        genero: item.genero,
        año: item.año,
        url: item.url,
        main_accords: item.main_accords,
        salida: item.notas_salida,
        corazon: item.notas_corazon,
        base: item.notas_base,
        perfumista: item.perfumista,
        fecha_adquisicion: item.fecha_adquisicion
      };
    } catch (err) {
      console.error('Error cargando detalles:', err);
      return {
        id: item.perfume_id,
        perfume: item.perfume_name,
        nombre: item.perfume_name,
        marca: item.marca,
        genero: item.genero,
        año: item.año,
        url: item.url,
        fecha_adquisicion: item.fecha_adquisicion
      };
    }
  };

  const handlePerfumeClick = async (item) => {
    try {
      const perfumeDetails = await loadPerfumeDetails(item);
      setSelectedPerfume(perfumeDetails);
      setShowDetailModal(true);
    } catch (err) {
      console.error('Error al abrir detalles:', err);
      setError('No se pudieron cargar los detalles del perfume');
    }
  };

  const handleRemoveFromCollection = async (perfumeId, perfumeName) => {
    if (!user) return;
    
    if (!window.confirm(`¿Eliminar "${perfumeName}" de tu colección?`)) {
      return;
    }
    
    try {
      const identifier = perfumeId || encodeURIComponent(perfumeName);
      await perfumeAPI.removeFromCollection(user.id, identifier);
      
      // Eliminar el item correcto
      const updatedCollection = collection.filter(item => {
        if (perfumeId && item.perfume_id === perfumeId) {
          return false;
        }
        if (item.perfume_name === perfumeName) {
          return false;
        }
        return true;
      });
      
      // Reordenar la colección actualizada
      const sortedUpdatedCollection = sortCollection(updatedCollection, sortOrder);
      
      setCollection(updatedCollection);
      setSortedCollection(sortedUpdatedCollection);
      
      // Actualizar imágenes
      const updatedImages = { ...perfumeImages };
      delete updatedImages[perfumeId || perfumeName];
      setPerfumeImages(updatedImages);
      
      setError(null);
    } catch (err) {
      console.error('Error eliminando de colección:', err);
      setError('Error al eliminar de la colección. Intenta de nuevo.');
    }
  };

  // Función para cambiar el orden
  const toggleSortOrder = () => {
    const newOrder = sortOrder === 'asc' ? 'desc' : 'asc';
    setSortOrder(newOrder);
    
    const sorted = sortCollection(collection, newOrder);
    setSortedCollection(sorted);
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

  // Grid con más columnas para mejor densidad
  const gridContainerStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
    gap: '1rem',
    marginTop: '1rem',
    overflowY: 'auto',
    paddingRight: '0.5rem',
    paddingBottom: '1rem'
  };

  const scrollbarStyles = `
    .perfumes-grid::-webkit-scrollbar {
      width: 6px;
    }
    
    .perfumes-grid::-webkit-scrollbar-track {
      background: rgba(243, 229, 171, 0.1);
      border-radius: 3px;
    }
    
    .perfumes-grid::-webkit-scrollbar-thumb {
      background: rgba(243, 229, 171, 0.6);
      border-radius: 3px;
    }
    
    .perfumes-grid::-webkit-scrollbar-thumb:hover {
      background: rgba(243, 229, 171, 0.8);
    }

    @media (max-width: 768px) {
      .perfumes-grid {
        grid-template-columns: repeat(2, 1fr) !important;
      }
    }

    @media (max-width: 480px) {
      .perfumes-grid {
        grid-template-columns: 1fr !important;
      }
    }
  `;

  if (loading) {
    return (
      <div style={containerStyle}>
        <video autoPlay muted loop playsInline style={videoStyle}>
          <source src="/videos/vid.mp4" type="video/mp4" />
        </video>
        <div style={overlayStyle}></div>
        
        <div style={{ 
          position: 'relative',
          zIndex: 2,
          maxWidth: '1200px', 
          margin: '0 auto',
          padding: '4rem 1rem',
          textAlign: 'center'
        }}>
          <div style={{
            width: '60px',
            height: '60px',
            border: '3px solid rgba(243, 229, 171, 0.2)',
            borderTop: '3px solid #F3E5AB',
            borderRadius: '50%',
            margin: '0 auto 2rem',
            animation: 'spin 1s linear infinite'
          }}></div>
          <h2 style={{ color: '#F3E5AB', marginBottom: '1rem', fontSize: '1.8rem' }}>
            Cargando tu colección...
          </h2>
          <p style={{ color: 'rgba(243, 229, 171, 0.6)', fontSize: '1rem' }}>
            Preparando tu colección personal
          </p>
        </div>
        <style jsx>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (!user) {
    return (
      <div style={containerStyle}>
        <video autoPlay muted loop playsInline style={videoStyle}>
          <source src="/videos/vid.mp4" type="video/mp4" />
        </video>
        <div style={overlayStyle}></div>
        
        <div style={{ 
          position: 'relative',
          zIndex: 2,
          maxWidth: '600px', 
          margin: '4rem auto',
          textAlign: 'center',
          padding: '3rem 2rem',
          background: 'rgba(0, 0, 0, 0.9)',
          borderRadius: '12px',
          border: '2px solid #F3E5AB',
          color: '#F3E5AB'
        }}>
          <AlertCircle size={64} color="rgba(243, 229, 171, 0.5)" style={{ marginBottom: '1.5rem' }} />
          <h2 style={{ color: '#F3E5AB', marginBottom: '1rem', fontSize: '1.8rem' }}>
            Acceso requerido
          </h2>
          <p style={{ color: 'rgba(243, 229, 171, 0.7)', marginBottom: '2rem', fontSize: '1rem' }}>
            Debes iniciar sesión para ver tu colección
          </p>
          <button
            onClick={onBack}
            style={{
              background: 'linear-gradient(135deg, #713600 0%, #FFD700 100%)',
              border: 'none',
              borderRadius: '25px',
              padding: '0.9rem 2rem',
              color: '#000',
              fontWeight: '600',
              cursor: 'pointer',
              fontSize: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            <Home size={18} /> Volver atrás
          </button>
        </div>
      </div>
    );
  }

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
              {user?.name || "Usuario"} {user?.lastname || ""}
            </span>
          </div>
          
          <button className="exit-btn" onClick={onBack}>
            <ArrowLeft size={16} /> Volver
          </button>
        </header>

        <div style={{ flex: 1, padding: '1rem', overflowY: 'auto' }}>
          <div className="card" style={{ maxWidth: '1400px', margin: '0 auto', width: '95%' }}>
            
            {/* Título con botón de ordenación */}
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '1rem', gap: '1rem' }}>
              <div style={{
                fontSize: '1.4rem',
                letterSpacing: '1px',
                fontWeight: '700',
                fontFamily: '"Orbitron", sans-serif',
                background: 'rgb(243,229,171)',
                padding: '0.6rem 1.2rem',
                borderRadius: '8px',
                display: 'inline-block',
                border: '2px solid #713600',
                boxShadow: '0 0 10px rgba(243,229,171), inset 0 0 8px rgba(255, 215, 0, 0.2)',
                color: '#713700ff',
                textAlign: 'center',
                position: 'relative'
              }}>
                MI COLECCIÓN ({collection.length})
                <div style={{
                  position: 'absolute',
                  top: '-10px',
                  right: '-10px',
                  background: 'linear-gradient(135deg, #00ff1eff 0%, #00ff1eff 100%)',
                  width: '25px',
                  height: '25px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid #000000ff',
                  boxShadow: '0 0 5px rgba(255, 255, 255, 1)'
                }}>
                  <CheckCircle size={12} color="#ffffffff" fill="#00ff1eff" />
                </div>
              </div>
           
            </div>

            {error && (
              <div style={{
                background: 'rgba(255, 107, 107, 0.1)',
                border: '1px solid rgba(255, 107, 107, 0.3)',
                borderRadius: '6px',
                padding: '0.8rem',
                marginBottom: '1rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                color: '#ff6b6b',
                fontSize: '0.9rem'
              }}>
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            {/* Grid de perfumes - TODOS los perfumes */}
            {!loading && (
              <div 
                className="perfumes-grid"
                style={gridContainerStyle}
                ref={gridRef}
              >
                {sortedCollection.length > 0 ? (
                  sortedCollection.map((item, index) => {
                    // Obtener imagen del estado de imágenes
                    const imageUrl = perfumeImages[item.perfume_id || item.perfume_name];
                    
                    return (
                      <div 
                        key={`${item.perfume_id || index}`}
                        style={{
                          background: 'rgba(0, 0, 0, 0.9)',
                          borderRadius: '8px',
                          border: '2px solid #F3E5AB',
                          color: '#F3E5AB',
                          display: 'flex',
                          flexDirection: 'column',
                          height: '130px',
                          position: 'relative',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          overflow: 'hidden'
                        }}
                        onClick={() => handlePerfumeClick(item)}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.transform = 'translateY(-2px)';
                          e.currentTarget.style.boxShadow = '0 4px 12px rgba(243, 229, 171, 0.15)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = 'none';
                        }}
                      >
                        {/* Contenido: Imagen + Nombre al lado */}
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '0.8rem',
                          flex: 1,
                          padding: '0.6rem'
                        }}>
                          {/* Imagen cuadrada */}
                          <div style={{
                            width: '60px',
                            height: '60px',
                            flexShrink: 0,
                            background: imageUrl ? 'transparent' : 'rgba(243, 229, 171, 0.05)',
                            borderRadius: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            overflow: 'hidden',
                            border: imageUrl ? '1px solid rgba(243, 229, 171, 0.2)' : '1px dashed rgba(243, 229, 171, 0.2)'
                          }}>
                            {imageUrl ? (
                              <img
                                src={imageUrl}
                                alt={item.perfume_name || 'Perfume'}
                                loading="lazy"
                                style={{
                                  width: '100%',
                                  height: '100%',
                                  objectFit: 'cover'
                                }}
                                onError={(e) => {
                                  console.error('Error cargando imagen:', imageUrl);
                                  e.target.style.display = 'none';
                                  const container = e.target.parentElement;
                                  container.innerHTML = `
                                    <div style="color: rgba(243, 229, 171, 0.2); font-size: 1.2rem; display: flex; align-items: center; justify-content: center;">
                                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                                        <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                                      </svg>
                                    </div>
                                  `;
                                }}
                              />
                            ) : (
                              <div style={{ 
                                color: 'rgba(243, 229, 171, 0.2)', 
                                fontSize: '1.2rem',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                              }}>
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                                  <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                                </svg>
                              </div>
                            )}
                          </div>
                          
                          {/* Nombre del perfume */}
                          <div style={{ 
                            flex: 1,
                            minWidth: 0
                          }}>
                            <h3 style={{ 
                              margin: '0', 
                              color: '#F3E5AB', 
                              fontSize: '0.85rem',
                              lineHeight: '1.2',
                              fontWeight: '600',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical'
                            }}>
                              {item.perfume_name || 'Nombre no disponible'}
                            </h3>
                            
                            {/* Pequeña información adicional */}
                            <div>
                              {item.marca && (
                                <div style={{ 
                                  color: 'rgba(243, 229, 171, 0.6)', 
                                  fontSize: '0.7rem',
                                  marginTop: '0.2rem',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap'
                                }}>
                                  {item.marca}
                                </div>
                              )}
                              {item.fecha_adquisicion && (
                                <div style={{ 
                                  color: 'rgba(9, 255, 0, 1)', 
                                  fontSize: '0.65rem',
                                  marginTop: '0.1rem'
                                }}>
                                  <Calendar size={8} style={{ verticalAlign: 'middle', marginRight: '2px' }} />
                                  {new Date(item.fecha_adquisicion).toLocaleDateString('es-ES', {
                                    year: 'numeric',
                                    month: 'short'
                                  })}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Botón eliminar COMPACTO en la parte inferior */}
                        <div style={{ 
                          borderTop: '1px solid rgba(243, 229, 171, 0.1)',
                          padding: '0.3rem 0.6rem',
                          display: 'flex',
                          justifyContent: 'flex-end'
                        }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRemoveFromCollection(item.perfume_id, item.perfume_name);
                            }}
                            style={{
                              background: 'rgba(255, 107, 107, 0.05)',
                              border: '1px solid rgba(255, 107, 107, 0.2)',
                              borderRadius: '3px',
                              padding: '0.15rem 0.4rem',
                              color: '#ffffffff',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.2rem',
                              fontSize: '0.65rem',
                              transition: 'all 0.15s ease'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = 'rgba(255, 107, 107, 0.15)';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = 'rgba(255, 107, 107, 0.05)';
                            }}
                            title="Eliminar de colección"
                          >
                            <Trash2 size={9} />
                            Eliminar
                          </button>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={{ 
                    gridColumn: '1 / -1', 
                    textAlign: 'center', 
                    padding: '3rem 2rem',
                    color: '#F3E5AB'
                  }}>
                    <div style={{ 
                      width: '80px', 
                      height: '80px', 
                      margin: '0 auto 1.5rem',
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, rgba(113, 54, 0, 0.2) 0%, rgba(255, 215, 0, 0.05) 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '2px dashed rgba(243, 229, 171, 0.3)'
                    }}>
                      <CheckCircle size={36} color="rgba(243, 229, 171, 0.5)" />
                    </div>
                    <h3 style={{ 
                      color: '#F3E5AB', 
                      marginBottom: '1rem',
                      fontSize: '1.3rem'
                    }}>
                      TU COLECCIÓN ESTÁ VACÍA
                    </h3>
                    <p style={{ 
                      color: 'rgba(255, 255, 255, 1)', 
                      marginBottom: '2rem',
                      maxWidth: '400px',
                      margin: '0 auto',
                      fontSize: '0.9rem',
                      lineHeight: '1.5'
                    }}>
                      Añade perfumes a tu colección usando el botón en los detalles de cada perfume.
                    </p>
                    <button
                      onClick={onBack}
                      style={{
                        background: 'linear-gradient(135deg, #713600 0%, #FFD700 100%)',
                        border: 'none',
                        borderRadius: '20px',
                        padding: '0.6rem 1.5rem',
                        color: '#000',
                        fontWeight: '600',
                        cursor: 'pointer',
                        fontSize: '0.85rem',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.5rem'
                      }}
                    >
                      <ExternalLink size={14} />
                      Explorar perfumes
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Información del total */}
            {!loading && sortedCollection.length > 0 && (
              <div style={{ 
                textAlign: 'center', 
                marginTop: '0.8rem',
                color: 'rgba(243, 229, 171, 0.8)',
                fontSize: '0.8rem',
                padding: '0.5rem',
                background: 'rgba(113, 54, 0, 0.2)',
                borderRadius: '0.4rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '0.8rem', flexWrap: 'wrap' }}>
                  <span>
                    Mostrando {sortedCollection.length} perfumes en total
                  </span>
                  <span style={{ color: '#00ff1eff' }}>
                    Ordenado: {sortOrder === 'asc' ? 'A → Z' : 'Z → A'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Modal de Detalle */}
        {showDetailModal && selectedPerfume && (
          <PerfumeDetailModal
            perfume={selectedPerfume}
            user={user}
            onClose={() => {
              setShowDetailModal(false);
              setSelectedPerfume(null);
              if (user?.id) {
                loadCollection(user.id);
              }
            }}
          />
        )}
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default CollectionScreen;