// src/components/TopRatedScreen.js
import React, { useState, useEffect } from 'react';
import { 
  ArrowLeft, 
  Star, 
  Users, 
  MessageSquare,
  Trophy,
  Award
} from 'lucide-react';
import { perfumeAPI } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import PerfumeDetailModal from './PerfumeDetailModal';

const TopRatedScreen = ({ onBack, searchMode }) => {
  const [perfumes, setPerfumes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedPerfume, setSelectedPerfume] = useState(null);
  const [perfumeImages, setPerfumeImages] = useState({});
  
  const auth = useAuth();

  // MÉTODO ACTUALIZADO: Mismo que en WishlistScreen
  const getPerfumeImage = async (perfumeName, perfumeId) => {
    try {
      let id = perfumeId;
      
      // Si no hay ID, buscar por nombre para obtener el ID
      if (!id && perfumeName) {
        try {
          const searchResults = await perfumeAPI.searchPerfumes(perfumeName, 1);
          if (searchResults && searchResults.length > 0) {
            const perfume = searchResults[0];
            
            // Intentar extraer ID de la URL
            if (perfume.url) {
              const match = perfume.url.match(/(\d+)\.html$/);
              if (match) {
                id = match[1];
              }
            }
            
            // Usar ID del perfume si está disponible
            if (perfume.id) {
              id = perfume.id;
            }
          }
        } catch (err) {
          console.error('Error buscando perfume:', err);
        }
      }
      
      // Generar URL de imagen si tenemos ID
      if (id) {
        return `https://fimgs.net/mdimg/perfume/375x500.${id}.jpg`;
      }
      
      return null;
    } catch (error) {
      console.error('Error obteniendo imagen:', error);
      return null;
    }
  };

  // Nueva función para cargar imágenes de los perfumes
  const loadPerfumeImages = async (perfumesList) => {
    const imagePromises = perfumesList.map(async (perfume, index) => {
      try {
        const imageUrl = await getPerfumeImage(perfume.perfume || perfume.nombre, perfume.id);
        return {
          index,
          perfumeId: perfume.id || `perfume-${index}`,
          imageUrl
        };
      } catch (error) {
        console.error(`Error cargando imagen para ${perfume.perfume}:`, error);
        return {
          index,
          perfumeId: perfume.id || `perfume-${index}`,
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
      console.log(`✅ ${Object.keys(imagesMap).length} imágenes cargadas`);
    } catch (error) {
      console.error('Error cargando imágenes:', error);
    }
  };

  useEffect(() => {
    loadTopRatedPerfumes();
  }, []);

  const loadTopRatedPerfumes = async () => {
    setLoading(true);
    setError(null);
    setPerfumeImages({}); // Limpiar imágenes anteriores
    
    try {
      console.log('🏁 Cargando TOP 10 perfumes más votados...');
      
      // Obtener los perfumes más votados
      const perfumesData = await perfumeAPI.getTopRatedPerfumes();
      
      console.log('📦 Datos recibidos:', {
        cantidad: perfumesData?.length || 0,
        datos: perfumesData ? perfumesData.slice(0, 3) : 'No hay datos'
      });
      
      if (perfumesData && Array.isArray(perfumesData)) {
        // Tomar solo los primeros 10 perfumes
        const top10Perfumes = perfumesData.slice(0, 10);
        
        console.log(`🎨 Formateando ${top10Perfumes.length} perfumes TOP 10...`);
        
        // Formatear datos
        const formattedPerfumes = top10Perfumes.map((perfume, index) => {
          // Asegurar que main_accords sea un array
          let mainAccords = [];
          if (perfume.main_accords) {
            if (Array.isArray(perfume.main_accords)) {
              mainAccords = perfume.main_accords;
            } else if (typeof perfume.main_accords === 'string') {
              try {
                mainAccords = JSON.parse(perfume.main_accords);
              } catch (e) {
                mainAccords = perfume.main_accords.split(',').map(a => a.trim()).filter(a => a);
              }
            }
          }
          
          // Convertir guiones a espacios para mostrar mejor
          const formatName = (name) => {
            if (!name) return `Perfume ${index + 1}`;
            return name.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          };
          
          const perfumeName = formatName(perfume.perfume || perfume.nombre);
          const brand = formatName(perfume.marca || perfume.brand || 'Marca desconocida');
          
          return {
            id: perfume.id?.toString() || `perfume-${index}`,
            perfume: perfumeName,
            nombre: perfumeName,
            marca: brand,
            genero: perfume.genero || perfume.gender || 'Unisex',
            año: perfume.año || perfume.year || 'N/A',
            avg_rating: parseFloat(perfume.avg_rating || perfume.average_rating || 0),
            average_rating: parseFloat(perfume.average_rating || perfume.avg_rating || 0),
            total_votes: parseInt(perfume.total_votes || perfume.vote_count || 0),
            total_comments: parseInt(perfume.total_comments || perfume.comment_count || 0),
            main_accords: mainAccords,
            salida: perfume.salida || perfume.top_notes || perfume.notas_salida || '',
            corazon: perfume.corazon || perfume.heart_notes || perfume.notas_corazon || '',
            base: perfume.base || perfume.base_notes || perfume.notas_base || '',
            url: perfume.url || '',
            top_notes: perfume.top_notes || perfume.salida || perfume.notas_salida || '',
            heart_notes: perfume.heart_notes || perfume.corazon || perfume.notas_corazon || '',
            base_notes: perfume.base_notes || perfume.base || perfume.notas_base || ''
          };
        });
        
        console.log(`✅ ${formattedPerfumes.length} perfumes TOP 10 formateados`);
        
        // Establecer perfumes primero
        setPerfumes(formattedPerfumes);
        
        // Luego cargar las imágenes de forma asíncrona
        await loadPerfumeImages(formattedPerfumes);
        
      } else {
        console.log('⚠️ No se recibieron datos para el TOP 10');
        setPerfumes([]);
        setError('No se encontraron perfumes con votos. ¡Sé el primero en votar un perfume!');
      }
      
    } catch (err) {
      console.error('❌ Error cargando TOP 10 perfumes:', err);
      setError('Error cargando los perfumes más votados. Intenta nuevamente.');
      setPerfumes([]);
    } finally {
      setLoading(false);
      console.log('🏁 Carga TOP 10 completada');
    }
  };

  // Resto del código permanece igual...
  // Calcular color según rating
  const getRatingColor = (rating) => {
    if (rating >= 4.5) return '#10B981'; // Verde
    if (rating >= 4.0) return '#3B82F6'; // Azul
    if (rating >= 3.5) return '#F59E0B'; // Amarillo
    return '#EF4444'; // Rojo
  };

  // Formatear número
  const formatNumber = (num) => {
    if (!num) return '0';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toString();
  };

  const handleViewDetails = (perfume) => {
    setSelectedPerfume(perfume);
  };

  const getUserDisplayName = () => {
    if (searchMode === 'user' && auth.user) {
      return `${auth.user.name || 'Usuario'} ${auth.user.lastname || ''}`.trim();
    }
    return 'Usuario';
  };

  // Calcular estadísticas SOLO de los 10 perfumes
  const totalPerfumes = perfumes.length;
  const totalVotes = perfumes.reduce((sum, p) => sum + (p.total_votes || 0), 0);
  const avgRating = perfumes.length > 0 
    ? perfumes.reduce((sum, p) => {
        const rating = p.average_rating || p.avg_rating || 0;
        return sum + rating;
      }, 0) / perfumes.length
    : 0;

  // Función para obtener las notas formateadas
  const getFormattedNotes = (perfume) => {
    const notes = [];
    
    // Usar los campos correctos
    if (perfume.top_notes || perfume.salida) {
      const topNotes = (perfume.top_notes || perfume.salida || '');
      if (topNotes.trim()) {
        notes.push(`Salida: ${topNotes.split(',').slice(0, 2).join(',').trim()}`);
      }
    }
    if (perfume.heart_notes || perfume.corazon) {
      const heartNotes = (perfume.heart_notes || perfume.corazon || '');
      if (heartNotes.trim()) {
        notes.push(`Corazón: ${heartNotes.split(',').slice(0, 2).join(',').trim()}`);
      }
    }
    if (perfume.base_notes || perfume.base) {
      const baseNotes = (perfume.base_notes || perfume.base || '');
      if (baseNotes.trim()) {
        notes.push(`Base: ${baseNotes.split(',').slice(0, 2).join(',').trim()}`);
      }
    }
    
    return notes.slice(0, 3); // Limitar a 3 notas
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

  const gridContainerStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1.2rem',
    marginTop: '1rem',
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
          
          <button className="exit-btn" onClick={onBack}>
            <ArrowLeft size={16} /> Volver
          </button>
        </header>

        <div style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
          <div className="card" style={{ maxWidth: '1100px', margin: '0 auto', width: '95%' }}>
            
            {/* Título principal */}
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
                color: '#713700ff',
                textAlign: 'center'
              }}>
                <Trophy style={{ marginRight: '10px', display: 'inline' }} />
                TOP 10 + VOTADOS
              </div>
            </div>

            {/* Estadísticas */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)', 
              gap: '1rem', 
              marginBottom: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{
                background: 'rgba(113, 54, 0, 0.3)',
                padding: '1rem',
                borderRadius: '10px',
                border: '1px solid rgba(243, 229, 171, 0.2)'
              }}>
                <Award size={24} style={{ marginBottom: '0.5rem', color: '#FFD700' }} />
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#F3E5AB' }}>
                  {totalPerfumes}
                </div>
                <div style={{ color: 'rgba(243, 229, 171, 0.7)', fontSize: '0.9rem' }}>
                  Perfumes TOP
                </div>
              </div>
              
              <div style={{
                background: 'rgba(113, 54, 0, 0.3)',
                padding: '1rem',
                borderRadius: '10px',
                border: '1px solid rgba(243, 229, 171, 0.2)'
              }}>
                <Users size={24} style={{ marginBottom: '0.5rem', color: '#FFD700' }} />
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#F3E5AB' }}>
                  {formatNumber(totalVotes)}
                </div>
                <div style={{ color: 'rgba(243, 229, 171, 0.7)', fontSize: '0.9rem' }}>
                  Votos totales
                </div>
              </div>
              
              <div style={{
                background: 'rgba(113, 54, 0, 0.3)',
                padding: '1rem',
                borderRadius: '10px',
                border: '1px solid rgba(243, 229, 171, 0.2)'
              }}>
                <Star size={24} style={{ marginBottom: '0.5rem', color: '#FFD700' }} />
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#F3E5AB' }}>
                  {avgRating.toFixed(1)}
                </div>
                <div style={{ color: 'rgba(243, 229, 171, 0.7)', fontSize: '0.9rem' }}>
                  Puntuación promedio
                </div>
              </div>
            </div>

            {/* Error message */}
            {error && !loading && (
              <div style={{ 
                textAlign: 'center', 
                padding: '2rem', 
                color: '#F3E5AB',
                background: 'rgba(239, 68, 68, 0.2)',
                borderRadius: '0.5rem',
                marginBottom: '1rem',
                border: '1px solid rgba(239, 68, 68, 0.5)'
              }}>
                <p>{error}</p>
                <button 
                  onClick={loadTopRatedPerfumes}
                  style={{
                    marginTop: '1rem',
                    padding: '0.5rem 1rem',
                    background: 'rgba(243, 229, 171, 0.9)',
                    color: '#713600',
                    border: '1px solid #F3E5AB',
                    borderRadius: '0.3rem',
                    cursor: 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  Reintentar
                </button>
              </div>
            )}

            {/* Grid de perfumes TOP 10 */}
            {loading ? (
              <div style={{ textAlign: 'center', padding: '4rem', color: '#F3E5AB' }}>
                <div className="spinner" style={{
                  width: '50px',
                  height: '50px',
                  border: '4px solid rgba(243, 229, 171, 0.1)',
                  borderTop: '4px solid #F3E5AB',
                  borderRadius: '50%',
                  margin: '0 auto 1rem',
                  animation: 'spin 1s linear infinite'
                }}></div>
                <p>Cargando los 10 perfumes más votados...</p>
              </div>
            ) : (
              <div 
                className="perfumes-grid"
                style={gridContainerStyle}
              >
                {perfumes.length > 0 ? (
                  perfumes.map((perfume, index) => {
                    const rating = perfume.average_rating || perfume.avg_rating || 0;
                    const ratingColor = getRatingColor(rating);
                    const rank = index + 1;
                    const votes = perfume.total_votes || 0;
                    const comments = perfume.total_comments || 0;
                    const perfumeName = perfume.perfume || perfume.nombre || 'Sin nombre';
                    const brand = perfume.marca || 'Marca desconocida';
                    const gender = perfume.genero || 'Unisex';
                    const year = perfume.año || 'N/A';
                    
                    // Obtener imagen del estado de imágenes (MISMO MÉTODO que WishlistScreen)
                    const imageUrl = perfumeImages[perfume.id || `perfume-${index}`];
                    
                    return (
                      <div 
                        key={perfume.id || index}
                        style={{
                          padding: '1.2rem',
                          background: 'rgba(0, 0, 0, 0.9)',
                          borderRadius: '0.8rem',
                          border: '2px solid #F3E5AB',
                          color: '#F3E5AB',
                          display: 'flex',
                          flexDirection: 'column',
                          minHeight: '380px',
                          position: 'relative'
                        }}
                      >
                        {/* Badge de ranking */}
                        <div style={{
                          position: 'absolute',
                          top: '-10px',
                          left: '-10px',
                          width: '40px',
                          height: '40px',
                          background: rank <= 3 
                            ? ['#FFD700', '#C0C0C0', '#CD7F32'][rank - 1]
                            : 'rgba(113, 54, 0, 0.9)',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontWeight: 'bold',
                          fontSize: '1.1rem',
                          color: rank <= 3 ? '#000' : '#F3E5AB',
                          border: '2px solid #F3E5AB',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
                        }}>
                          #{rank}
                        </div>

                        {/* Imagen del perfume - AHORA CON EL MISMO MÉTODO que WishlistScreen */}
                        <div style={{ 
                          display: 'flex', 
                          justifyContent: 'center', 
                          marginBottom: '0.8rem',
                          marginTop: '0.5rem',
                          height: '120px'
                        }}>
                          {imageUrl ? (
                            <img
                              src={imageUrl}
                              alt={perfumeName}
                              loading="lazy"
                              style={{
                                width: '80px',
                                height: '100px',
                                objectFit: 'contain',
                                borderRadius: '0.4rem',
                                background: 'rgba(243, 229, 171, 0.15)',
                                padding: '0.3rem'
                              }}
                              onError={(e) => {
                                console.log(`❌ Error cargando imagen para ${perfumeName}:`, imageUrl);
                                e.target.style.display = 'none';
                                // Mostrar placeholder si falla
                                e.target.parentElement.innerHTML = `
                                  <div style="
                                    width: 80px;
                                    height: 100px;
                                    background: rgba(243, 229, 171, 0.1);
                                    border-radius: 0.4rem;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    color: #F3E5AB;
                                    font-size: 2rem;
                                  ">
                                    🎯
                                  </div>
                                `;
                              }}
                              onLoad={(e) => {
                                console.log(`✅ Imagen cargada para ${perfumeName}:`, imageUrl);
                              }}
                            />
                          ) : (
                            <div style={{
                              width: '80px',
                              height: '100px',
                              background: 'rgba(243, 229, 171, 0.1)',
                              borderRadius: '0.4rem',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: '#F3E5AB',
                              fontSize: '2rem'
                            }}>
                              🎯
                            </div>
                          )}
                        </div>

                        {/* Nombre y puntuación */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.8rem' }}>
                          <h3 style={{ 
                            margin: '0', 
                            color: '#F3E5AB', 
                            fontSize: '1.1rem',
                            lineHeight: '1.3',
                            flex: 1
                          }}>
                            {perfumeName}
                          </h3>
                          
                          <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            background: `${ratingColor}20`,
                            padding: '0.3rem 0.6rem',
                            borderRadius: '20px',
                            border: `1px solid ${ratingColor}`,
                            minWidth: '60px',
                            justifyContent: 'center'
                          }}>
                            <Star size={14} fill={ratingColor} color={ratingColor} />
                            <span style={{ 
                              marginLeft: '0.3rem',
                              fontWeight: 'bold',
                              color: ratingColor,
                              fontSize: '1rem'
                            }}>
                              {rating.toFixed(1)}
                            </span>
                          </div>
                        </div>

                        {/* Información básica */}
                        <div style={{ 
                          display: 'grid', 
                          gridTemplateColumns: '1fr 1fr', 
                          gap: '0.6rem', 
                          fontSize: '0.85rem',
                          marginBottom: '0.8rem'
                        }}>
                          <div><strong>Marca:</strong> {brand}</div>
                          <div><strong>Género:</strong> {gender}</div>
                        </div>

                        {/* Notas principales */}
                        {getFormattedNotes(perfume).length > 0 && (
                          <div style={{ 
                            marginBottom: '0.8rem', 
                            fontSize: '0.8rem',
                            background: 'rgba(243, 229, 171, 0.1)',
                            padding: '0.6rem',
                            borderRadius: '0.4rem',
                            flex: 1
                          }}>
                            <strong style={{ display: 'block', marginBottom: '0.2rem', fontSize: '0.85rem' }}>
                              Notas destacadas:
                            </strong>
                            {getFormattedNotes(perfume).map((note, idx) => (
                              <div key={idx} style={{ fontSize: '0.75rem', lineHeight: '1.3' }}>
                                • {note}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Estadísticas de comentarios */}
                        <div style={{ 
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginTop: 'auto',
                          paddingTop: '0.6rem',
                          borderTop: '1px solid rgba(243, 229, 171, 0.2)'
                        }}>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                              <Users size={12} color="#F3E5AB" />
                              <span style={{ fontSize: '0.75rem' }}>{formatNumber(votes)}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                              <MessageSquare size={12} color="#F3E5AB" />
                              <span style={{ fontSize: '0.75rem' }}>{formatNumber(comments)}</span>
                            </div>
                          </div>
                          
                          <button 
                            className="btn"
                            style={{ 
                              padding: '0.4rem 0.8rem', 
                              fontSize: '0.8rem',
                              background: 'rgba(243, 229, 171, 0.9)',
                              color: '#713600',
                              border: '1px solid #F3E5AB',
                              fontWeight: 'bold',
                              cursor: 'pointer',
                              borderRadius: '0.3rem'
                            }}
                            onClick={() => handleViewDetails(perfume)}
                          >
                            Ver Detalles
                          </button>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={{ 
                    gridColumn: '1 / -1', 
                    textAlign: 'center', 
                    padding: '3rem',
                    color: '#F3E5AB'
                  }}>
                    <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🏆</div>
                    <p>No hay perfumes votados aún. ¡Sé el primero en votar!</p>
                    <button 
                      onClick={loadTopRatedPerfumes}
                      style={{
                        marginTop: '1rem',
                        padding: '0.5rem 1rem',
                        background: 'rgba(243, 229, 171, 0.9)',
                        color: '#713600',
                        border: '1px solid #F3E5AB',
                        borderRadius: '0.3rem',
                        cursor: 'pointer',
                        fontWeight: 'bold'
                      }}
                    >
                      Reintentar
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Información total */}
            {!loading && perfumes.length > 0 && (
              <div style={{ 
                textAlign: 'center', 
                marginTop: '1.5rem',
                color: 'rgba(243, 229, 171, 0.8)',
                fontSize: '0.9rem',
                padding: '0.8rem',
                background: 'rgba(113, 54, 0, 0.2)',
                borderRadius: '0.5rem'
              }}>
                <p>
                  Estos son los <strong>10 perfumes mejor valorados</strong> por la comunidad ESSENCE, 
                  basados en las puntuaciones de {formatNumber(totalVotes)} votos.
                </p>
                <p style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
                  * Solo se muestran los 10 perfumes con mayor puntuación promedio
                </p>
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

      {/* Animación CSS */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .spinner {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default TopRatedScreen;