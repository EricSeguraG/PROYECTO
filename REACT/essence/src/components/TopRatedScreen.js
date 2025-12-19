// src/components/TopRatedScreen.js
import React, { useState, useEffect } from 'react';
import { 
  ArrowLeft, 
  Star, 
  Search, 
  Users, 
  MessageSquare,
  TrendingUp,
  ChevronLeft,
  ChevronRight,
  Filter,
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
  
  // Estados de paginación
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  
  // Estado para búsqueda
  const [searchQuery, setSearchQuery] = useState('');
  
  const auth = useAuth();

  // Función para obtener la imagen del perfume
  const getPerfumeImage = (url) => {
    if (!url) return null;

    const match = url.match(/(\d+)\.html$/);
    if (!match) return null;

    const id = match[1];
    return `https://fimgs.net/mdimg/perfume/375x500.${id}.jpg`;
  };

  useEffect(() => {
    loadTopRatedPerfumes();
  }, []);

  const loadTopRatedPerfumes = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Suponiendo que tu API tiene este endpoint
      const response = await perfumeAPI.getTopRated({
        sort: 'avg_rating',
        limit: 100
      });
      
      // Si la respuesta tiene estructura { success, data }
      const perfumesData = response.success ? response.data : response;
      setPerfumes(perfumesData || []);
      
      // Calcular total de páginas
      const total = Math.ceil((perfumesData?.length || 0) / itemsPerPage);
      setTotalPages(total || 1);
      
    } catch (err) {
      console.error('Error cargando perfumes más votados:', err);
      setError('Error cargando los perfumes más votados');
      // Datos de ejemplo para desarrollo
      setPerfumes(getSampleData());
      setTotalPages(1);
    } finally {
      setLoading(false);
    }
  };

  // Datos de ejemplo
  const getSampleData = () => {
    return [
      {
        id: 1,
        perfume: "Sauvage",
        nombre: "Sauvage",
        marca: "Dior",
        genero: "Masculino",
        año: "2015",
        avg_rating: 4.8,
        total_votes: 1284,
        total_comments: 245,
        main_accords: ["Amaderado", "Ambarino", "Especiado"],
        salida: "Pimienta de Sichuan, Lavanda",
        corazon: "Pachulí, Vetiver, Lavanda absoluta",
        base: "Ámbar, Almizcle, Cedro",
        url: "https://www.fragrantica.com/perfume/Dior/Sauvage-22993.html"
      },
      {
        id: 2,
        perfume: "Baccarat Rouge 540",
        nombre: "Baccarat Rouge 540",
        marca: "Maison Francis Kurkdjian",
        genero: "Unisex",
        año: "2015",
        avg_rating: 4.7,
        total_votes: 1056,
        total_comments: 189,
        main_accords: ["Ambarino", "Azafrán", "Madera de Cedro"],
        salida: "Azafrán, Azahar",
        corazon: "Madera de cedro, Ámbar",
        base: "Madera de fresno, Almizcle",
        url: "https://www.fragrantica.com/perfume/Maison-Francis-Kurkdjian/Baccarat-Rouge-540-41673.html"
      },
      {
        id: 3,
        perfume: "Aventus",
        nombre: "Aventus",
        marca: "Creed",
        genero: "Masculino",
        año: "2010",
        avg_rating: 4.6,
        total_votes: 892,
        total_comments: 167,
        main_accords: ["Frutal", "Amaderado", "Ahumado"],
        salida: "Piña, Manzana, Bergamota",
        corazon: "Abedul, Pimienta rosa, Pachulí",
        base: "Musk, Almizcle, Roble",
        url: "https://www.fragrantica.com/perfume/Creed/Aventus-9828.html"
      },
      {
        id: 4,
        perfume: "Black Opium",
        nombre: "Black Opium",
        marca: "Yves Saint Laurent",
        genero: "Femenino",
        año: "2014",
        avg_rating: 4.5,
        total_votes: 756,
        total_comments: 142,
        main_accords: ["Café", "Vainilla", "Floral"],
        salida: "Pera rosa, Pera",
        corazon: "Café, Flor de naranjo",
        base: "Vainilla, Sándalo, Cedro",
        url: "https://www.fragrantica.com/perfume/Yves-Saint-Laurent/Black-Opium-27406.html"
      },
      {
        id: 5,
        perfume: "Bleu de Chanel",
        nombre: "Bleu de Chanel",
        marca: "Chanel",
        genero: "Masculino",
        año: "2010",
        avg_rating: 4.4,
        total_votes: 689,
        total_comments: 123,
        main_accords: ["Cítrico", "Amaderado", "Especiado"],
        salida: "Limón, Menta, Pomelo",
        corazon: "Jengibre, Nuez moscada, Jasmine",
        base: "Incienso, Vetiver, Sándalo",
        url: "https://www.fragrantica.com/perfume/Chanel/Bleu-de-Chanel-7098.html"
      }
    ];
  };

  // Filtrar perfumes según búsqueda
  const getFilteredPerfumes = () => {
    if (!searchQuery.trim()) {
      return perfumes;
    }
    
    const query = searchQuery.toLowerCase();
    return perfumes.filter(perfume => {
      const perfumeName = (perfume.perfume || perfume.nombre || '').toLowerCase();
      const brand = (perfume.marca || '').toLowerCase();
      const notes = [
        perfume.salida || '',
        perfume.corazon || '',
        perfume.base || '',
        (perfume.main_accords || []).join(' ') || ''
      ].join(' ').toLowerCase();
      
      return perfumeName.includes(query) || 
             brand.includes(query) || 
             notes.includes(query);
    });
  };

  // Obtener perfumes para mostrar
  const getDisplayPerfumes = () => {
    const filtered = getFilteredPerfumes();
    const totalFilteredPages = Math.ceil(filtered.length / itemsPerPage);
    
    if (totalFilteredPages !== totalPages) {
      setTotalPages(totalFilteredPages);
    }
    
    if (currentPage > totalFilteredPages && totalFilteredPages > 0) {
      setCurrentPage(totalFilteredPages);
    }
    
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filtered.slice(startIndex, endIndex);
  };

  // Cambiar de página
  const goToPage = (pageNumber) => {
    if (pageNumber < 1 || pageNumber > totalPages) return;
    setCurrentPage(pageNumber);
  };

  // Calcular color según rating
  const getRatingColor = (rating) => {
    if (rating >= 4.5) return '#10B981'; // Verde
    if (rating >= 4.0) return '#3B82F6'; // Azul
    if (rating >= 3.5) return '#F59E0B'; // Amarillo
    return '#EF4444'; // Rojo
  };

  // Formatear número
  const formatNumber = (num) => {
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

  // Calcular estadísticas
  const totalPerfumes = perfumes.length;
  const totalVotes = perfumes.reduce((sum, p) => sum + (p.total_votes || 0), 0);
  const avgRating = perfumes.length > 0 
    ? perfumes.reduce((sum, p) => sum + (p.avg_rating || 0), 0) / perfumes.length
    : 0;

  const filteredPerfumes = getFilteredPerfumes();
  const displayPerfumes = getDisplayPerfumes();

  // --- ESTILOS COMO PerfumesByBrandScreen ---
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
    maxHeight: 'calc(350px * 5 + 1.2rem * 4)',
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
                LOS + VOTADOS
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
                  Perfumes
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

            {/* Buscador */}
            <div style={{ position: 'relative', marginBottom: '1.5rem' }}>
              <input
                type="text"
                placeholder="Buscar entre los más votados..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setCurrentPage(1);
                }}
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

            {/* Información de búsqueda */}
            {searchQuery && (
              <div style={{ 
                color: '#F3E5AB', 
                textAlign: 'center', 
                marginBottom: '1rem',
                fontSize: '0.9rem'
              }}>
                🔍 Mostrando {filteredPerfumes.length} perfumes de {totalPerfumes} totales
                {filteredPerfumes.length === 0 && ' - No se encontraron resultados'}
              </div>
            )}

            {/* Grid de perfumes */}
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
                <p>Cargando los perfumes más votados...</p>
              </div>
            ) : (
              <div 
                className="perfumes-grid"
                style={gridContainerStyle}
              >
                {displayPerfumes.length > 0 ? (
                  displayPerfumes.map((perfume, index) => {
                    const ratingColor = getRatingColor(perfume.avg_rating);
                    const rank = ((currentPage - 1) * itemsPerPage) + index + 1;
                    
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
                          minHeight: '350px',
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

                        {/* Imagen del perfume - AÑADIDO */}
                        {perfume.url && getPerfumeImage(perfume.url) && (
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'center', 
                            marginBottom: '0.8rem',
                            marginTop: '0.5rem'
                          }}>
                            <img
                              src={getPerfumeImage(perfume.url)}
                              alt={perfume.perfume || perfume.nombre}
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
                                e.target.style.display = 'none';
                              }}
                            />
                          </div>
                        )}

                        {/* Nombre y puntuación */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                          <h3 style={{ 
                            margin: '0', 
                            color: '#F3E5AB', 
                            fontSize: '1.1rem',
                            lineHeight: '1.3',
                            flex: 1
                          }}>
                            {perfume.perfume || perfume.nombre}
                          </h3>
                          
                          <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            background: `${ratingColor}20`,
                            padding: '0.3rem 0.6rem',
                            borderRadius: '20px',
                            border: `1px solid ${ratingColor}`
                          }}>
                            <Star size={14} fill={ratingColor} color={ratingColor} />
                            <span style={{ 
                              marginLeft: '0.3rem',
                              fontWeight: 'bold',
                              color: ratingColor,
                              fontSize: '1rem'
                            }}>
                              {perfume.avg_rating?.toFixed(1) || '0.0'}
                            </span>
                          </div>
                        </div>

                        {/* Información básica */}
                        <div style={{ 
                          display: 'grid', 
                          gridTemplateColumns: '1fr 1fr', 
                          gap: '0.8rem', 
                          fontSize: '0.9rem',
                          marginBottom: '1rem'
                        }}>
                          <div><strong>Marca:</strong> {perfume.marca}</div>
                          <div><strong>Género:</strong> {perfume.genero}</div>
                          <div><strong>Año:</strong> {perfume.año || 'N/A'}</div>
                          <div><strong>Votos:</strong> {formatNumber(perfume.total_votes || 0)}</div>
                        </div>

                        {/* Notas principales */}
                        {(perfume.salida || perfume.corazon || perfume.base) && (
                          <div style={{ 
                            marginBottom: '1rem', 
                            fontSize: '0.85rem',
                            background: 'rgba(243, 229, 171, 0.1)',
                            padding: '0.8rem',
                            borderRadius: '0.4rem',
                            flex: 1
                          }}>
                            <strong style={{ display: 'block', marginBottom: '0.3rem' }}>Notas destacadas:</strong>
                            {perfume.salida && <div style={{ fontSize: '0.8rem' }}>• {perfume.salida.split(',').slice(0, 2).join(',')}</div>}
                            {perfume.corazon && <div style={{ fontSize: '0.8rem' }}>• {perfume.corazon.split(',').slice(0, 2).join(',')}</div>}
                            {perfume.base && <div style={{ fontSize: '0.8rem' }}>• {perfume.base.split(',').slice(0, 2).join(',')}</div>}
                          </div>
                        )}

                        {/* Estadísticas de comentarios */}
                        <div style={{ 
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginTop: 'auto',
                          paddingTop: '0.8rem',
                          borderTop: '1px solid rgba(243, 229, 171, 0.2)'
                        }}>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                              <Users size={14} color="#F3E5AB" />
                              <span style={{ fontSize: '0.8rem' }}>{formatNumber(perfume.total_votes || 0)}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                              <MessageSquare size={14} color="#F3E5AB" />
                              <span style={{ fontSize: '0.8rem' }}>{formatNumber(perfume.total_comments || 0)}</span>
                            </div>
                          </div>
                          
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
                    <p>
                      {searchQuery 
                        ? `No se encontraron perfumes que coincidan con "${searchQuery}"`
                        : 'No hay perfumes votados aún. ¡Sé el primero en votar!'
                      }
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Paginación */}
            {!loading && totalPages > 1 && filteredPerfumes.length > 0 && (
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
                  onClick={() => goToPage(currentPage - 1)}
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
                
                {/* Números de página */}
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = currentPage - 2 + i;
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => goToPage(pageNum)}
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
                        cursor: 'pointer',
                        fontWeight: pageNum === currentPage ? 'bold' : 'normal'
                      }}
                    >
                      {pageNum}
                    </button>
                  );
                })}
                
                <button
                  onClick={() => goToPage(currentPage + 1)}
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
            {!loading && perfumes.length > 0 && (
              <div style={{ 
                textAlign: 'center', 
                marginTop: '1rem',
                color: 'rgba(243, 229, 171, 0.8)',
                fontSize: '0.9rem',
                padding: '0.8rem',
                background: 'rgba(113, 54, 0, 0.2)',
                borderRadius: '0.5rem'
              }}>
                <p>
                  Estos son los perfumes mejor valorados por la comunidad ESSENCE, 
                  basados en las puntuaciones de {formatNumber(totalVotes)} votos.
                </p>
                <p style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
                  * Las puntuaciones se calculan a partir de los comentarios de los usuarios
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