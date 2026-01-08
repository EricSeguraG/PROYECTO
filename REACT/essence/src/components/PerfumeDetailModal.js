import React, { useState, useEffect } from 'react';
import { X, Star, Send, User, Calendar, Heart, Bookmark } from 'lucide-react';
import { perfumeAPI } from '../services/api';

const PerfumeDetailModal = ({ perfume, user, onClose }) => {
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState("");
  const [rating, setRating] = useState(5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  // Estados para wishlist y colección
  const [inWishlist, setInWishlist] = useState(false);
  const [inCollection, setInCollection] = useState(false);
  const [loadingWishlist, setLoadingWishlist] = useState(false);
  const [loadingCollection, setLoadingCollection] = useState(false);

  useEffect(() => {
    if (perfume?.perfume || perfume?.id) {
      loadComments();
      checkUserLists();
    }
  }, [perfume, user]);

  const checkUserLists = async () => {
    if (!user || !perfume) return;
    
    try {
      // Usamos encodeURIComponent para manejar espacios y caracteres especiales
      const identifier = perfume.id || encodeURIComponent(perfume.perfume || perfume.nombre);
      
      // Check wishlist
      const wishlistCheck = await perfumeAPI.checkInWishlist(user.id, identifier);
      setInWishlist(wishlistCheck.exists || false);
      
      // Check collection
      const collectionCheck = await perfumeAPI.checkInCollection(user.id, identifier);
      setInCollection(collectionCheck.exists || false);
    } catch (err) {
      console.error("Error checking user lists:", err);
    }
  };

  const loadComments = async () => {
    setLoading(true);
    try {
      const identifier = perfume.id || encodeURIComponent(perfume.perfume || perfume.nombre);
      const data = await perfumeAPI.getComments(identifier);
      setComments(data || []);
    } catch (err) {
      console.error("Error cargando comentarios:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleWishlistToggle = async () => {
    if (!user) {
      setError("Inicia sesión para usar la wishlist");
      return;
    }
    
    setLoadingWishlist(true);
    try {
      const identifier = perfume.id || encodeURIComponent(perfume.perfume || perfume.nombre);
      
      if (inWishlist) {
        await perfumeAPI.removeFromWishlist(user.id, identifier);
        setInWishlist(false);
      } else {
        await perfumeAPI.addToWishlist({
          usuario_id: user.id,
          perfume_id: perfume.id || null,
          perfume_name: perfume.perfume || perfume.nombre,
          marca: perfume.marca,
          genero: perfume.genero,
          año: perfume.año,
          main_accords: perfume.main_accords,
          notas_salida: perfume.salida,
          notas_corazon: perfume.corazon,
          notas_base: perfume.base,
          perfumista: perfume.perfumista
        });
        setInWishlist(true);
      }
    } catch (err) {
      setError("Error al actualizar wishlist: " + err.message);
      console.error(err);
    } finally {
      setLoadingWishlist(false);
    }
  };

  const handleCollectionToggle = async () => {
    if (!user) {
      setError("Inicia sesión para gestionar tu colección");
      return;
    }
    
    setLoadingCollection(true);
    try {
      const identifier = perfume.id || encodeURIComponent(perfume.perfume || perfume.nombre);
      
      if (inCollection) {
        await perfumeAPI.removeFromCollection(user.id, identifier);
        setInCollection(false);
      } else {
        await perfumeAPI.addToCollection({
          usuario_id: user.id,
          perfume_id: perfume.id || null,
          perfume_name: perfume.perfume || perfume.nombre,
          marca: perfume.marca,
          genero: perfume.genero,
          año: perfume.año,
          main_accords: perfume.main_accords,
          notas_salida: perfume.salida,
          notas_corazon: perfume.corazon,
          notas_base: perfume.base,
          perfumista: perfume.perfumista,
          fecha_adquisicion: new Date().toISOString().split('T')[0]
        });
        setInCollection(true);
      }
    } catch (err) {
      setError("Error al actualizar colección: " + err.message);
      console.error(err);
    } finally {
      setLoadingCollection(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!newComment.trim() || !user) {
      setError("Escribe un comentario primero");
      return;
    }

    try {
      setError(null);
      await perfumeAPI.addComment({
        usuario_id: user.id,
        perfume_id: perfume.id || null,
        perfume_name: perfume.perfume || perfume.nombre,
        texto: newComment,
        puntuacion: rating
      });
      
      setNewComment("");
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
      loadComments();
    } catch (error) {
      setError("Error al enviar comentario: " + (error.message || "Intenta de nuevo"));
    }
  };

  // --- ESTILOS MEJORADOS ---
  const overlayStyle = {
    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    display: 'flex', justifyContent: 'center', alignItems: 'center',
    zIndex: 1000, padding: '1rem',
    backdropFilter: 'blur(5px)'
  };

  const modalStyle = {
    backgroundColor: 'rgba(17, 17, 17, 0.95)',
    border: '2px solid #F3E5AB',
    borderRadius: '16px',
    width: '100%', maxWidth: '600px', maxHeight: '85vh',
    display: 'flex', flexDirection: 'column',
    position: 'relative', 
    boxShadow: '0 0 40px rgba(243, 229, 171, 0.3)',
    overflow: 'hidden'
  };

  if (!perfume) return null;

  return (
    <div style={overlayStyle}>
      <div style={modalStyle}>
        
        {/* HEADER CON BOTONES DE WISHLIST Y COLECCIÓN */}
        <div style={{ 
          padding: '1.5rem 1.5rem 1rem 1.5rem', 
          background: 'linear-gradient(135deg, #713600 0%, #3d1c00 100%)',
          borderBottom: '2px solid #F3E5AB',
          position: 'relative'
        }}>
          <button 
            onClick={onClose} 
            style={{ 
              position: 'absolute', 
              top: '1rem', 
              right: '1rem',
              background: 'rgba(243, 229, 171, 0.1)', 
              border: '1px solid #F3E5AB',
              borderRadius: '50%',
              width: '32px',
              height: '32px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#F3E5AB', 
              cursor: 'pointer'
            }}
          >
            <X size={18} />
          </button>
          
          <h2 style={{ 
            color: '#F3E5AB', 
            margin: '0 0 0.5rem 0', 
            fontSize: '1.4rem',
            fontFamily: '"Orbitron", sans-serif',
            letterSpacing: '1px',
            textShadow: '0 2px 4px rgba(0,0,0,0.5)'
          }}>
            {perfume.perfume || perfume.nombre}
          </h2>
          
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem'
          }}>
            <div style={{ 
              display: 'flex', 
              gap: '1rem', 
              color: 'rgba(243, 229, 171, 0.8)',
              fontSize: '0.9rem'
            }}>
              <span>🏷️ {perfume.marca}</span>
              <span>👤 {perfume.genero}</span>
              {perfume.año && <span>📅 {perfume.año}</span>}
            </div>
            
            {/* BOTONES DE WISHLIST Y COLECCIÓN */}
            {user && (
              <div style={{ 
                display: 'flex', 
                gap: '0.5rem',
                alignItems: 'center'
              }}>
                {/* BOTÓN WISHLIST */}
                <button
                  onClick={handleWishlistToggle}
                  disabled={loadingWishlist}
                  style={{
                    background: inWishlist 
                      ? 'linear-gradient(135deg, #ff3366 0%, #cc0044 100%)' 
                      : 'rgba(243, 229, 171, 0.1)',
                    border: `1px solid ${inWishlist ? '#ff3366' : '#ffee00ff'}`,
                    borderRadius: '20px',
                    padding: '0.5rem 1rem',
                    color: inWishlist ? '#FFFFFF' : '#F3E5AB',
                    cursor: loadingWishlist ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.3rem',
                    fontSize: '0.85rem',
                    fontWeight: '500',
                    transition: 'all 0.3s ease',
                    opacity: loadingWishlist ? 0.7 : 1
                  }}
                  title={inWishlist ? "Quitar de wishlist" : "Añadir a wishlist"}
                >
                  {loadingWishlist ? (
                    <div style={{ 
                      width: '14px', 
                      height: '14px', 
                      border: '2px solid rgba(243, 229, 171, 0.3)',
                      borderTop: '2px solid #F3E5AB',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }}></div>
                  ) : (
                    <Heart size={16} fill={inWishlist ? "#FFFFFF" : "transparent"} />
                  )}
                  {inWishlist ? 'En Wishlist' : 'Wishlist'}
                </button>
                
                {/* BOTÓN COLECCIÓN */}
                <button
                  onClick={handleCollectionToggle}
                  disabled={loadingCollection}
                  style={{
                    background: inCollection 
                      ? 'linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%)' 
                      : 'rgba(243, 229, 171, 0.1)',
                    border: `1px solid ${inCollection ? '#4CAF50' : '#ffee00ff'}`,
                    borderRadius: '20px',
                    padding: '0.5rem 1rem',
                    color: inCollection ? '#FFFFFF' : '#F3E5AB',
                    cursor: loadingCollection ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.3rem',
                    fontSize: '0.85rem',
                    fontWeight: '500',
                    transition: 'all 0.3s ease',
                    opacity: loadingCollection ? 0.7 : 1
                  }}
                  title={inCollection ? "Quitar de mi colección" : "Añadir a mi colección"}
                >
                  {loadingCollection ? (
                    <div style={{ 
                      width: '14px', 
                      height: '14px', 
                      border: '2px solid rgba(243, 229, 171, 0.3)',
                      borderTop: '2px solid #F3E5AB',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }}></div>
                  ) : (
                    <Bookmark size={16} fill={inCollection ? "#FFFFFF" : "transparent"} />
                  )}
                  {inCollection ? 'En Colección' : 'Colección'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* NOTA PARA USUARIOS NO LOGUEADOS */}
        {!user && (
          <div style={{ 
            padding: '0.8rem 1.5rem', 
            background: 'rgba(113, 54, 0, 0.2)',
            borderBottom: '1px solid rgba(243, 229, 171, 0.2)',
            textAlign: 'center'
          }}>
            <p style={{ 
              color: 'rgba(243, 229, 171, 0.7)', 
              margin: 0, 
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}>
              <User size={14} />
              Inicia sesión para añadir perfumes a tu wishlist o colección
            </p>
          </div>
        )}

        {/* CONTENIDO SCROLLEABLE */}
        <div style={{ 
          padding: '1.5rem', 
          overflowY: 'auto',
          flex: 1,
          background: 'linear-gradient(180deg, rgba(26,26,26,1) 0%, rgba(17,17,17,1) 100%)'
        }}>
          
          {/* INFO ADICIONAL DEL PERFUME */}
          <div style={{ 
            marginBottom: '2rem', 
            padding: '1.2rem',
            background: 'rgba(113, 54, 0, 0.15)',
            border: '1px solid rgba(243, 229, 171, 0.2)',
            borderRadius: '10px'
          }}>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '1fr 1fr', 
              gap: '1rem',
              color: 'rgba(255,255,255,0.8)',
              fontSize: '0.9rem'
            }}>
              {perfume.perfumista && (
                <div>
                  <strong style={{ color: '#F3E5AB' }}>Perfumista:</strong> {perfume.perfumista}
                </div>
              )}
              {perfume.salida && (
                <div>
                  <strong style={{ color: '#F3E5AB' }}>Notas de Salida:</strong> {perfume.salida}
                </div>
              )}
              {perfume.corazon && (
                <div>
                  <strong style={{ color: '#F3E5AB' }}>Notas de Corazón:</strong> {perfume.corazon}
                </div>
              )}
              {perfume.base && (
                <div>
                  <strong style={{ color: '#F3E5AB' }}>Notas de Base:</strong> {perfume.base}
                </div>
              )}
              
              {/* SECCIÓN DE ACORDES */}
              {perfume.main_accords && (
                <div style={{ 
                  gridColumn: 'span 2',
                  marginTop: '0.5rem',
                  paddingTop: '0.5rem',
                  borderTop: '1px solid rgba(243, 229, 171, 0.2)'
                }}>
                  <strong style={{ 
                    color: '#F3E5AB',
                    display: 'block',
                    marginBottom: '0.3rem'
                  }}>
                    Acordes:
                  </strong>
                  <div style={{ 
                    color: 'rgba(255,255,255,0.9)',
                    fontSize: '0.9rem',
                    lineHeight: '1.4'
                  }}>
                    {Array.isArray(perfume.main_accords) 
                      ? perfume.main_accords.join(', ')
                      : perfume.main_accords}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* SECCIÓN COMENTARIOS */}
          <div style={{ 
            marginBottom: '1.5rem'
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '1rem'
            }}>
              <h3 style={{ 
                color: '#F3E5AB', 
                fontSize: '1.1rem',
                fontFamily: '"Orbitron", sans-serif',
                letterSpacing: '0.5px',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                💬 Comentarios 
                <span style={{ 
                  background: 'rgba(243, 229, 171, 0.2)',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '0.8rem'
                }}>
                  {comments.length}
                </span>
              </h3>
            </div>

            {/* LISTA DE COMENTARIOS */}
            <div style={{ 
              maxHeight: '250px', 
              overflowY: 'auto',
              marginBottom: '1rem',
              paddingRight: '5px'
            }}>
              {loading ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '2rem',
                  color: 'rgba(243, 229, 171, 0.6)'
                }}>
                  <div style={{ 
                    width: '40px', 
                    height: '40px', 
                    border: '3px solid rgba(243, 229, 171, 0.3)',
                    borderTop: '3px solid #FFD700',
                    borderRadius: '50%',
                    margin: '0 auto 1rem',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  Cargando opiniones...
                </div>
              ) : comments.length === 0 ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '2rem',
                  background: 'rgba(113, 54, 0, 0.1)',
                  borderRadius: '10px',
                  border: '1px dashed rgba(243, 229, 171, 0.3)'
                }}>
                  <div style={{ 
                    fontSize: '2rem',
                    marginBottom: '0.5rem',
                    opacity: 0.5
                  }}>
                    💬
                  </div>
                  <p style={{ 
                    color: 'rgba(243, 229, 171, 0.6)', 
                    fontStyle: 'italic',
                    margin: 0
                  }}>
                    Sé el primero en opinar sobre este perfume.
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  {comments.map((c) => (
                    <div key={c.id} style={{ 
                      background: 'rgba(113, 54, 0, 0.2)',
                      padding: '1rem',
                      borderRadius: '12px',
                      border: '1px solid rgba(243, 229, 171, 0.15)'
                    }}>
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'flex-start',
                        marginBottom: '0.5rem'
                      }}>
                        <div>
                          <div style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '0.5rem',
                            marginBottom: '0.2rem'
                          }}>
                            <div style={{
                              width: '32px',
                              height: '32px',
                              borderRadius: '50%',
                              background: 'linear-gradient(135deg, #713600 0%, #FFD700 100%)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: '#000',
                              fontWeight: 'bold',
                              fontSize: '0.9rem'
                            }}>
                              {c.nombre.charAt(0)}{c.apellido.charAt(0)}
                            </div>
                            <div>
                              <strong style={{ 
                                color: '#F3E5AB', 
                                fontSize: '0.95rem',
                                display: 'block'
                              }}>
                                {c.nombre} {c.apellido}
                              </strong>
                              <small style={{ 
                                color: 'rgba(243, 229, 171, 0.6)', 
                                fontSize: '0.75rem',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.3rem'
                              }}>
                                <Calendar size={10} />
                                {new Date(c.fecha_creacion).toLocaleDateString('es-ES', {
                                  day: 'numeric',
                                  month: 'short',
                                  year: 'numeric'
                                })}
                              </small>
                            </div>
                          </div>
                        </div>
                        
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center',
                          gap: '0.2rem',
                          background: 'rgba(0,0,0,0.3)',
                          padding: '0.3rem 0.6rem',
                          borderRadius: '20px',
                          border: '1px solid rgba(255, 215, 0, 0.3)'
                        }}>
                          <span style={{ 
                            color: '#FFD700',
                            fontSize: '0.9rem',
                            fontWeight: 'bold',
                            textShadow: '0 0 3px rgba(255, 215, 0, 0.5)'
                          }}>
                            {c.puntuacion}
                          </span>
                          <Star size={14} fill="#FFD700" color="#FFD700" />
                        </div>
                      </div>
                      
                      <p style={{ 
                        color: 'rgba(255,255,255,0.9)', 
                        margin: 0, 
                        fontSize: '0.9rem',
                        lineHeight: '1.5',
                        paddingLeft: '2.5rem'
                      }}>
                        {c.texto}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* FORMULARIO PARA AGREGAR COMENTARIO */}
          {user ? (
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '1rem'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#F3E5AB' }}>
                  <span style={{ 
                    fontSize: '0.9rem',
                    fontWeight: '500'
                  }}>
                    Tu puntuación:
                  </span>
                  <div style={{ 
                    display: 'flex', 
                    gap: '0.2rem',
                    background: 'rgba(255, 215, 0, 0.1)',
                    padding: '0.4rem 0.8rem',
                    borderRadius: '20px',
                    border: '1px solid rgba(255, 215, 0, 0.2)'
                  }}>
                    {[1, 2, 3, 4, 5].map((star) => (
                      <Star 
                        key={star} 
                        size={24} 
                        fill={star <= rating ? "#FFD700" : "rgba(255, 215, 0, 0.2)"} 
                        color={star <= rating ? "#FFD700" : "rgba(255, 215, 0, 0.5)"}
                        style={{ 
                          cursor: 'pointer'
                        }}
                        onClick={() => setRating(star)}
                      />
                    ))}
                  </div>
                </div>
                
                {error && (
                  <div style={{ 
                    background: 'rgba(255, 107, 107, 0.1)', 
                    color: '#ff6b6b',
                    padding: '0.5rem 0.8rem',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    border: '1px solid rgba(255, 107, 107, 0.3)'
                  }}>
                    {error}
                  </div>
                )}
                
                {success && (
                  <div style={{ 
                    background: 'rgba(76, 175, 80, 0.1)', 
                    color: '#4CAF50',
                    padding: '0.5rem 0.8rem',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    border: '1px solid rgba(76, 175, 80, 0.3)'
                  }}>
                    ✅ Comentario enviado correctamente
                  </div>
                )}
              </div>
              
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input
                  type="text"
                  placeholder="Comparte tu experiencia con este perfume..."
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  style={{ 
                    flex: 1, 
                    padding: '0.9rem 1rem', 
                    borderRadius: '10px', 
                    border: '1px solid rgba(243, 229, 171, 0.3)',
                    backgroundColor: 'rgba(0, 0, 0, 0.4)', 
                    color: 'white',
                    fontSize: '0.9rem'
                  }}
                />
                <button 
                  type="submit" 
                  disabled={!newComment.trim()}
                  style={{ 
                    background: 'linear-gradient(135deg, #713600 0%, #FFD700 100%)',
                    border: 'none', 
                    borderRadius: '10px', 
                    width: '50px', 
                    height: '50px',
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    cursor: newComment.trim() ? 'pointer' : 'not-allowed', 
                    opacity: newComment.trim() ? 1 : 0.5
                  }}
                >
                  <Send size={20} color="#000" />
                </button>
              </div>
              
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem',
                color: 'rgba(243, 229, 171, 0.6)',
                fontSize: '0.8rem',
                paddingLeft: '0.5rem'
              }}>
                <User size={14} />
                Comentando como: {user.name} {user.lastname}
              </div>
            </form>
          ) : (
            <div style={{ 
              textAlign: 'center', 
              padding: '1.5rem', 
              background: 'rgba(113, 54, 0, 0.15)',
              borderRadius: '10px', 
              border: '1px solid rgba(243, 229, 171, 0.2)'
            }}>
              <div style={{ 
                fontSize: '1.8rem',
                marginBottom: '0.8rem'
              }}>
                🔒
              </div>
              <p style={{ 
                color: 'rgba(243, 229, 171, 0.8)', 
                marginBottom: '0.5rem',
                fontWeight: '500'
              }}>
                Inicia sesión para compartir tu opinión
              </p>
              <p style={{ 
                color: 'rgba(243, 229, 171, 0.6)', 
                fontSize: '0.9rem',
                margin: 0
              }}>
                Únete a la comunidad ESSENCE y comenta sobre este perfume
              </p>
            </div>
          )}

        </div>
      </div>
      
      {/* Animación CSS para el spinner */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        /* Estilos para scrollbar personalizado */
        div::-webkit-scrollbar {
          width: 6px;
        }
        
        div::-webkit-scrollbar-track {
          background: rgba(113, 54, 0, 0.1);
          border-radius: 3px;
        }
        
        div::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #713600, #FFD700);
          border-radius: 3px;
        }
        
        div::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #FFD700, #713600);
        }
      `}</style>
    </div>
  );
};

export default PerfumeDetailModal;