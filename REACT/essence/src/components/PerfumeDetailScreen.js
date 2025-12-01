import React, { useState, useEffect } from 'react';
import { ArrowLeft, User, MessageSquare, Send, Star, Calendar, Award, Droplet, Sparkles } from 'lucide-react';
import { perfumeAPI } from '../services/api';
import { useAuth } from '../hooks/useAuth';

const PerfumeDetailScreen = ({ perfume, onBack }) => {
  const { user } = useAuth();
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [commentText, setCommentText] = useState('');
  const [rating, setRating] = useState(5);
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  const perfumeIdentifier = perfume.id || encodeURIComponent(perfume.perfume);

  useEffect(() => {
    loadComments();
  }, [perfume]);

  const loadComments = async () => {
    setLoading(true);
    try {
      const data = await perfumeAPI.getComments(perfumeIdentifier);
      setComments(data || []);
    } catch (err) {
      console.error("Error cargando comentarios:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!commentText.trim() || !user) {
      setError("Escribe un comentario primero");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await perfumeAPI.addComment({
        usuario_id: user.id,
        perfume_id: perfume.id || null,
        perfume_name: perfume.perfume || perfume.nombre,
        texto: commentText,
        puntuacion: rating
      });
      
      setCommentText('');
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
      loadComments();
    } catch (err) {
      console.error("Error al guardar:", err);
      setError("No se pudo guardar el comentario. Inténtalo de nuevo.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // --- ESTILOS MEJORADOS ---
  const containerStyle = {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: '#000'
  };

  const videoStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    zIndex: 0,
    filter: 'brightness(0.3)'
  };

  const overlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'linear-gradient(135deg, rgba(113, 54, 0, 0.4) 0%, rgba(0, 0, 0, 0.8) 100%)',
    zIndex: 1
  };

  const contentStyle = {
    position: 'relative',
    zIndex: 2,
    flex: 1,
    padding: '2rem',
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '1200px',
    margin: '0 auto',
    width: '100%'
  };

  return (
    <div style={containerStyle}>
      <video autoPlay muted loop playsInline style={videoStyle}>
        <source src="/videos/vid2.mp4" type="video/mp4" />
      </video>
      <div style={overlayStyle}></div>

      <div style={contentStyle}>
        {/* HEADER MEJORADO */}
        <header style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          marginBottom: '2rem',
          padding: '1rem 1.5rem',
          background: 'rgba(17, 17, 17, 0.8)',
          borderRadius: '12px',
          border: '1px solid rgba(243, 229, 171, 0.3)',
          backdropFilter: 'blur(10px)'
        }}>
          <button 
            className="exit-btn" 
            onClick={onBack}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              background: 'rgba(113, 54, 0, 0.3)',
              border: '1px solid #F3E5AB',
              color: '#F3E5AB',
              padding: '0.6rem 1.2rem',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'all 0.3s',
              fontWeight: '500'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = 'rgba(113, 54, 0, 0.5)';
              e.target.style.transform = 'translateX(-3px)';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'rgba(113, 54, 0, 0.3)';
              e.target.style.transform = 'translateX(0)';
            }}
          >
            <ArrowLeft size={16} /> Volver
          </button>
          
          <h1 className="logo" style={{ 
            fontSize: '1.8rem', 
            margin: 0,
            background: 'linear-gradient(135deg, #F3E5AB 0%, #FFD700 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontFamily: '"Orbitron", sans-serif',
            letterSpacing: '1px',
            textShadow: '0 2px 10px rgba(243, 229, 171, 0.3)'
          }}>
            DETALLES DEL PERFUME
          </h1>
          
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.5rem',
            color: 'rgba(243, 229, 171, 0.7)',
            fontSize: '0.9rem'
          }}>
            <Sparkles size={16} />
            <span>ESSENCE</span>
          </div>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', flex: 1 }}>
          
          {/* COLUMNA IZQUIERDA: INFORMACIÓN DEL PERFUME - MEJORADA */}
          <div className="card" style={{ 
            height: 'fit-content',
            background: 'rgba(17, 17, 17, 0.85)',
            borderRadius: '16px',
            border: '2px solid rgba(243, 229, 171, 0.3)',
            padding: '2rem',
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)',
            backdropFilter: 'blur(10px)'
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '1.5rem'
            }}>
              <div>
                <h2 style={{ 
                  color: '#F3E5AB', 
                  fontSize: '2.2rem', 
                  marginBottom: '0.2rem',
                  fontFamily: '"Orbitron", sans-serif',
                  letterSpacing: '0.5px',
                  textShadow: '0 2px 4px rgba(0,0,0,0.5)'
                }}>
                  {perfume.perfume}
                </h2>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.8rem',
                  color: 'rgba(243, 229, 171, 0.8)',
                  fontSize: '1rem'
                }}>
                  <span>🏷️ {perfume.marca}</span>
                  <span style={{
                    background: 'rgba(113, 54, 0, 0.3)',
                    padding: '0.2rem 0.8rem',
                    borderRadius: '20px',
                    fontSize: '0.85rem',
                    border: '1px solid rgba(243, 229, 171, 0.2)'
                  }}>
                    👤 {perfume.genero}
                  </span>
                </div>
              </div>
              
              <div style={{
                width: '60px',
                height: '60px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #713600 0%, #F3E5AB 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 15px rgba(113, 54, 0, 0.4)'
              }}>
                <Droplet size={28} color="#000" />
              </div>
            </div>

            {/* INFORMACIÓN DETALLADA */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(2, 1fr)', 
              gap: '1rem', 
              marginBottom: '2rem'
            }}>
              <div style={{ 
                background: 'rgba(113, 54, 0, 0.15)',
                padding: '1rem',
                borderRadius: '10px',
                border: '1px solid rgba(243, 229, 171, 0.1)'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem',
                  marginBottom: '0.5rem',
                  color: '#F3E5AB'
                }}>
                  <Calendar size={16} />
                  <strong>Año</strong>
                </div>
                <span style={{ color: 'rgba(255,255,255,0.9)' }}>
                  {perfume.año || 'Desconocido'}
                </span>
              </div>
              
              <div style={{ 
                background: 'rgba(113, 54, 0, 0.15)',
                padding: '1rem',
                borderRadius: '10px',
                border: '1px solid rgba(243, 229, 171, 0.1)'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem',
                  marginBottom: '0.5rem',
                  color: '#F3E5AB'
                }}>
                  <Award size={16} />
                  <strong>Perfumista</strong>
                </div>
                <span style={{ color: 'rgba(255,255,255,0.9)' }}>
                  {perfume.perfumista || 'N/A'}
                </span>
              </div>
            </div>

            {/* PIRÁMIDE OLFATIVA */}
            <div style={{ 
              borderTop: '2px solid rgba(243, 229, 171, 0.2)', 
              paddingTop: '1.5rem'
            }}>
              <h3 style={{ 
                color: '#FFD700', 
                marginBottom: '1.2rem',
                fontSize: '1.2rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontFamily: '"Orbitron", sans-serif',
                letterSpacing: '0.5px'
              }}>
                <Sparkles size={18} />
                Pirámide Olfativa
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.5rem',
                    marginBottom: '0.3rem'
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: '#FFD700'
                    }}></div>
                    <strong style={{ color: '#F3E5AB', fontSize: '0.95rem' }}>Notas de Salida</strong>
                  </div>
                  <p style={{ 
                    color: 'rgba(255,255,255,0.8)', 
                    margin: 0,
                    paddingLeft: '1.3rem',
                    fontSize: '0.9rem'
                  }}>
                    {perfume.salida || perfume.top || 'Información no disponible'}
                  </p>
                </div>
                
                <div>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.5rem',
                    marginBottom: '0.3rem'
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: '#F3E5AB'
                    }}></div>
                    <strong style={{ color: '#F3E5AB', fontSize: '0.95rem' }}>Notas de Corazón</strong>
                  </div>
                  <p style={{ 
                    color: 'rgba(255,255,255,0.8)', 
                    margin: 0,
                    paddingLeft: '1.3rem',
                    fontSize: '0.9rem'
                  }}>
                    {perfume.corazon || perfume.middle || 'Información no disponible'}
                  </p>
                </div>
                
                <div>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.5rem',
                    marginBottom: '0.3rem'
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: '#713600'
                    }}></div>
                    <strong style={{ color: '#F3E5AB', fontSize: '0.95rem' }}>Notas de Fondo</strong>
                  </div>
                  <p style={{ 
                    color: 'rgba(255,255,255,0.8)', 
                    margin: 0,
                    paddingLeft: '1.3rem',
                    fontSize: '0.9rem'
                  }}>
                    {perfume.base || 'Información no disponible'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* COLUMNA DERECHA: COMENTARIOS - MEJORADA */}
          <div className="card" style={{ 
            display: 'flex', 
            flexDirection: 'column',
            background: 'rgba(17, 17, 17, 0.85)',
            borderRadius: '16px',
            border: '2px solid rgba(243, 229, 171, 0.3)',
            padding: '2rem',
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)',
            backdropFilter: 'blur(10px)'
          }}>
            {/* HEADER COMENTARIOS */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '1.5rem',
              paddingBottom: '1rem',
              borderBottom: '2px solid rgba(243, 229, 171, 0.2)'
            }}>
              <h3 style={{ 
                color: '#FFD700', 
                fontSize: '1.3rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontFamily: '"Orbitron", sans-serif',
                letterSpacing: '0.5px'
              }}>
                <MessageSquare size={20} /> 
                Opiniones de la Comunidad
                <span style={{ 
                  background: 'rgba(113, 54, 0, 0.3)',
                  padding: '0.2rem 0.8rem',
                  borderRadius: '20px',
                  fontSize: '0.9rem',
                  marginLeft: '0.5rem',
                  border: '1px solid rgba(243, 229, 171, 0.2)'
                }}>
                  {comments.length}
                </span>
              </h3>
            </div>

            {/* LISTA DE COMENTARIOS */}
            <div style={{ 
              flex: 1, 
              overflowY: 'auto', 
              maxHeight: '450px', 
              marginBottom: '1.5rem',
              paddingRight: '10px'
            }}>
              {loading ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '3rem 1rem',
                  color: 'rgba(243, 229, 171, 0.6)'
                }}>
                  <div style={{ 
                    width: '50px', 
                    height: '50px', 
                    border: '3px solid rgba(243, 229, 171, 0.3)',
                    borderTop: '3px solid #F3E5AB',
                    borderRadius: '50%',
                    margin: '0 auto 1rem',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  Cargando opiniones...
                </div>
              ) : comments.length === 0 ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '3rem 1rem',
                  background: 'rgba(113, 54, 0, 0.1)',
                  borderRadius: '12px',
                  border: '2px dashed rgba(243, 229, 171, 0.2)'
                }}>
                  <div style={{ 
                    fontSize: '3rem',
                    marginBottom: '1rem',
                    opacity: 0.5
                  }}>
                    💬
                  </div>
                  <p style={{ 
                    color: 'rgba(243, 229, 171, 0.8)', 
                    fontSize: '1.1rem',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Aún no hay opiniones
                  </p>
                  <p style={{ 
                    color: 'rgba(243, 229, 171, 0.6)', 
                    fontSize: '0.9rem',
                    margin: 0
                  }}>
                    ¡Sé el primero en compartir tu experiencia!
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {comments.map((c, i) => (
                    <div key={i} style={{ 
                      background: 'rgba(113, 54, 0, 0.15)',
                      padding: '1.2rem',
                      borderRadius: '12px',
                      border: '1px solid rgba(243, 229, 171, 0.15)',
                      transition: 'all 0.3s'
                    }}>
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'flex-start',
                        marginBottom: '0.8rem'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                          <div style={{
                            width: '40px',
                            height: '40px',
                            borderRadius: '50%',
                            background: 'linear-gradient(135deg, #713600 0%, #F3E5AB 100%)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: '#000',
                            fontWeight: 'bold',
                            fontSize: '1rem'
                          }}>
                            {c.nombre.charAt(0)}{c.apellido.charAt(0)}
                          </div>
                          <div>
                            <strong style={{ 
                              color: '#F3E5AB', 
                              fontSize: '1rem',
                              display: 'block'
                            }}>
                              {c.nombre} {c.apellido}
                            </strong>
                            <small style={{ 
                              color: 'rgba(243, 229, 171, 0.6)', 
                              fontSize: '0.8rem',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.3rem'
                            }}>
                              <Calendar size={12} />
                              {new Date(c.fecha_creacion).toLocaleDateString('es-ES', {
                                day: 'numeric',
                                month: 'long',
                                year: 'numeric'
                              })}
                            </small>
                          </div>
                        </div>
                        
                        <div style={{ 
                          display: 'flex', 
                          alignItems: 'center',
                          gap: '0.3rem',
                          background: 'rgba(0,0,0,0.3)',
                          padding: '0.4rem 0.8rem',
                          borderRadius: '20px'
                        }}>
                          <Star size={16} fill="#FFD700" color="#FFD700" />
                          <span style={{ 
                            color: '#FFD700', 
                            fontSize: '1rem',
                            fontWeight: 'bold'
                          }}>
                            {c.puntuacion}/5
                          </span>
                        </div>
                      </div>
                      
                      <p style={{ 
                        color: 'rgba(255,255,255,0.9)', 
                        margin: 0, 
                        fontSize: '0.95rem',
                        lineHeight: '1.6',
                        paddingLeft: '3.2rem'
                      }}>
                        {c.texto}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* FORMULARIO DE COMENTARIO */}
            {user ? (
              <form onSubmit={handleSubmit} style={{ 
                background: 'rgba(113, 54, 0, 0.1)',
                padding: '1.5rem',
                borderRadius: '12px',
                border: '1px solid rgba(243, 229, 171, 0.2)'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  marginBottom: '1rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#F3E5AB' }}>
                    <span>Tu puntuación:</span>
                    <div style={{ display: 'flex', gap: '0.2rem' }}>
                      {[1, 2, 3, 4, 5].map((s) => (
                        <Star 
                          key={s} 
                          size={24} 
                          fill={s <= rating ? "#FFD700" : "rgba(243, 229, 171, 0.2)"} 
                          color={s <= rating ? "#FFD700" : "rgba(243, 229, 171, 0.5)"} 
                          style={{ 
                            cursor: 'pointer',
                            transition: 'transform 0.2s'
                          }}
                          onMouseEnter={(e) => e.target.style.transform = 'scale(1.2)'}
                          onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                          onClick={() => setRating(s)}
                        />
                      ))}
                    </div>
                  </div>
                  
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.5rem',
                    color: 'rgba(243, 229, 171, 0.7)',
                    fontSize: '0.9rem'
                  }}>
                    <User size={16} />
                    {user.name} {user.lastname}
                  </div>
                </div>
                
                <div style={{ display: 'flex', gap: '0.8rem' }}>
                  <input
                    type="text"
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="¿Qué opinas de este perfume? Comparte tu experiencia..."
                    style={{ 
                      flex: 1, 
                      padding: '1rem', 
                      borderRadius: '10px', 
                      border: '1px solid rgba(243, 229, 171, 0.3)',
                      background: 'rgba(0, 0, 0, 0.4)', 
                      color: '#fff',
                      fontSize: '0.95rem',
                      transition: 'all 0.3s'
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = '#F3E5AB';
                      e.target.style.boxShadow = '0 0 0 3px rgba(243, 229, 171, 0.2)';
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = 'rgba(243, 229, 171, 0.3)';
                      e.target.style.boxShadow = 'none';
                    }}
                  />
                  <button 
                    type="submit" 
                    disabled={isSubmitting || !commentText.trim()}
                    style={{ 
                      background: 'linear-gradient(135deg, #713600 0%, #F3E5AB 100%)', 
                      border: 'none', 
                      borderRadius: '10px', 
                      width: '55px',
                      height: '55px',
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      cursor: commentText.trim() && !isSubmitting ? 'pointer' : 'not-allowed',
                      opacity: commentText.trim() ? 1 : 0.5,
                      transition: 'all 0.3s',
                      boxShadow: '0 4px 15px rgba(113, 54, 0, 0.3)'
                    }}
                    onMouseEnter={(e) => {
                      if (commentText.trim() && !isSubmitting) {
                        e.target.style.transform = 'translateY(-2px)';
                        e.target.style.boxShadow = '0 6px 20px rgba(113, 54, 0, 0.4)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (commentText.trim() && !isSubmitting) {
                        e.target.style.transform = 'translateY(0)';
                        e.target.style.boxShadow = '0 4px 15px rgba(113, 54, 0, 0.3)';
                      }
                    }}
                  >
                    {isSubmitting ? (
                      <div style={{
                        width: '20px',
                        height: '20px',
                        border: '2px solid #000',
                        borderTop: '2px solid transparent',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite'
                      }}></div>
                    ) : (
                      <Send size={22} color="#000" />
                    )}
                  </button>
                </div>
                
                {/* MENSAJES DE ERROR/ÉXITO */}
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  marginTop: '0.8rem',
                  minHeight: '24px'
                }}>
                  {error && (
                    <span style={{ 
                      color: '#ff6b6b', 
                      fontSize: '0.85rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem'
                    }}>
                      ⚠️ {error}
                    </span>
                  )}
                  
                  {success && (
                    <span style={{ 
                      color: '#4CAF50', 
                      fontSize: '0.85rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem'
                    }}>
                      ✅ Comentario enviado con éxito
                    </span>
                  )}
                </div>
              </form>
            ) : (
              <div style={{ 
                textAlign: 'center', 
                padding: '2rem', 
                background: 'rgba(113, 54, 0, 0.15)',
                borderRadius: '12px', 
                border: '2px dashed rgba(243, 229, 171, 0.3)'
              }}>
                <div style={{ 
                  fontSize: '2.5rem',
                  marginBottom: '1rem'
                }}>
                  🔒
                </div>
                <p style={{ 
                  color: 'rgba(243, 229, 171, 0.9)', 
                  fontSize: '1.1rem',
                  marginBottom: '0.5rem',
                  fontWeight: '500'
                }}>
                  Inicia sesión para comentar
                </p>
                <p style={{ 
                  color: 'rgba(243, 229, 171, 0.7)', 
                  fontSize: '0.9rem',
                  margin: 0
                }}>
                  Únete a la comunidad ESSENCE para compartir tu opinión
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Animación CSS */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        /* Scrollbar personalizado */
        div::-webkit-scrollbar {
          width: 8px;
        }
        
        div::-webkit-scrollbar-track {
          background: rgba(113, 54, 0, 0.1);
          border-radius: 4px;
        }
        
        div::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #713600, #F3E5AB);
          border-radius: 4px;
        }
        
        div::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #F3E5AB, #713600);
        }
      `}</style>
    </div>
  );
};

export default PerfumeDetailScreen;