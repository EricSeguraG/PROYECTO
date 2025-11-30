import React, { useState, useEffect } from 'react';
import { X, Star, Send } from 'lucide-react';
import { perfumeAPI } from '../services/api';

const PerfumeDetailModal = ({ perfume, user, onClose }) => {
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState("");
  const [rating, setRating] = useState(5);
  const [loading, setLoading] = useState(true);

  // Cargar comentarios al abrir
  useEffect(() => {
    if (perfume?.id) {
      loadComments();
    }
  }, [perfume]);

  const loadComments = async () => {
    setLoading(true);
    const data = await perfumeAPI.getComments(perfume.id);
    setComments(data);
    setLoading(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!newComment.trim()) return;

    try {
      await perfumeAPI.addComment({
        usuario_id: user.id,
        perfume_id: perfume.id,
        texto: newComment,
        puntuacion: rating
      });
      setNewComment("");
      loadComments(); // Recargar lista
    } catch (error) {
      alert("Error al enviar comentario: " + error.message);
    }
  };

  // --- ESTILOS ---
  const overlayStyle = {
    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    display: 'flex', justifyContent: 'center', alignItems: 'center',
    zIndex: 1000, padding: '1rem'
  };

  const modalStyle = {
    backgroundColor: '#1a1a1a',
    border: '1px solid #F3E5AB',
    borderRadius: '8px',
    width: '100%', maxWidth: '500px', maxHeight: '90vh',
    display: 'flex', flexDirection: 'column',
    position: 'relative', boxShadow: '0 0 20px rgba(243, 229, 171, 0.2)'
  };

  if (!perfume) return null;

  return (
    <div style={overlayStyle}>
      <div style={modalStyle}>
        
        {/* HEADER */}
        <div style={{ padding: '1rem', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ color: '#F3E5AB', margin: 0, fontSize: '1.2rem' }}>{perfume.perfume}</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#F3E5AB', cursor: 'pointer' }}>
            <X size={24} />
          </button>
        </div>

        {/* CONTENIDO SCROLLEABLE */}
        <div style={{ padding: '1rem', overflowY: 'auto' }}>
          
          {/* INFO DEL PERFUME */}
          <div style={{ marginBottom: '2rem', color: '#ccc' }}>
            <p><strong>Marca:</strong> {perfume.marca}</p>
            <p><strong>Género:</strong> {perfume.genero}</p>
            {perfume.año && <p><strong>Año:</strong> {perfume.año}</p>}
            {perfume.perfumista && <p><strong>Perfumista:</strong> {perfume.perfumista}</p>}
          </div>

          {/* SECCIÓN COMENTARIOS */}
          <h3 style={{ color: '#F3E5AB', fontSize: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>
            Comentarios ({comments.length})
          </h3>

          <div style={{ maxHeight: '200px', overflowY: 'auto', marginBottom: '1rem' }}>
            {loading ? (
              <p style={{ color: '#666' }}>Cargando opiniones...</p>
            ) : comments.length === 0 ? (
              <p style={{ color: '#666', fontStyle: 'italic' }}>Sé el primero en opinar.</p>
            ) : (
              comments.map((c) => (
                <div key={c.id} style={{ backgroundColor: '#2a2a2a', padding: '0.8rem', borderRadius: '4px', marginBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                    <strong style={{ color: '#F3E5AB', fontSize: '0.9rem' }}>{c.username}</strong>
                    <span style={{ color: '#F3E5AB' }}>★ {c.puntuacion}</span>
                  </div>
                  <p style={{ color: '#ddd', margin: 0, fontSize: '0.9rem' }}>{c.texto}</p>
                </div>
              ))
            )}
          </div>

          {/* FORMULARIO PARA AGREGAR */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fff' }}>
              <span>Tu puntuación:</span>
              {[1, 2, 3, 4, 5].map((star) => (
                <Star 
                  key={star} 
                  size={20} 
                  fill={star <= rating ? "#F3E5AB" : "none"} 
                  color={star <= rating ? "#F3E5AB" : "#666"}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setRating(star)}
                />
              ))}
            </div>
            
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                placeholder="Escribe tu opinión..."
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                style={{ 
                  flex: 1, padding: '0.8rem', borderRadius: '4px', border: 'none', 
                  backgroundColor: '#333', color: 'white' 
                }}
              />
              <button 
                type="submit" 
                disabled={!newComment.trim()}
                style={{ 
                  background: '#F3E5AB', border: 'none', borderRadius: '4px', 
                  width: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: newComment.trim() ? 'pointer' : 'not-allowed', opacity: newComment.trim() ? 1 : 0.5
                }}
              >
                <Send size={20} color="#000" />
              </button>
            </div>
          </form>

        </div>
      </div>
    </div>
  );
};

export default PerfumeDetailModal;