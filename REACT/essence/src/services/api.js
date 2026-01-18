// src/services/api.js

// 🟢 NODE.JS (Usuarios y Comentarios - MySQL)
const NODE_API = 'http://localhost:3001';

// 🔵 PYTHON FLASK (Perfumes, Buscador y Clones - CSV)
const FLASK_API = 'http://localhost:5000';

// ==========================================
// 🟣 WISHLIST Y COLECCIÓN (NUEVAS FUNCIONES)
// ==========================================

// Helper para obtener el token de autenticación
const getToken = () => {
  return localStorage.getItem('token') || sessionStorage.getItem('token');
};

// Helper para construir headers con auth
const getAuthHeaders = () => {
  const token = getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
};

// ==========================================
// WISHLIST FUNCTIONS
// ==========================================

// Añadir a wishlist
const addToWishlist = async (wishlistData) => {
  try {
    const response = await fetch(`${NODE_API}/wishlist`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(wishlistData)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Error al añadir a wishlist');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error en addToWishlist:', error);
    throw error;
  }
};

// Eliminar de wishlist
const removeFromWishlist = async (usuarioId, perfumeId) => {
  try {
    const response = await fetch(`${NODE_API}/wishlist/${usuarioId}/${perfumeId}`, {
      method: 'DELETE',
      headers: getAuthHeaders()
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Error al eliminar de wishlist');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error en removeFromWishlist:', error);
    throw error;
  }
};

// Verificar si está en wishlist
const checkInWishlist = async (usuarioId, perfumeId) => {
  try {
    const response = await fetch(`${NODE_API}/wishlist/check/${usuarioId}/${perfumeId}`, {
      headers: getAuthHeaders()
    });
    
    if (!response.ok) {
      return { exists: false };
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error en checkInWishlist:', error);
    return { exists: false };
  }
};

// Obtener wishlist del usuario
const getWishlist = async (usuarioId) => {
  try {
    const response = await fetch(`${NODE_API}/wishlist/user/${usuarioId}`, {
      headers: getAuthHeaders()
    });
    
    if (!response.ok) {
      return [];
    }
    
    const data = await response.json();
    return data || [];
  } catch (error) {
    console.error('Error en getWishlist:', error);
    return [];
  }
};

// ==========================================
// COLECCIÓN FUNCTIONS
// ==========================================

// Añadir a colección
const addToCollection = async (collectionData) => {
  try {
    const response = await fetch(`${NODE_API}/collection`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(collectionData)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Error al añadir a colección');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error en addToCollection:', error);
    throw error;
  }
};

// Eliminar de colección
const removeFromCollection = async (usuarioId, perfumeId) => {
  try {
    const response = await fetch(`${NODE_API}/collection/${usuarioId}/${perfumeId}`, {
      method: 'DELETE',
      headers: getAuthHeaders()
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Error al eliminar de colección');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error en removeFromCollection:', error);
    throw error;
  }
};

// Verificar si está en colección
const checkInCollection = async (usuarioId, perfumeId) => {
  try {
    const response = await fetch(`${NODE_API}/collection/check/${usuarioId}/${perfumeId}`, {
      headers: getAuthHeaders()
    });
    
    if (!response.ok) {
      return { exists: false };
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error en checkInCollection:', error);
    return { exists: false };
  }
};

// Obtener colección del usuario
const getCollection = async (usuarioId) => {
  try {
    const response = await fetch(`${NODE_API}/collection/user/${usuarioId}`, {
      headers: getAuthHeaders()
    });
    
    if (!response.ok) {
      return [];
    }
    
    const data = await response.json();
    return data || [];
  } catch (error) {
    console.error('Error en getCollection:', error);
    return [];
  }
};

// ==========================================
// 🔧 FUNCIONES EXISTENTES (NO MODIFICAR)
// ==========================================

// Función para limpiar guiones visualmente
const replaceHyphensInData = (data) => {
  if (typeof data === 'string') return data.replace(/-/g, ' ');
  if (Array.isArray(data)) return data.map(item => replaceHyphensInData(item));
  if (typeof data === 'object' && data !== null) {
    const newObj = {};
    for (const key in data) newObj[key] = replaceHyphensInData(data[key]);
    return newObj;
  }
  return data;
};

// --- 🟡 FUNCIÓN PARA CARGAR CELEBRITIES (LOCAL) ---
const loadCelebritiesDB = async () => {
  try {
    const module = await import('../data/celebritiesDB.js');
    const db = module.default || module.celebritiesDB;
    return replaceHyphensInData(db || []);
  } catch (error) {
    console.error("⚠️ No se encontró celebritiesDB.js en src/data/", error);
    return [];
  }
};

// Wrapper para llamadas a PYTHON
const fetchPython = async (endpoint, options = {}) => {
  try {
    console.log(`🚀 Llamando a Flask API: ${FLASK_API}${endpoint}`);
    
    const response = await fetch(`${FLASK_API}${endpoint}`, {
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {})
      },
      body: options.body ? JSON.stringify(options.body) : undefined
    });
    
    console.log(`📊 Respuesta de Flask: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Error en Flask API (${response.status}):`, errorText);
      throw new Error(`Error Python API: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Datos recibidos de Flask:', {
      endpoint: endpoint,
      data_keys: Object.keys(data),
      data_length: Array.isArray(data) ? data.length : Object.keys(data).length
    });
    
    return replaceHyphensInData(data);
  } catch (error) {
    console.error('❌ Error fetching Python:', error);
    return options.returnOnError || {};
  }
};

// Wrapper para llamadas a NODE.JS
const fetchNode = async (endpoint, method = 'GET', body = null) => {
  const options = {
    method,
    headers: { 'Content-Type': 'application/json' }
  };
  if (body) options.body = JSON.stringify(body);

  const response = await fetch(`${NODE_API}${endpoint}`, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Error en Node API');
  return data;
};

// ==========================================
// 🏆 FUNCIÓN PARA PERFUMES MÁS VOTADOS (ACTUALIZADA)
// ==========================================

const getTopRatedPerfumesAPI = async (params = {}) => {
  try {
    console.log('🎯 Iniciando getTopRatedPerfumesAPI (TOP 10)...');
    
    // USAR SIEMPRE EL ENDPOINT DE NODE.JS
    console.log('1. Llamando a /perfumes/top-rated en Node.js...');
    const response = await fetch(`${NODE_API}/perfumes/top-rated`);
    
    console.log('📡 Status de respuesta:', response.status, response.statusText);
    
    if (response.ok) {
      const data = await response.json();
      console.log(`✅ ${data.length} perfumes TOP 10 obtenidos de Node.js`);
      
      if (data && data.length > 0) {
        // Asegurar formato consistente
        const formattedData = data.map(perfume => ({
          id: perfume.id?.toString() || `node-${Date.now()}`,
          perfume: perfume.perfume || perfume.nombre || 'Sin nombre',
          nombre: perfume.nombre || perfume.perfume || 'Sin nombre',
          marca: perfume.marca || 'Marca desconocida',
          genero: perfume.genero || 'Unisex',
          año: perfume.año || 'N/A',
          avg_rating: parseFloat(perfume.avg_rating || perfume.average_rating || 0),
          average_rating: parseFloat(perfume.average_rating || perfume.avg_rating || 0),
          total_votes: parseInt(perfume.total_votes || 0),
          total_comments: parseInt(perfume.total_comments || 0),
          vote_count: parseInt(perfume.vote_count || 0),
          main_accords: Array.isArray(perfume.main_accords) ? 
            perfume.main_accords : 
            (perfume.main_accords ? [perfume.main_accords] : []),
          salida: perfume.salida || perfume.top_notes || '',
          corazon: perfume.corazon || perfume.heart_notes || '',
          base: perfume.base || perfume.base_notes || '',
          url: perfume.url || '',
          top_notes: perfume.top_notes || perfume.salida || '',
          heart_notes: perfume.heart_notes || perfume.corazon || '',
          base_notes: perfume.base_notes || perfume.base || ''
        }));
        
        console.log('📊 Primer perfume TOP 10:', formattedData[0]);
        return formattedData;
      }
    }
    
    // Si falla, devolver array vacío
    console.log('⚠️ No se pudieron obtener perfumes TOP 10');
    return [];
    
  } catch (error) {
    console.error('❌ Error en getTopRatedPerfumes:', error);
    return [];
  }
};

// Función combinada para obtener perfumes con estadísticas
const getCombinedTopRated = async (params) => {
  try {
    console.log('🔍 Obteniendo comentarios agrupados...');
    
    // Obtener comentarios agrupados de Node.js
    const commentsResponse = await fetch(`${NODE_API}/comentarios/all/grouped`);
    
    if (!commentsResponse || !commentsResponse.ok) {
      console.log('⚠️ No se pudieron obtener comentarios agrupados');
      return [];
    }
    
    const commentsData = await commentsResponse.json();
    console.log(`📊 ${commentsData.length} grupos de comentarios obtenidos`);
    
    if (commentsData.length === 0) {
      console.log('ℹ️ No hay comentarios en la base de datos');
      return [];
    }
    
    // Mostrar primeros comentarios para debug
    console.log('Primeros 3 grupos de comentarios:', commentsData.slice(0, 3));
    
    // Obtener detalles de perfumes de Flask
    const perfumesWithStats = [];
    const limit = Math.min(params.limit || 50, commentsData.length);
    
    for (let i = 0; i < limit; i++) {
      const commentGroup = commentsData[i];
      console.log(`🔍 Buscando perfume: "${commentGroup.perfume_name}"`);
      
      try {
        // Buscar perfume por nombre en Flask
        const flaskResponse = await fetch(
          `${FLASK_API}/perfumes/search?perfume=${encodeURIComponent(commentGroup.perfume_name)}&limit=1`
        );
        
        if (flaskResponse && flaskResponse.ok) {
          const flaskData = await flaskResponse.json();
          
          if (flaskData.resultados && flaskData.resultados.length > 0) {
            const perfume = flaskData.resultados[0];
            console.log(`✅ Encontrado en Flask: ${perfume.nombre}`);
            
            perfumesWithStats.push({
              ...perfume,
              id: perfume.id || `combined-${i}`,
              perfume: perfume.nombre || commentGroup.perfume_name,
              nombre: perfume.nombre || commentGroup.perfume_name,
              total_votes: parseInt(commentGroup.comment_count || 0),
              total_comments: parseInt(commentGroup.comment_count || 0),
              vote_count: parseInt(commentGroup.comment_count || 0),
              avg_rating: parseFloat(commentGroup.average_rating || 0),
              average_rating: parseFloat(commentGroup.average_rating || 0)
            });
          } else {
            console.log(`⚠️ No se encontró en Flask: "${commentGroup.perfume_name}"`);
          }
        }
      } catch (error) {
        console.log(`⚠️ Error buscando perfume "${commentGroup.perfume_name}":`, error.message);
      }
    }
    
    console.log(`✅ ${perfumesWithStats.length} perfumes combinados obtenidos`);
    
    if (perfumesWithStats.length === 0) {
      console.log('ℹ️ No se pudieron obtener perfumes de Flask');
      return [];
    }
    
    // Ordenar por rating descendente
    const sorted = perfumesWithStats.sort((a, b) => b.average_rating - a.average_rating);
    return sorted;
    
  } catch (error) {
    console.error('❌ Error en getCombinedTopRated:', error);
    return [];
  }
};

// Función simple alternativa (para desarrollo)
const getTopRatedSimpleAPI = async (params = {}) => {
  try {
    console.log('🔄 Usando método simple para desarrollo...');
    
    // Obtener algunos perfumes de Flask
    const response = await fetch(`${FLASK_API}/perfumes/search?limit=20`);
    
    if (!response.ok) {
      console.log('⚠️ Flask no responde');
      return [];
    }
    
    const data = await response.json();
    const perfumes = data.resultados || [];
    
    console.log(`📊 ${perfumes.length} perfumes obtenidos de Flask`);
    
    if (perfumes.length > 0) {
      // Añadir ratings y votos simulados
      return perfumes.map((perfume, index) => ({
        ...perfume,
        id: perfume.id || `simple-${index}`,
        perfume: perfume.nombre || perfume.perfume || `Perfume ${index + 1}`,
        nombre: perfume.nombre || perfume.perfume || `Perfume ${index + 1}`,
        marca: perfume.marca || 'Marca desconocida',
        genero: perfume.genero || 'Unisex',
        año: perfume.año || 'N/A',
        total_votes: Math.floor(Math.random() * 500) + 100,
        total_comments: Math.floor(Math.random() * 100) + 10,
        avg_rating: 4.8 - (index * 0.1),
        average_rating: 4.8 - (index * 0.1),
        vote_count: Math.floor(Math.random() * 500) + 100,
        main_accords: Array.isArray(perfume.main_accords) ? 
          perfume.main_accords : 
          (perfume.main_accords ? [perfume.main_accords] : []),
        salida: perfume.salida || perfume.top_notes || '',
        corazon: perfume.corazon || perfume.heart_notes || '',
        base: perfume.base || perfume.base_notes || '',
        url: perfume.url || '',
        top_notes: perfume.top_notes || perfume.salida || '',
        heart_notes: perfume.heart_notes || perfume.corazon || '',
        base_notes: perfume.base_notes || perfume.base || ''
      })).sort((a, b) => b.avg_rating - a.avg_rating);
    }
    
    return [];
    
  } catch (error) {
    console.error('❌ Error en getTopRatedSimple:', error);
    return [];
  }
};

// ==========================================
// 🟢 API COMPLETA
// ==========================================

export const perfumeAPI = {

  // ==========================================
  //  🟢 USUARIOS Y COMENTARIOS (NODE - MySQL)
  // ==========================================

  register: async (userData) => {
    return await fetchNode('/register', 'POST', userData);
  },

  login: async (credentials) => {
    return await fetchNode('/login', 'POST', credentials);
  },

  getComments: async (perfumeId) => {
    try {
      const response = await fetch(`${NODE_API}/comentarios/${perfumeId}`);
      if (!response.ok) return [];
      return await response.json();
    } catch (error) {
      return [];
    }
  },

  addComment: async (commentData) => {
    return await fetchNode('/comentarios', 'POST', commentData);
  },

  // ==========================================
  //  🏆 NUEVA: PERFUMES MÁS VOTADOS (ACTUALIZADA)
  // ==========================================
  
  getTopRatedPerfumes: async (params = {}) => {
    console.log('🎯 perfumeAPI.getTopRatedPerfumes llamado');
    const result = await getTopRatedPerfumesAPI(params);
    console.log(`🎯 Resultado: ${result.length} perfumes`);
    return result;
  },
  
  getTopRated: async (params = {}) => {
    return await getTopRatedSimpleAPI(params);
  },

  // ==========================================
  //  🟣 WISHLIST Y COLECCIÓN (NUEVO)
  // ==========================================

  // Wishlist
  addToWishlist: async (wishlistData) => {
    return await addToWishlist(wishlistData);
  },

  removeFromWishlist: async (usuarioId, perfumeId) => {
    return await removeFromWishlist(usuarioId, perfumeId);
  },

  checkInWishlist: async (usuarioId, perfumeId) => {
    return await checkInWishlist(usuarioId, perfumeId);
  },

  getWishlist: async (usuarioId) => {
    return await getWishlist(usuarioId);
  },

  // Colección
  addToCollection: async (collectionData) => {
    return await addToCollection(collectionData);
  },

  removeFromCollection: async (usuarioId, perfumeId) => {
    return await removeFromCollection(usuarioId, perfumeId);
  },

  checkInCollection: async (usuarioId, perfumeId) => {
    return await checkInCollection(usuarioId, perfumeId);
  },

  getCollection: async (usuarioId) => {
    return await getCollection(usuarioId);
  },

  // ==========================================
  //  🔵 PERFUMES Y DATOS (PYTHON - CSV)
  // ==========================================

  getBrands: async () => {
    const data = await fetchPython('/perfumes/marcas');
    return data.marcas || [];
  },

  getPerfumesByBrand: async (brandName, page = 1) => {
    const data = await fetchPython(`/perfumes/marca/${encodeURIComponent(brandName)}?pagina=${page}`);
    return data.perfumes || [];
  },

  searchPerfumes: async (query, filters = {}) => {
    const params = new URLSearchParams();
    if (query) params.append('perfume', query);
    if (filters.perfume) params.append('perfume', filters.perfume);
    if (filters.marca) params.append('marca', filters.marca);
    if (filters.genero) params.append('genero', filters.genero);
    if (filters.nota) params.append('nota', filters.nota);
    if (filters.acorde) params.append('acorde', filters.acorde);
    if (filters.perfumista) params.append('perfumista', filters.perfumista);
    if (filters.sort) params.append('sort', filters.sort);
    if (filters.order) params.append('order', filters.order);
    
    const data = await fetchPython(`/perfumes/search?${params}`);
    return data.resultados || [];
  },

  searchClones: async (perfumeName, similarityThreshold = 70) => {
    console.log('📡 API: Llamando a /perfumes/similares con:', {
      perfumeName,
      similarityThreshold
    });
    
    const data = await fetchPython(
      `/perfumes/similares?nombre=${encodeURIComponent(perfumeName)}&umbral=${similarityThreshold}`
    );
    
    console.log('📦 API: Datos recibidos de Flask:', {
      similares_recibidos: data.similares ? data.similares.length : 0,
      umbral_en_api: data.umbral || 'No especificado'
    });
    
    return data.similares || [];
  },

  // ==========================================
  //  🟡 CELEBRITIES (ARCHIVO LOCAL)
  // ==========================================
  
  searchByCelebrity: async (celebrityName) => {
    try {
       const db = await loadCelebritiesDB();
       
       if (!celebrityName) return [];
       const term = celebrityName.toLowerCase().trim();
       
       const results = db.filter(c => 
         c.celebrity && c.celebrity.toLowerCase().includes(term)
       );
       
       return results;
    } catch (error) {
       console.error("Error buscando celebrity:", error);
       return [];
    }
  },
  
  getPerfumeDetails: async (perfumeId) => {
    try {
      // Primero intentar con Node.js (si tiene detalles)
      const response = await fetch(`${NODE_API}/perfumes/${perfumeId}`);
      if (response.ok) {
        const data = await response.json();
        if (data && Object.keys(data).length > 0) {
          return data;
        }
      }
      
      // Si no, intentar con Flask
      const flaskData = await fetchPython(`/perfumes/${perfumeId}`);
      return flaskData || {};
    } catch (error) {
      console.error('Error obteniendo detalles del perfume:', error);
      return {};
    }
  },
  
  // Nueva función para obtener todos los perfumes
  getAllPerfumes: async () => {
    try {
      const data = await fetchPython('/perfumes/all');
      return data || [];
    } catch (error) {
      console.error('Error obteniendo todos los perfumes:', error);
      return [];
    }
  }
};

// Helper para verificar autenticación
export const isAuthenticated = () => {
  return !!getToken();
};

// Helper para obtener datos del usuario desde localStorage/sessionStorage
export const getCurrentUser = () => {
  const userData = localStorage.getItem('user') || sessionStorage.getItem('user');
  return userData ? JSON.parse(userData) : null;
};