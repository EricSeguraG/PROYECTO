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
const fetchPython = async (endpoint) => {
  try {
    console.log(`🚀 Llamando a Flask API: ${FLASK_API}${endpoint}`);
    
    const response = await fetch(`${FLASK_API}${endpoint}`);
    
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
      has_similares: !!(data.similares),
      similares_count: data.similares ? data.similares.length : 0
    });
    
    return replaceHyphensInData(data);
  } catch (error) {
    console.error('❌ Error fetching Python:', error);
    return {};
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
     return {}; 
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