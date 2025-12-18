// src/services/api.js

// 🟢 NODE.JS (Usuarios y Comentarios - MySQL)
const NODE_API = 'http://localhost:3001';

// 🔵 PYTHON FLASK (Perfumes, Buscador y Clones - CSV)
const FLASK_API = 'http://localhost:5000';

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
    // Intenta cargar el archivo local que tenías antes
    const module = await import('../data/celebritiesDB.js');
    const db = module.default || module.celebritiesDB;
    return replaceHyphensInData(db || []);
  } catch (error) {
    console.error("⚠️ No se encontró celebritiesDB.js en src/data/", error);
    return [];
  }
};

// Wrapper para llamadas a PYTHON
// Modifica la función fetchPython para incluir logs:
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
    return {}; // Cambié de [] a {} porque algunas respuestas son objetos
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
  //  🔵 PERFUMES Y DATOS (PYTHON - CSV)
  // ==========================================

  getBrands: async () => {
    const data = await fetchPython('/perfumes/marcas');
    return data.marcas || [];
  },

  getPerfumesByBrand: async (brandName, page = 1) => {
    // Usamos encodeURIComponent para que funcione con "Tom Ford" o "Tom-Ford"
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
       // Cargamos la base de datos local
       const db = await loadCelebritiesDB();
       
       if (!celebrityName) return [];
       const term = celebrityName.toLowerCase().trim();
       
       // Filtramos localmente como hacías antes
       const results = db.filter(c => 
         c.celebrity && c.celebrity.toLowerCase().includes(term)
       );
       
       return results;
    } catch (error) {
       console.error("Error buscando celebrity:", error);
       return [];
    }
  },
  
  // Detalle del perfume (Python)
  getPerfumeDetails: async (perfumeId) => {
     // Si necesitas buscar por ID, Python tendría que tener una ruta para esto.
     // Si no, puedes confiar en que la info ya viene cargada.
     return {}; 
  }
};