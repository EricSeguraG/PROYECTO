// src/services/api.js
const API_BASE_URL = 'http://localhost:5000';

// Función para reemplazar guiones por espacios en cualquier dato (solo para visualización)
const replaceHyphensInData = (data) => {
  if (typeof data === 'string') {
    return data.replace(/-/g, ' ');
  }
  
  if (Array.isArray(data)) {
    return data.map(item => replaceHyphensInData(item));
  }
  
  if (typeof data === 'object' && data !== null) {
    const newObj = {};
    for (const key in data) {
      newObj[key] = replaceHyphensInData(data[key]);
    }
    return newObj;
  }
  
  return data;
};

// Función wrapper para fetch que aplica el reemplazo de guiones SOLO para visualización
const fetchWithHyphenReplacement = async (url, options = {}) => {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      // Intentamos parsear el error si es JSON
      try {
          const errorJson = JSON.parse(errorText);
          throw new Error(errorJson.error || `Error ${response.status}: ${response.statusText}`);
      } catch (e) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
    }
    
    const data = await response.json();
    const transformedData = replaceHyphensInData(data);
    
    return transformedData;
  } catch (error) {
    console.error('❌ Error en fetch:', error);
    throw error;
  }
};

// Función para cargar la base de datos de celebrities (Local)
const loadCelebritiesDB = async () => {
  try {
    console.log('📁 Cargando base de datos de celebrities...');
    
    const module = await import('../data/celebritiesDB.js');
    const celebritiesData = module.default || module.celebritiesDB;
    
    if (celebritiesData && Array.isArray(celebritiesData)) {
      console.log(`✅ Base de datos cargada: ${celebritiesData.length} celebridades`);
      return replaceHyphensInData(celebritiesData);
    } else {
      throw new Error('Formato inválido');
    }
  } catch (error) {
    console.error('❌ Error cargando celebritiesDB.js:', error);
    return [];
  }
};

export const perfumeAPI = {

  // ==========================================
  //  NUEVAS FUNCIONES DE USUARIO (BACKEND)
  // ==========================================

  // --- REGISTRO DE USUARIO ---
  register: async (userData) => {
    try {
      console.log('👤 Registrando usuario:', userData.username);
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Error en el registro');
      }
      
      console.log('✅ Usuario registrado con éxito');
      return data;
    } catch (error) {
      console.error('❌ Error en register:', error);
      throw error;
    }
  },

  // --- LOGIN DE USUARIO ---
  login: async (credentials) => {
    try {
      console.log('🔑 Iniciando sesión:', credentials.username);
      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Error en el inicio de sesión');
      }

      console.log('✅ Login exitoso');
      return data;
    } catch (error) {
      console.error('❌ Error en login:', error);
      throw error;
    }
  },

  // ==========================================
  //  FUNCIONES EXISTENTES DE PERFUMES
  // ==========================================

  // --- BÚSQUEDA DE CLONES ---
  searchClones: async (perfumeName, similarityThreshold = 70) => {
    try {
      const url = `${API_BASE_URL}/perfumes/similares?nombre=${encodeURIComponent(perfumeName)}`;
      console.log('🌐 URL de clones:', url);
      
      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Datos de clones recibidos:', data);
      
      if (!data.similares) {
        console.warn('⚠️ No hay campo "similares" en la respuesta');
        return [];
      }
      
      // Filtrar por similitud
      if (similarityThreshold) {
        const filtered = data.similares.filter(perfume => {
          const similarityMatch = (perfume.similitud || '0').match(/(\d+(\.\d+)?)/);
          const similarity = similarityMatch ? parseFloat(similarityMatch[1]) : 0;
          console.log(`📊 Perfume: ${perfume.perfume}, Similitud: ${similarity}%`);
          return similarity >= similarityThreshold;
        });
        console.log(`🎯 Clones filtrados (≥${similarityThreshold}%):`, filtered.length);
        return filtered;
      }
      
      return data.similares || [];
    } catch (error) {
      console.error('❌ Error en searchClones:', error);
      throw error;
    }
  },

  // --- BÚSQUEDA DE SIMILARES ---
  searchSimilar: async (perfumeName, limit = 10) => {
    try {
      const url = `${API_BASE_URL}/perfumes/similares?nombre=${encodeURIComponent(perfumeName)}&n=${limit}`;
      console.log('🌐 URL de similares:', url);
      
      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Datos de similares recibidos:', data);
      return data.similares || [];
    } catch (error) {
      console.error('❌ Error en searchSimilar:', error);
      throw error;
    }
  },

  // --- BÚSQUEDA GENERAL DE PERFUMES ---
  searchPerfumes: async (query, filters = {}) => {
    try {
      const params = new URLSearchParams();
      
      console.log('🔍 Parámetros de búsqueda recibidos:');
      console.log('- Query:', query);
      console.log('- Filtros:', filters);
      
      // Agregar parámetros de búsqueda (usar espacios - el backend los convertirá)
      if (query) {
        params.append('perfume', query);
      } else if (filters.perfume) {
        params.append('perfume', filters.perfume);
      }
      
      if (filters.marca) params.append('marca', filters.marca);
      if (filters.genero) params.append('genero', filters.genero);
      if (filters.nota) params.append('nota', filters.nota);
      if (filters.acorde) params.append('acorde', filters.acorde);
      if (filters.perfumista) params.append('perfumista', filters.perfumista);
      
      const url = `${API_BASE_URL}/perfumes/search?${params}`;
      console.log('🌐 URL de búsqueda:', url);

      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Datos de búsqueda recibidos:', data);
      console.log('📊 Número de resultados:', data.resultados?.length || 0);
      
      return data.resultados || [];
    } catch (error) {
      console.error('❌ Error en searchPerfumes:', error);
      throw error;
    }
  },

  // --- BÚSQUEDA POR CELEBRIDAD ---
  searchByCelebrity: async (celebrityName) => {
    try {
      console.log('🌟 Buscando celebridad:', celebrityName);
      
      if (!celebrityName || !celebrityName.trim()) {
        console.log('⚠️ Nombre de celebridad vacío');
        return [];
      }

      // Cargar la base de datos
      const celebritiesDB = await loadCelebritiesDB();
      
      if (!celebritiesDB || !Array.isArray(celebritiesDB)) {
        console.error('❌ Base de datos no disponible');
        return [];
      }

      console.log(`📊 Buscando "${celebrityName}" entre ${celebritiesDB.length} celebridades...`);

      // Búsqueda case-insensitive
      const searchTerm = celebrityName.toLowerCase().trim();
      const results = celebritiesDB.filter(celebrity => {
        if (!celebrity || !celebrity.celebrity) return false;
        return celebrity.celebrity.toLowerCase().includes(searchTerm);
      });

      console.log('✅ Resultados encontrados:', results.length);
      
      if (results.length === 0) {
        console.log(`🔍 No se encontró "${celebrityName}"`);
      } else {
        results.forEach(result => {
          console.log(`🎯 Encontrado: ${result.celebrity} - ${result.perfumes?.length || 0} perfumes`);
        });
      }

      return results;

    } catch (error) {
      console.error('❌ Error en searchByCelebrity:', error);
      return [];
    }
  },

  // --- OBTENER TODAS LAS MARCAS ---
  getBrands: async () => {
    try {
      const url = `${API_BASE_URL}/perfumes/marcas`;
      console.log('🌐 URL de marcas:', url);
      
      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Marcas recibidas:', data);
      return data.marcas || [];
    } catch (error) {
      console.error('❌ Error en getBrands:', error);
      throw error;
    }
  },

  // --- OBTENER PERFUMES POR MARCA ---
  getPerfumesByBrand: async (brandName, page = 1, perPage = 20) => {
    try {
      // Usar el nombre con espacios - el backend lo convertirá a guiones
      const url = `${API_BASE_URL}/perfumes/marca/${encodeURIComponent(brandName)}?pagina=${page}&por_pagina=${perPage}`;
      console.log('🌐 URL de perfumes por marca:', url);
      
      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Perfumes por marca recibidos:', data);
      return data.perfumes || [];
    } catch (error) {
      console.error('❌ Error en getPerfumesByBrand:', error);
      throw error;
    }
  },

  // --- OBTENER LISTA PAGINADA DE PERFUMES ---
  getPerfumes: async (page = 1, perPage = 20) => {
    try {
      const url = `${API_BASE_URL}/perfumes?pagina=${page}&por_pagina=${perPage}`;
      console.log('🌐 URL de perfumes:', url);
      
      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Datos de perfumes recibidos:', data);
      return data.perfumes || [];
    } catch (error) {
      console.error('❌ Error en getPerfumes:', error);
      throw error;
    }
  },

  // --- OBTENER DETALLES DE UN PERFUME ---
  getPerfumeDetails: async (perfumeId) => {
    try {
      const url = `${API_BASE_URL}/perfumes/${perfumeId}`;
      console.log('🌐 URL de detalles:', url);
      
      const data = await fetchWithHyphenReplacement(url);
      console.log('✅ Datos de detalles recibidos:', data);
      return data;
    } catch (error) {
      console.error('❌ Error en getPerfumeDetails:', error);
      throw error;
    }
  }
};