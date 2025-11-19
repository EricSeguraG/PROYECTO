const API_BASE_URL = 'http://localhost:5000';

export const perfumeAPI = {
  // Buscar clones/inspiraciones
  searchClones: async (perfumeName, similarityThreshold = 70) => {
    try {
      const url = `${API_BASE_URL}/perfumes/similares?nombre=${encodeURIComponent(perfumeName)}`;
      console.log('🌐 URL de clones:', url);
      
      const response = await fetch(url);
      console.log('📡 Estado de respuesta:', response.status);
      
      if (!response.ok) {
        // Intentar leer el error como texto primero
        const errorText = await response.text();
        console.error('❌ Error response text:', errorText);
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
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

  // Buscar perfumes similares
  searchSimilar: async (perfumeName, limit = 10) => {
    try {
      const url = `${API_BASE_URL}/perfumes/similares?nombre=${encodeURIComponent(perfumeName)}&n=${limit}`;
      console.log('🌐 URL de similares:', url);
      
      const response = await fetch(url);
      console.log('📡 Estado de respuesta similares:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response similares:', errorText);
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('✅ Datos de similares recibidos:', data);
      return data.similares || [];
    } catch (error) {
      console.error('❌ Error en searchSimilar:', error);
      throw error;
    }
  },

  // Buscar perfumes
  searchPerfumes: async (query, filters = {}) => {
    try {
      const params = new URLSearchParams();
      
      console.log('🔍 Parámetros de búsqueda recibidos:');
      console.log('- Query:', query);
      console.log('- Filtros:', filters);
      
      // Agregar parámetros de búsqueda
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

      const response = await fetch(url);
      console.log('📡 Estado de respuesta búsqueda:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response búsqueda:', errorText);
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('✅ Datos de búsqueda recibidos:', data);
      console.log('📊 Número de resultados:', data.resultados?.length || 0);
      
      return data.resultados || [];
    } catch (error) {
      console.error('❌ Error en searchPerfumes:', error);
      throw error;
    }
  },

  // Obtener lista paginada de perfumes
  getPerfumes: async (page = 1, perPage = 20) => {
    try {
      const url = `${API_BASE_URL}/perfumes?pagina=${page}&por_pagina=${perPage}`;
      console.log('🌐 URL de perfumes:', url);
      
      const response = await fetch(url);
      console.log('📡 Estado de respuesta perfumes:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response perfumes:', errorText);
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('✅ Datos de perfumes recibidos:', data);
      return data.perfumes || [];
    } catch (error) {
      console.error('❌ Error en getPerfumes:', error);
      throw error;
    }
  },

  // Obtener detalles de un perfume (si tu API lo soporta)
  getPerfumeDetails: async (perfumeId) => {
    try {
      const url = `${API_BASE_URL}/perfumes/${perfumeId}`;
      console.log('🌐 URL de detalles:', url);
      
      const response = await fetch(url);
      console.log('📡 Estado de respuesta detalles:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response detalles:', errorText);
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('✅ Datos de detalles recibidos:', data);
      return data;
    } catch (error) {
      console.error('❌ Error en getPerfumeDetails:', error);
      throw error;
    }
  }
};