const API_BASE_URL = 'http://localhost:5000';

export const perfumeAPI = {
  // Buscar perfumes con los parámetros de tu API Flask
  searchPerfumes: async (query, filters = {}) => {
    try {
      const params = new URLSearchParams();
      
      // Agregar parámetros de búsqueda según tu API Flask
      // Priorizar la búsqueda principal (query) sobre el filtro de perfume
      if (query) {
        params.append('perfume', query);
      } else if (filters.perfume) {
        params.append('perfume', filters.perfume);
      }
      
      // Agregar los demás filtros
      if (filters.marca) params.append('marca', filters.marca);
      if (filters.genero) params.append('genero', filters.genero);
      if (filters.nota) params.append('nota', filters.nota);
      if (filters.acorde) params.append('acorde', filters.acorde);
      if (filters.perfumista) params.append('perfumista', filters.perfumista);
      
      const url = `${API_BASE_URL}/perfumes/search?${params}`;
      console.log('Buscando en:', url); // Para debug

      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Error en la búsqueda');
      }
      
      const data = await response.json();
      return data.resultados || [];
    } catch (error) {
      console.error('Error searching perfumes:', error);
      throw error;
    }
  },

  // ... (el resto del código permanece igual)
};