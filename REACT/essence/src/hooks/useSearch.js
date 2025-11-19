import { useState } from 'react'; // Quita useEffect
import { perfumeAPI } from '../services/api';

export const useSearch = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchType, setSearchType] = useState('general');

  // Quita el useEffect que carga perfumes iniciales

  const searchPerfumes = async (query, filters = {}) => {
    // Si no hay criterios de búsqueda, mantener array vacío
    if (!query.trim() && 
        !filters.perfume && 
        !filters.marca && 
        !filters.genero && 
        !filters.nota && 
        !filters.acorde && 
        !filters.perfumista) {
      setSearchResults([]);
      return;
    }

    setLoading(true);
    setError(null);
    setSearchType('general');
    
    try {
      console.log('🔍 Ejecutando búsqueda con:', { query, filters });
      const results = await perfumeAPI.searchPerfumes(query, filters);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  // ... el resto de las funciones se mantienen igual
  const searchSimilar = async (perfumeName) => {
    setLoading(true);
    setError(null);
    setSearchType('general');
    
    try {
      console.log('🔍 Buscando similares para:', perfumeName);
      const results = await perfumeAPI.searchSimilar(perfumeName);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const searchClones = async (originalPerfumeId) => {
    setLoading(true);
    setError(null);
    setSearchType('clones');
    
    try {
      const results = await perfumeAPI.searchClones(originalPerfumeId);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const searchByCelebrity = async (celebrityName) => {
    setLoading(true);
    setError(null);
    setSearchType('celebrities');
    
    try {
      const results = await perfumeAPI.searchByCelebrity(celebrityName);
      setSearchResults(results);
    } catch (err) {
      setError(err.message);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setSearchResults([]);
    setError(null);
    setSearchType('general');
  };

  return {
    searchResults,
    loading,
    error,
    searchType,
    searchPerfumes,
    searchSimilar,
    searchClones,
    searchByCelebrity,
    clearResults
  };
};