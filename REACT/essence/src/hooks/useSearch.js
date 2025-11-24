import { useState } from 'react';
import { perfumeAPI } from '../services/api';

export const useSearch = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchType, setSearchType] = useState('general');

  // --- BÚSQUEDA GENERAL DE PERFUMES ---
  const searchPerfumes = async (query, filters = {}) => {
    // Si no hay criterios de búsqueda, mantener array vacío y salir
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

  // --- BÚSQUEDA DE SIMILARES ---
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

  // --- BÚSQUEDA DE CLONES ---
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

  // --- BÚSQUEDA DE CELEBRITIES (NUEVO) ---
  const searchByCelebrity = async (celebrityName) => {
    // Evitar búsquedas vacías
    if (!celebrityName || !celebrityName.trim()) return;

    setLoading(true);
    setError(null);
    setSearchType('celebrities');
    
    try {
      const results = await perfumeAPI.searchByCelebrity(celebrityName);
      
      // MEJORA: Si no hay resultados, activamos el error para que la UI avise al usuario
      if (!results || results.length === 0) {
        setError(`No encontramos a "${celebrityName}" en nuestra base de datos.`);
        setSearchResults([]);
      } else {
        setSearchResults(results);
      }
    } catch (err) {
      console.error("Error buscando celebrity:", err);
      setError("Ocurrió un error al intentar buscar el famoso.");
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  // --- LIMPIAR RESULTADOS ---
  const clearResults = () => {
    setSearchResults([]);
    setError(null);
    setSearchType('general');
    setLoading(false);
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