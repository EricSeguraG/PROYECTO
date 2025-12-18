// src/hooks/useClones.js
import { useState } from 'react';
import { perfumeAPI } from '../services/api';

export const useClones = () => {
  const [clonesResults, setClonesResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPerfume, setSelectedPerfume] = useState(null);

  const searchClones = async (perfumeName, similarityThreshold = 70) => {
    if (!perfumeName.trim()) {
      setError("Por favor ingresa un nombre de perfume");
      return;
    }

    setLoading(true);
    setError(null);
    setSelectedPerfume(perfumeName);
    
    try {
      console.log('🔍 useClones: Iniciando búsqueda con:', {
        perfumeName,
        similarityThreshold
      });
      
      const results = await perfumeAPI.searchClones(perfumeName, similarityThreshold);
      
      console.log('✅ useClones: Resultados procesados:', {
        cantidad: results.length,
        primerResultado: results[0] ? {
          nombre: results[0].perfume,
          similitud: results[0].similitud
        } : 'No hay resultados'
      });
      
      setClonesResults(results);
    } catch (err) {
      console.error('❌ Error en useClones:', err);
      setError(`Error al buscar clones: ${err.message}`);
      setClonesResults([]);
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setClonesResults([]);
    setError(null);
    setSelectedPerfume(null);
  };

  return {
    clonesResults,
    loading,
    error,
    selectedPerfume,
    searchClones,
    clearResults
  };
};