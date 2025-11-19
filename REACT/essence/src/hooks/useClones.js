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
      console.log('🔍 Buscando clones para:', perfumeName);
      const results = await perfumeAPI.searchClones(perfumeName, similarityThreshold);
      console.log('✅ Resultados recibidos:', results);
      setClonesResults(results);
    } catch (err) {
      console.error('❌ Error en searchClones:', err);
      setError(`Error: ${err.message}`);
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