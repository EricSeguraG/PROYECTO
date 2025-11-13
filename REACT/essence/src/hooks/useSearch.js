import { useState, useEffect } from 'react';
import { perfumeAPI } from '../services/api';

export const useSearch = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchType, setSearchType] = useState('general'); // general, clones, celebrities

  const searchPerfumes = async (query, filters = {}) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const results = await perfumeAPI.searchPerfumes(query, filters);
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
    searchClones,
    searchByCelebrity,
    clearResults
  };
};